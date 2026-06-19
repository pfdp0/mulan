# MIT License
# Copyright (c) 2026 MULAN authors
# Inspired by DINO (https://github.com/facebookresearch/dino)
#   and BarlowTwins (https://github.com/facebookresearch/barlowtwins).

from pathlib import Path
import argparse
import json
import os
import sys
import urllib

from torch import nn, optim
from torchvision import datasets, transforms
import torch
from torch.utils.data import DistributedSampler

import config as cfg
import utils
from models import get_backbone

import distributed as dist_utils


def get_arguments():
    parser = argparse.ArgumentParser(
        description="Evaluate a pretrained model on ImageNet"
    )

    parser.add_argument('--experiment-name', default='default', type=str, help='name of the experiment')
    parser.add_argument(
        "--print-freq", default=50, type=int, metavar="N", help="print frequency"
    )

    # Data
    parser.add_argument(
        "--train-percent",
        default=100,
        type=int,
        choices=(100, 10, 1),
        help="size of training set in percent",
    )

    # Checkpoint
    parser.add_argument("--pretrained", type=Path, help="path to pretrained model")

    # Model
    parser.add_argument("--arch", type=str, default="resnet50")
    parser.add_argument("--patch-size", type=int, default=16,
                        help='Patch size for ViT (ignored for ResNet)')
    parser.add_argument('--stop-grad-conv1', action='store_true',
                        help='stop-grad after first conv, or patch embedding')
    parser.add_argument('--n-last-blocks', default=1, type=int,
                        help="Concatenate [CLS] tokens for the `n` last blocks. "
                             "DINO uses `n=4` when evaluating ViT-Small and `n=1` with ViT-Base.")
    parser.add_argument('--avgpool_patchtokens', action='store_true',
                        help='use avgpool of patch tokens in addition to [CLS] token (default: False)')
    parser.add_argument('--drop-path-rate', type=float, default=0.0,
                        help='stochastic depth rate (not used for linear eval, set to 0)')
    parser.add_argument('--nesterov', action='store_true', help='use Nesterov momentum')

    parser.add_argument("--lin-model-choice", type=str, default="online_model",
                        help="which model to use for linear evaluation when multiple are present in the checkpoint")

    # Optim
    parser.add_argument(
        "--epochs",
        default=100,
        type=int,
        metavar="N",
        help="number of total epochs to run",
    )
    parser.add_argument(
        "--batch-size", default=256, type=int, metavar="N", help="mini-batch size"
    )
    parser.add_argument(
        "--lr-backbone",
        default=0.0,
        type=float,
        metavar="LR",
        help="backbone base learning rate",
    )
    parser.add_argument(
        "--lr-head",
        default=0.3,
        type=float,
        metavar="LR",
        help="classifier base learning rate",
    )
    parser.add_argument(
        "--wd", default=1e-6, type=float, metavar="W", help="weight decay"
    )
    parser.add_argument(
        "--weights",
        default="freeze",
        type=str,
        choices=("finetune", "freeze"),
        help="finetune or freeze resnet weights",
    )
    parser.add_argument(
        "--normalize-features",
        action="store_true",
        help="normalize features before linear evaluation",
    )

    # Running
    parser.add_argument(
        "--num-workers",
        default=10,
        type=int,
        metavar="N",
        help="number of data loader workers",
    )
    parser.add_argument('--dtype', default='fp16', choices=['fp16', 'fp32', 'bf16'],
                        help='data type to use for training (note: bf16 is only available on Ampere and later GPUs)')

    parser.add_argument("--use-jax-resnet", action="store_true",
                        help="use JAX ResNet50 implementation (for models converted from JAX)")

    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')

    return parser


def main():
    parser = get_arguments()
    args = parser.parse_args()

    torch.backends.cudnn.benchmark = True
    dist_utils.init_distributed_mode(args)
    torch.set_num_threads(2)

    DTYPE_MAP = {
        'fp32': torch.float32,
        'fp16': torch.float16,
        'bf16': torch.bfloat16,
    }
    amp_dtype = DTYPE_MAP[args.dtype]

    args.output_dir = Path(cfg.EXP_ROOT) / args.experiment_name
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if dist_utils.is_main_process():
        stats_file = open(args.output_dir / "stats.txt", "a", buffering=1)
        print(" ".join(sys.argv))
        print(" ".join(sys.argv), file=stats_file)

    if args.train_percent in {1, 10}:
        # download the semi-supervised train file list from SimCLR
        args.train_files = urllib.request.urlopen(
            f"https://raw.githubusercontent.com/google-research/simclr/master/imagenet_subsets/{args.train_percent}percent.txt"
        ).readlines()

    backbone, embedding_dim = get_backbone(args)
    if "vit" in args.arch:
        embedding_dim = embedding_dim * (args.n_last_blocks + int(args.avgpool_patchtokens))
        print(f"Using CLS tokens from the last {args.n_last_blocks} blocks, "
              f"{'and avgpool of patch tokens, ' if args.avgpool_patchtokens else ''}"
              f"effective embedding dimension is {embedding_dim}")
    state_dict = torch.load(args.pretrained, map_location="cpu", weights_only=False)
    if args.lin_model_choice in state_dict:
        print(f"Linear evaluation with {args.lin_model_choice}")
        state_dict = state_dict[args.lin_model_choice]
        state_dict = {
            key.replace("module.backbone.", ""): value
            for (key, value) in state_dict.items()
        }
    missing_keys, unexpected_keys = backbone.load_state_dict(state_dict, strict=False)
    if len(missing_keys) > 0:
        print(f"Missing keys in the state dict: {missing_keys}")
    if len(unexpected_keys) > 0:
        print(f"Unexpected keys in the state dict: {unexpected_keys}")

    head = nn.Linear(embedding_dim, 1000)
    head.weight.data.normal_(mean=0.0, std=0.01)
    head.bias.data.zero_()

    backbone = backbone.to(args.device)
    head = head.to(args.device)

    if args.weights == "freeze":
        backbone.requires_grad_(False)
        head.requires_grad_(True)

    if args.distributed:
        backbone = torch.nn.parallel.DistributedDataParallel(backbone, device_ids=[args.gpu],
                                                             output_device=args.gpu, find_unused_parameters=False)
        head = torch.nn.parallel.DistributedDataParallel(head, device_ids=[args.gpu], output_device=args.gpu,
                                                         find_unused_parameters=False)

    criterion = nn.CrossEntropyLoss().to(args.device)

    param_groups = [dict(params=head.parameters(), lr=args.lr_head)]
    if args.weights == "finetune":
        param_groups.append(dict(params=backbone.parameters(), lr=args.lr_backbone))
    optimizer = optim.SGD(param_groups, 0, momentum=0.9, weight_decay=args.wd, nesterov=args.nesterov)
    scaler = torch.amp.GradScaler(enabled=(args.dtype == 'fp16'))
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)

    # automatically resume from checkpoint if it exists
    resume_path = Path(args.output_dir) / f'checkpoint.pth'
    if os.path.isfile(resume_path):
        checkpoint = torch.load(resume_path, map_location="cpu", weights_only=False)
        start_epoch = checkpoint["epoch"]
        best_acc = checkpoint["best_acc"]
        backbone.load_state_dict(checkpoint["backbone"])
        head.load_state_dict(checkpoint["head"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        scaler.load_state_dict(checkpoint['scaler'])
    else:
        start_epoch = 0
        best_acc = argparse.Namespace(top1=0, top5=0)

    # Data loading code
    traindir = Path(cfg.IMAGENET_ROOT_DIR) / "train"
    valdir = Path(cfg.IMAGENET_ROOT_DIR) / "val"
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )

    train_dataset = datasets.ImageFolder(
        traindir,
        transforms.Compose(
            [
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ]
        ),
    )
    val_dataset = datasets.ImageFolder(
        valdir,
        transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize,
            ]
        ),
    )

    if args.train_percent in {1, 10}:
        train_dataset.samples = []
        for fname in args.train_files:
            fname = fname.decode().strip()
            cls = fname.split("_")[0]
            train_dataset.samples.append(
                (traindir / cls / fname, train_dataset.class_to_idx[cls])
            )

    if args.distributed:
        train_sampler = DistributedSampler(train_dataset, shuffle=True)
    else:
        train_sampler = torch.utils.data.RandomSampler(train_dataset)

    args.batch_size_per_gpu = args.batch_size // dist_utils.get_world_size()

    kwargs = dict(
        batch_size=args.batch_size_per_gpu,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset, sampler=train_sampler, **kwargs
    )
    val_loader = torch.utils.data.DataLoader(val_dataset, **kwargs)  # so far, each GPU has full validation set

    # estimate the mean and std of the features (per feature dimension)
    # TODO: adapt for DDP (if val set is split across GPUs, the estimated mean/std will be incorrect)
    if args.normalize_features:
        # compute the mean and std of features for normalization (optional; removes the need to tune LR)
        backbone.eval()
        features_all = []
        for i, (images, targets) in enumerate(val_loader):
            images = images.to(args.device, non_blocking=True)
            with torch.no_grad():
                if "vit" in args.arch:
                    intermediate_output = backbone.get_intermediate_layers(images, args.n_last_blocks)
                    feats = torch.cat([x[:, 0] for x in intermediate_output], dim=-1)
                    if args.avgpool_patchtokens:
                        feats = torch.cat((feats.unsqueeze(-1), torch.mean(intermediate_output[-1][:, 1:], dim=1).unsqueeze(-1)), dim=-1)
                        feats = feats.reshape(feats.shape[0], -1)
                else:
                    feats = backbone(images)
            features_all.append(feats.cpu())
        features_all = torch.cat(features_all, dim=0)

        features_mean = features_all.mean(dim=0).to(args.device)
        features_std = features_all.std(dim=0).to(args.device)
        print(f"Features mean: {features_mean.mean().item()} and std: {features_std.mean().item()}")
        del features_all  # discard the features
        print("Features will be normalized during linear evaluation")

    eval_type = f"Finetune {args.train_percent}%" if args.weights == "finetune" else f"Linear"

    num_steps = len(train_loader)

    print(f"{eval_type} evaluation with {args.dtype} dtype")
    for epoch in range(start_epoch, args.epochs):
        # set epoch for distributed sampler
        if args.distributed:
            train_sampler.set_epoch(epoch)

        # train
        if args.weights == "finetune":
            backbone.train()
            head.train()
        elif args.weights == "freeze":
            backbone.eval()
            head.train()
        else:
            assert False, f"Invalid weights argument: {args.weights}"

        metric_logger = utils.MetricLogger(delimiter="  ")
        metric_logger.add_meter('lr_backbone', utils.SmoothedValue(window_size=1, fmt='{value}'))
        metric_logger.add_meter('lr_head', utils.SmoothedValue(window_size=1, fmt='{value}'))
        header = f'{eval_type} EVAL epoch: [{epoch}]'

        for step, (images, targets) in enumerate(metric_logger.log_every(train_loader, args.print_freq, header)):
            global_step = epoch * num_steps + step
            images = images.to(args.device, non_blocking=True)
            targets = targets.to(args.device, non_blocking=True)

            optimizer.zero_grad()

            # forward pass backbone
            with torch.amp.autocast('cuda', enabled=(args.dtype != 'fp32'), dtype=amp_dtype):
                if "vit" in args.arch:
                    intermediate_output = backbone.get_intermediate_layers(images, args.n_last_blocks)
                    features = torch.cat([x[:, 0] for x in intermediate_output], dim=-1)
                    if args.avgpool_patchtokens:
                        features = torch.cat((features.unsqueeze(-1), torch.mean(intermediate_output[-1][:, 1:], dim=1).unsqueeze(-1)), dim=-1)
                        features = features.reshape(features.shape[0], -1)
                else:
                    features = backbone(images)

            # normalize features if specified
            if args.normalize_features:
                features = (features - features_mean) / (features_std[None, :] + 1e-8)  # normalize the features

            # forward pass head
            with torch.amp.autocast('cuda', enabled=(args.dtype != 'fp32'), dtype=amp_dtype):
                output = head(features)

            loss = criterion(output, targets)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            pg = optimizer.param_groups
            lr_head = pg[0]["lr"]
            lr_backbone = pg[1]["lr"] if len(pg) == 2 else 0
            stats = dict(loss=loss.item(), lr_backbone=lr_backbone, lr_head=lr_head)
            metric_logger.update(**stats)
            if step % args.print_freq == 0:
                stats.update(epoch=epoch, global_step=global_step)
                if dist_utils.is_main_process():
                    print(json.dumps(stats), file=stats_file)

        # evaluate
        backbone.eval()
        head.eval()
        top1 = AverageMeter("Acc@1")
        top5 = AverageMeter("Acc@5")
        with torch.no_grad():
            for images, targets in val_loader:
                images = images.to(args.device, non_blocking=True)
                targets = targets.to(args.device, non_blocking=True)

                if "vit" in args.arch:
                    intermediate_output = backbone.get_intermediate_layers(images, args.n_last_blocks)
                    features = torch.cat([x[:, 0] for x in intermediate_output], dim=-1)
                    if args.avgpool_patchtokens:
                        features = torch.cat((features.unsqueeze(-1), torch.mean(intermediate_output[-1][:, 1:], dim=1).unsqueeze(-1)), dim=-1)
                        features = features.reshape(features.shape[0], -1)
                else:
                    features = backbone(images)

                if args.normalize_features:
                    features = (features - features_mean) / (features_std[None, :] + 1e-8)  # normalize the features

                outputs = head(features)

                acc1, acc5 = accuracy(
                    outputs, targets, topk=(1, 5)
                )
                top1.update(acc1[0].item(), images.size(0))
                top5.update(acc5[0].item(), images.size(0))

        top1_avg = top1.avg
        top5_avg = top5.avg

        if dist_utils.is_main_process():
            best_acc.top1 = max(best_acc.top1, top1_avg)
            best_acc.top5 = max(best_acc.top5, top5_avg)

            stats = dict(
                epoch=epoch,
                acc1=top1_avg,
                acc5=top5_avg,
                best_acc1=best_acc.top1,
                best_acc5=best_acc.top5,
            )
            print(json.dumps(stats))
            print(json.dumps(stats), file=stats_file)

            state = dict(
                epoch=epoch + 1,
                best_acc=best_acc,
                backbone=backbone.state_dict(),
                head=head.state_dict(),
                optimizer=optimizer.state_dict(),
                scheduler=scheduler.state_dict(),
                scaler=scaler.state_dict()
            )
            torch.save(state, args.output_dir / "checkpoint.pth")

        scheduler.step()

    if args.distributed:
        torch.distributed.barrier()


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=":f"):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = "{name} {val" + self.fmt + "} ({avg" + self.fmt + "})"
        return fmtstr.format(**self.__dict__)


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


if __name__ == "__main__":
    main()
