# MIT License
# Copyright (c) 2026 MULAN authors
# Inspired by BarlowTwins (https://github.com/facebookresearch/barlowtwins).

from pathlib import Path
import argparse
import json
import os
import sys
import warnings
import math

import torch
from torch import nn
import torchvision.datasets as datasets
import wandb

import config as cfg
import augmentations as aug
import utils
from distributed import init_distributed_mode, is_main_process
from models import BYOLNetwork, BYOLChimeraNetwork
from algorithms import get_algorithm
from eval_knn import evaluate_knn
from optim import get_optimizer, adjust_learning_rate, adjust_learning_rate_per_epoch


def get_arguments():
    parser = argparse.ArgumentParser(description="Pretrain a Siamese asymmetric SSL model", add_help=False)

    # Model
    parser.add_argument("--arch", type=str, default="resnet50",
                        help='Architecture of the backbone encoder network')
    parser.add_argument("--patch-size", type=int, default=16,
                        help='Patch size for ViT (ignored for ResNet)')
    parser.add_argument("--drop-path-rate", type=float, default=0.0,
                        help='Drop path rate (for ViT)')
    parser.add_argument("--projector-mlp", default="4096-256",
                        help='Size and number of layers of the MLP projector head')
    parser.add_argument("--predictor-mlp", default="4096-256",
                        help='Size and number of layers of the MLP predictor head')
    parser.add_argument("--projector-activ", type=str, default="relu",
                        help='Activation function for the projector MLP (default: relu)')
    parser.add_argument("--predictor-activ", type=str, default="relu",
                        help='Activation function for the predictor MLP (default: relu)')
    parser.add_argument("--single-pred-head", action='store_true',
                        help='Use a single prediction head for all views (for multitask algorithm)')
    parser.add_argument("--use-jax-resnet", action="store_true",
                        help="use JAX ResNet50 implementation (for models converted from JAX)")

    # SimSiam & MoCo v3 specific parameters
    parser.add_argument('--use-proj-output-bn', action='store_true',
                        help='use batch-norm at the output of the projector (default in SimSiam & MoCov3)')
    parser.add_argument('--use-pred-output-bn', action='store_true',
                        help='use batch-norm at the output of the projector (default in MoCov3)')
    parser.add_argument('--use-constant-pred-lr', action='store_true',
                        help='use a constant learning rate for the predictor (default in SimSiam)')
    parser.add_argument('--use-per-epoch-sched', action='store_true',
                        help='update LR (and WD) only once per epoch (default in SimSiam)')
    parser.add_argument('--moco-loss-temp', type=float, default=1.0)
    parser.add_argument('--stop-grad-conv1', action='store_true',
                        help='stop-grad after first conv, or patch embedding (for ViT)')

    # Training
    parser.add_argument("--training-fn", type=str, default="byol_default",
                        help='Training function to use for the model')
    parser.add_argument("--tasks", nargs='*', type=str, default=['global'])
    parser.add_argument("--num-views-per-task", type=int, nargs='*', default=[2])
    parser.add_argument("--task-weights", type=float, nargs='*', default=[1.0],
                        help='Weights for each task (only used for multitask algorithm)')

    # Optimization
    parser.add_argument("--epochs", type=int, default=200,
                        help='Number of epochs')
    parser.add_argument("--batch-size", type=int, default=1024,
                        help='Effective batch size (per worker batch size is [batch-size] / world-size)')
    parser.add_argument("--base-target-ema", type=float, default=0.996,
                        help='Base target EMA for the momentum encoder')
    parser.add_argument('--optimizer', default='lars', type=str, choices=['adamw', 'lars', 'sgd'],
                        help='Optimizer to use for training')
    parser.add_argument("--base-lr", type=float, default=0.4,
                        help='Base learning rate, effective learning after warmup is [base-lr] * [batch-size] / 256')
    parser.add_argument("--base-lr-biases", type=float, default=None,
                        help='Base learning rate for biases, if None, use base-lr')
    parser.add_argument("--num-warmup-epochs", type=int, default=10)
    parser.add_argument("--wd", type=float, default=1.5e-6,
                        help='Weight decay')
    parser.add_argument("--wd-end", type=float, default=None,
                        help='Weight decay at the end of cosine schedule (if None, use [wd])')
    parser.add_argument("--clip-grad", type=float, default=0,
                        help='Gradient clipping threshold (0 for no clipping)')

    # Data augmentations
    parser.add_argument("--transform", type=str, default="byol",
                        help='Type of data augmentation to use (byol, simsiam, moco, multitask)')
    parser.add_argument("--min-loc-scale", type=float, default=0.08,
                        help='Min local scale in the multi-task augmentations')
    parser.add_argument("--max-loc-scale", type=float, default=0.25,
                        help='Max local scale in the multi-task augmentations')
    parser.add_argument("--min-glob-scale", type=float, default=0.25,
                        help='Min global scale in the multi-task augmentations')
    parser.add_argument("--loc-crop-size", type=int, default=96,
                        help="Size of the local crops (in pixels)")
    parser.add_argument("--cutout-range", default=(0.2, 0.4), type=float, nargs=2,
                        help='Range of the cutout size (in fraction of the image area)')

    # kNN evaluation parameters
    parser.add_argument("--knn-k", default=20, type=int, metavar="K", help="number of nearest neighbors for KNN")
    parser.add_argument("--knn-model-choice", type=str, default="online_model",
                        help="which model to use for kNN evaluation when multiple are present in the checkpoint")
    parser.add_argument("--knn-model-mode", type=str, default="eval",
                        help="which model mode to use for kNN evaluation, either 'eval' or 'train'")

    # Logging and checkpointing
    parser.add_argument('--experiment-name', default='default', type=str, help='name of the experiment')
    parser.add_argument('--resume', default='auto', type=str,
                        help='resume from checkpoint (\"auto\", path or \"\" for no resume)')
    parser.add_argument('--no-wandb', action='store_true', help='disable wandb logging')
    parser.add_argument("--log-freq", type=int, default=20,
                        help='Print logs to the stats.txt file every [log-freq] steps')
    parser.add_argument("--knn-eval-freq", type=int, default=10,
                        help='Run kNN evaluation every [knn-eval-freq] epochs')
    parser.add_argument("--save-freq", type=int, default=-1,
                        help='Save additional checkpoints every [save-freq] epochs')

    # Distributed training and performance
    parser.add_argument("--num-workers", type=int, default=12)
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--compile', action='store_true',
                        help='use torch.compile() to compile the training loop')
    parser.add_argument('--sync-target-bn', action='store_true')
    parser.add_argument('--no-sync-online-bn', action='store_true')
    parser.add_argument('--fp16', action=argparse.BooleanOptionalAction, default=True,
                        help='mixed precision training (use --no-fp16 to disable)')

    return parser


class ReturnIndexDataset(datasets.ImageFolder):
    def __getitem__(self, idx):
        img, lab = super(ReturnIndexDataset, self).__getitem__(idx)
        return img, idx


def main(args):
    torch.backends.cudnn.benchmark = True
    init_distributed_mode(args)
    torch.set_num_threads(2)

    print(args)

    if is_main_process():
        stats_file = open(args.output_dir / "stats.txt", "a", buffering=1)
        print(" ".join(sys.argv))
        print(" ".join(sys.argv), file=stats_file)
        if not args.no_wandb:
            wandb.init(
                project=cfg.WANDB_PROJECT, entity=cfg.WANDB_ENTITY,
                name=args.experiment_name,
                config=vars(args),
                dir=args.output_dir,
                resume=True if args.resume == 'auto' else False
            )

    transforms = aug.get_train_transforms(args)

    print(f"Loading dataset from {cfg.IMAGENET_ROOT_DIR}")
    dataset = ReturnIndexDataset(Path(cfg.IMAGENET_ROOT_DIR) / "train", transforms)
    if args.distributed:
        sampler = torch.utils.data.DistributedSampler(dataset, shuffle=True)
    else:
        sampler = torch.utils.data.RandomSampler(dataset)
    if args.distributed:
        assert args.batch_size % args.world_size == 0
        per_device_batch_size = args.batch_size // args.world_size
        print(f"Using distributed training with {args.world_size} workers, {per_device_batch_size = }")
    else:
        per_device_batch_size = args.batch_size
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=per_device_batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        sampler=sampler,
    )

    # create model
    if "multitask" in args.training_fn and not args.single_pred_head:
        print(f"Using BYOL network with {len(args.tasks)} prediction heads!")
        online_model = BYOLChimeraNetwork(args, is_target=False)
        target_model = BYOLChimeraNetwork(args, is_target=True)
    else:
        online_model = BYOLNetwork(args, is_target=False)
        target_model = BYOLNetwork(args, is_target=True)

    print("Online model:")
    print(online_model)
    print("Target model:")
    print(target_model)

    # initialize target model with online model weights
    target_model.load_state_dict(online_model.state_dict(), strict=True)

    online_model = online_model.to(args.device)
    target_model = target_model.to(args.device)

    if args.distributed:
        if not args.no_sync_online_bn:
            print("SyncBN on online model")
            online_model = nn.SyncBatchNorm.convert_sync_batchnorm(online_model)
        if args.sync_target_bn:
            print("SyncBN on target model")
            target_model = nn.SyncBatchNorm.convert_sync_batchnorm(target_model)

        broadcast_buffers = (not args.no_sync_online_bn or args.sync_target_bn)
        if not broadcast_buffers:
            warnings.warn("Disabling DDP buffers broadcasting, may have a negative impact at EVAL.")
        online_model = torch.nn.parallel.DistributedDataParallel(
            online_model,
            device_ids=[args.gpu],
            broadcast_buffers=broadcast_buffers,
            find_unused_parameters=("multitask" in args.training_fn and not args.single_pred_head)  # needed for multitask
        )
        if args.sync_target_bn:
            # DDP is required for SyncBN to work (and broadcast_buffers=False is safe in this case)
            target_model = torch.nn.parallel.DistributedDataParallel(
                target_model,
                device_ids=[args.gpu],
                broadcast_buffers=False
            )

    # freeze after DDP to avoid issues
    target_model.requires_grad_(False)

    if args.compile:
        # compile models only if not in DDP wrapper
        if not args.distributed:
            print("Compiling online model with online_model.backbone.compile(mode='default')")
            online_model.backbone.compile(mode='default')
        if not args.sync_target_bn:
            print("Compiling target model with: target_model.backbone.compile(mode='max-autotune')")
            target_model.backbone.compile(mode='max-autotune')

    # create optimizer and scaler
    optimizer = get_optimizer(args, online_model)
    scaler = torch.amp.GradScaler(enabled=args.fp16)

    if args.use_per_epoch_sched:
        print("Updating learning rate once an epoch instead of every step")

    print(f"Starting training for {args.epochs} epochs, mixed precision: {args.fp16}")
    # resume from checkpoint if needed
    start_epoch = 0
    if args.resume:  # and is_main_process():
        resume_path = Path(args.output_dir) / f'checkpoint.pth'
        if os.path.isfile(args.resume):  # manual resume
            resume_path = args.resume
        elif os.path.isfile(resume_path):  # auto resume from output_dir
            pass
        else:
            resume_path = None

        if resume_path is not None:
            checkpoint = torch.load(resume_path, map_location='cpu', weights_only=False)
            online_model.load_state_dict(checkpoint["online_model"])
            target_model.load_state_dict(checkpoint["target_model"])
            optimizer.load_state_dict(checkpoint["optimizer"])
            scaler.load_state_dict(checkpoint["scaler"])
            start_epoch = checkpoint['epoch'] + 1
            print("loaded checkpoint '{}' (epoch {})".format(resume_path, checkpoint['epoch']))

    # set training algorithm
    total_num_steps = args.epochs * len(loader)
    algorithm = get_algorithm(online_model, target_model, total_num_steps, args)

    online_model.train()
    target_model.train()
    for epoch in range(start_epoch, args.epochs):
        if args.distributed:
            sampler.set_epoch(epoch)

        num_steps = len(loader)
        metric_logger = utils.MetricLogger(delimiter="  ")
        for i, param_group in enumerate(optimizer.param_groups):
            metric_logger.add_meter(f'lr_group_{i}', utils.SmoothedValue(window_size=1, fmt='{value}'))
            metric_logger.add_meter(f'wd_group_{i}', utils.SmoothedValue(window_size=1, fmt='{value}'))
        header = 'Epoch: [{}]'.format(epoch)

        if args.use_per_epoch_sched:
            adjust_learning_rate_per_epoch(args, optimizer, epoch)

        for step, (images_tuple, _) in enumerate(metric_logger.log_every(loader, args.log_freq, header)):
            for i in range(len(images_tuple)):
                images_tuple[i] = images_tuple[i].to(args.device, non_blocking=True)

            # adjust the learning rate (and weight decay) based on the current step
            global_step = epoch * num_steps + step
            if not args.use_per_epoch_sched:
                adjust_learning_rate(args, optimizer, loader, global_step)

            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast('cuda', enabled=args.fp16):
                log_dict = algorithm.step(images_tuple, step=global_step, scaler=scaler)

            if not math.isfinite(log_dict['loss']):
                print(f"Loss is {log_dict['loss']}, stopping training")
                sys.exit(1)

            if args.clip_grad > 0:
                scaler.unscale_(optimizer)  # unscale for correct gradient clipping
                grad_norm = torch.nn.utils.clip_grad_norm_(online_model.parameters(), args.clip_grad)
            else:
                scaler.unscale_(optimizer)
                grad_norm = utils.compute_grad_norm(online_model.parameters())
            scaler.step(optimizer)
            scaler.update()

            # log lr, wd, grad_norm
            log_dict["grad_norm"] = grad_norm.item()
            for i, param_group in enumerate(optimizer.param_groups):
                log_dict[f"lr_group_{i}"] = param_group['lr']
                log_dict[f"wd_group_{i}"] = param_group['weight_decay']

            # update metric logger and write stats
            metric_logger.update(**log_dict)
            if is_main_process() and step % args.log_freq == 0:
                log_dict.update(epoch=epoch, global_step=global_step)
                print(json.dumps(log_dict), file=stats_file)
                if wandb.run is not None:
                    wandb.log(log_dict, step=global_step)

        if is_main_process():
            state = dict(
                epoch=epoch,
                online_model=online_model.state_dict(),
                target_model=target_model.state_dict(),
                optimizer=optimizer.state_dict(),
                scaler=scaler.state_dict()
            )
            torch.save(state, args.output_dir / "checkpoint.pth")
            if args.save_freq != -1 and (epoch + 1) % args.save_freq == 0:
                torch.save(state, args.output_dir / f"checkpoint_epoch_{epoch+1:0>4}.pth")

        # wait for process 0 to save the checkpoint
        if args.distributed:
            torch.distributed.barrier()

        # kNN evaluation (every knn_eval_freq epochs)
        if (epoch+1) % args.knn_eval_freq == 0:
            print("kNN evaluation...")
            args.pretrained = args.output_dir / "checkpoint.pth"
            acc1, acc5 = evaluate_knn(args, train_dataset=dataset)
            if is_main_process():
                log_dict = dict(acc1=acc1, acc5=acc5, epoch=epoch)
                print(json.dumps(log_dict), file=stats_file)
                if wandb.run is not None:
                    wandb.log(log_dict, step=(epoch+1) * num_steps)

    if is_main_process():
        torch.save(online_model.module.backbone.state_dict(), args.output_dir / "backbone.pth")


if __name__ == "__main__":
    parser = argparse.ArgumentParser('BYOL training script', parents=[get_arguments()])
    args = parser.parse_args()

    # create output dir
    args.output_dir = Path(cfg.EXP_ROOT) / args.experiment_name
    args.output_dir.mkdir(parents=True, exist_ok=True)

    main(args)
