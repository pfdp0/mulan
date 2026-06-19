# MIT License
# Copyright (c) 2026 MULAN authors
# Inspired by DINO (https://github.com/facebookresearch/dino)

from pathlib import Path
import argparse
import sys
from typing import Optional
import warnings

import torchvision.transforms.v2 as transforms_v2
import torch
from torch.amp import autocast
from torch.nn import functional as F
from torch import Tensor
from tqdm import tqdm

import config as cfg
from datasets import ReturnIndexDataset
from distributed import init_distributed_mode, is_main_process, get_world_size
from models import get_backbone


def get_arguments():
    parser = argparse.ArgumentParser(
        description="Evaluate a pretrained model on ImageNet"
    )

    parser.add_argument('--experiment-name', default='default', type=str, help='name of the experiment')
    parser.add_argument("--pretrained", type=Path, help="path to pretrained model")
    parser.add_argument("--arch", type=str, default="resnet50")
    parser.add_argument("--patch-size", type=int, default=16,
                        help='Patch size for ViT (ignored for ResNet)')
    parser.add_argument('--stop-grad-conv1', action='store_true',
                        help='stop-grad after first conv, or patch embedding')
    parser.add_argument("--drop-path-rate", type=float, default=0.0,
                        help="stochastic depth drop path rate (keep to 0.0 for kNN evaluation)")
    parser.add_argument("--knn-model-choice", type=str, default="online_model",
                        help="which model to use for kNN evaluation when multiple are present in the checkpoint")
    parser.add_argument("--knn-model-mode", type=str, default="eval",
                        help="which model mode to use for kNN evaluation, either 'eval' or 'train'")
    parser.add_argument(
        "--batch-size", default=256, type=int, metavar="N", help="mini-batch size"
    )
    parser.add_argument(
        "--num-workers",
        default=8,
        type=int,
        metavar="N",
        help="number of data loader workers",
    )

    parser.add_argument("--update-bn-stats", action="store_true",
                        help="update BatchNorm statistics before evaluation")

    parser.add_argument("--use-jax-resnet", action="store_true",
                        help="use JAX ResNet50 implementation (for models converted from JAX)")

    return parser


@torch.no_grad()
def extract_features(model, data_loader, use_cuda: bool = True, distributed: bool = False):
    features = None
    for samples, index in tqdm(data_loader, desc="Extracting features"):
        samples = samples.cuda(non_blocking=True)
        index = index.cuda(non_blocking=True)

        with autocast("cuda", dtype=torch.float16):
            output_feats = model(samples)
        feats = output_feats.clone().float()

        # init storage feature matrix
        if is_main_process() and features is None:
            features = torch.zeros(len(data_loader.dataset), feats.shape[-1])
            if use_cuda:
                features = features.cuda(non_blocking=True)
            print(f"Storing features into tensor of shape {features.shape}")

        # get indexes from all processes
        if distributed:
            y_all = torch.empty(get_world_size(), index.size(0), dtype=index.dtype, device=index.device)
            y_l = list(y_all.unbind(0))
            y_all_reduce = torch.distributed.all_gather(y_l, index, async_op=True)
            y_all_reduce.wait()
            index_all = torch.cat(y_l)

            # share features between processes
            feats_all = torch.empty(
                get_world_size(),
                feats.size(0),
                feats.size(1),
                dtype=feats.dtype,
                device=feats.device,
            )
            output_l = list(feats_all.unbind(0))
            output_all_reduce = torch.distributed.all_gather(output_l, feats, async_op=True)
            output_all_reduce.wait()
        else:
            index_all = index
            output_l = [feats]

        # update storage feature matrix
        if is_main_process():
            if use_cuda:
                features.index_copy_(0, index_all, torch.cat(output_l))
            else:
                features.index_copy_(0, index_all.cpu(), torch.cat(output_l).cpu())

    return features


@torch.no_grad()
def knn_classifier(train_features, train_labels, test_features, test_labels,
                   k, T, num_classes=1000, similarity_type: str = 'cosine'):
    assert similarity_type in ['euclidean', 'cosine'], "similarity_type must be either 'euclidean' or 'cosine'"
    top1, top5, total = 0.0, 0.0, 0
    if similarity_type == 'cosine':
        print("Using cosine similarity for KNN evaluation")
        train_features = F.normalize(train_features, dim=1, p=2)
        test_features = F.normalize(test_features, dim=1, p=2)
        train_features = train_features.t()
    else:
        print("Using L2 distance for KNN evaluation")
    num_test_images, num_chunks = test_labels.shape[0], 500
    imgs_per_chunk = num_test_images // num_chunks
    retrieval_one_hot = torch.zeros(k, num_classes).to(train_features.device)

    all_indices = torch.zeros(num_test_images, k, dtype=torch.int32).to(train_features.device)
    for idx in range(0, num_test_images, imgs_per_chunk):
        # get the features for test images
        features = test_features[
            idx: min((idx + imgs_per_chunk), num_test_images), :
        ]
        targets = test_labels[idx: min((idx + imgs_per_chunk), num_test_images)]
        batch_size = targets.shape[0]

        # calculate the dot product / distance and compute top-k neighbors
        if similarity_type == 'cosine':
            similarity = torch.mm(features, train_features)
        else:
            similarity = - torch.cdist(features.unsqueeze(0), train_features.unsqueeze(0), p=2).squeeze(0)
        distances, indices = similarity.topk(k, largest=True, sorted=True)
        candidates = train_labels.view(1, -1).expand(batch_size, -1)
        retrieved_neighbors = torch.gather(candidates, 1, indices)

        retrieval_one_hot.resize_(batch_size * k, num_classes).zero_()
        retrieval_one_hot.scatter_(1, retrieved_neighbors.view(-1, 1), 1)
        distances_transform = distances.clone().div_(T).exp_()
        probs = torch.sum(
            torch.mul(
                retrieval_one_hot.view(batch_size, -1, num_classes),
                distances_transform.view(batch_size, -1, 1),
            ),
            1,
        )
        _, predictions = probs.sort(1, True)

        # find the predictions that match the target
        correct = predictions.eq(targets.data.view(-1, 1))
        top1 = top1 + correct.narrow(1, 0, 1).sum().item()
        if num_classes > 5:
            top5 = top5 + correct.narrow(1, 0, min(5, k)).sum().item()  # top5 does not make sense if k < 5
        else:
            top5 = targets.size(0)  # top5 is always 100% if num_classes <= 5
        total += targets.size(0)

        all_indices[idx: min((idx + imgs_per_chunk), num_test_images), :] = indices

        if idx % (imgs_per_chunk * 100) == 0:
            print(f"Processed {idx}/{num_test_images} images")

    top1 = top1 * 100.0 / total
    top5 = top5 * 100.0 / total

    return top1, top5, all_indices


def evaluate_knn(args, train_dataset: Optional[ReturnIndexDataset] = None):
    backbone, embedding_dim = get_backbone(args)
    state_dict = torch.load(args.pretrained, map_location="cpu", weights_only=False)
    if args.knn_model_choice in state_dict:
        print(f"kNN evaluation with {args.knn_model_choice}")
        state_dict = state_dict[args.knn_model_choice]
        state_dict = {
            key.replace("module.", ""): value
            for (key, value) in state_dict.items()
        }
        state_dict = {
            key.replace("backbone.", ""): value
            for (key, value) in state_dict.items()
        }
        state_dict = {  # for moco v3 compatibility
            key.replace("base_encoder.", ""): value
            for (key, value) in state_dict.items()
        }
    load_res = backbone.load_state_dict(state_dict, strict=False)
    print("Missing keys:", load_res.missing_keys)
    print("Unexpected keys:", load_res.unexpected_keys)
    backbone.to('cuda')

    # Data loading code
    traindir = Path(cfg.IMAGENET_ROOT_DIR) / "train"
    valdir = Path(cfg.IMAGENET_ROOT_DIR) / "val"
    test_transforms = transforms_v2.Compose(
        [
            transforms_v2.ToImage(),  # ensures the image is a Tensor
            transforms_v2.Resize(256),
            transforms_v2.CenterCrop(224),
            transforms_v2.ToDtype(torch.float32, scale=True),  # values are scaled to [0, 1] here
            transforms_v2.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            )
        ]
    )

    old_transform = None
    if train_dataset is not None:
        # save transform and then edit
        old_transform = train_dataset.transform
        train_dataset.transform = test_transforms
    else:
        train_dataset = ReturnIndexDataset(traindir, test_transforms)
    val_dataset = ReturnIndexDataset(valdir, test_transforms)

    if args.distributed:
        per_device_batch_size = args.batch_size // args.world_size
    else:
        per_device_batch_size = args.batch_size
    kwargs = dict(
        batch_size=per_device_batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    if args.distributed:
        train_sampler = torch.utils.data.DistributedSampler(train_dataset, shuffle=False)
    else:
        train_sampler = torch.utils.data.RandomSampler(train_dataset)
    train_loader = torch.utils.data.DataLoader(train_dataset, sampler=train_sampler, **kwargs)
    val_loader = torch.utils.data.DataLoader(val_dataset, **kwargs)

    if args.knn_model_mode == "eval":
        backbone.eval()
    elif args.knn_model_mode == "train":
        warnings.warn("Setting backbone to train mode for kNN evaluation")
        backbone.train()
    else:
        raise ValueError(f"Unknown knn_model_mode: {args.knn_model_mode}")

    print("Extracting features for train set...")
    train_features = extract_features(backbone, train_loader, use_cuda=True, distributed=args.distributed)
    print("Extracting features for val set...")
    val_features = extract_features(backbone, val_loader, use_cuda=True, distributed=args.distributed)

    if args.distributed:
        torch.distributed.barrier()
    print("Done with extraction")

    if is_main_process():
        train_labels = torch.tensor(train_dataset.targets).to(train_features.device)
        val_labels = torch.tensor(val_dataset.targets).to(train_features.device)

        acc1_dict, acc5_dict = {}, {}
        for knn_k in [10, 20]:
            acc1_dict[knn_k], acc5_dict[knn_k], topk_indices = knn_classifier(
                train_features, train_labels,
                val_features, val_labels,
                k=knn_k, T=0.07,
                num_classes=len(train_dataset.classes),
                similarity_type="cosine"
            )

            print(f"{knn_k}-NN classifier result: Acc@1: {acc1_dict[knn_k]}%, Acc@5: {acc5_dict[knn_k]}%")

    if args.distributed:
        torch.distributed.barrier()

    del train_features, val_features

    if old_transform is not None:
        # set back transform
        train_dataset.transform = old_transform
        print(f"back to:\n\t{train_dataset.transform = }")

    if is_main_process():
        return acc1_dict, acc5_dict
    return -1, -1


def main():
    parser = get_arguments()
    args = parser.parse_args()

    args.output_dir = Path(cfg.EXP_ROOT) / args.experiment_name
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stats_file = open(args.output_dir / "stats.txt", "a", buffering=1)
    print(" ".join(sys.argv))
    print(" ".join(sys.argv), file=stats_file)

    torch.backends.cudnn.benchmark = True
    init_distributed_mode(args)

    acc1_dict, acc5_dict, = evaluate_knn(args)
    print(f"Done."
          f"\n\tAcc@1 dict: {acc1_dict}"
          f"\n\tAcc@5 dict: {acc5_dict}")
    if args.distributed:
        torch.distributed.destroy_process_group()


def exclude_bias_and_norm(p: Tensor):
    return p.ndim == 1


if __name__ == "__main__":
    main()
