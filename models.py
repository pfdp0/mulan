# MIT License
# Copyright (c) 2026 MULAN authors
# Inspired by BarlowTwins (https://github.com/facebookresearch/barlowtwins).

import argparse
from typing import Tuple, Optional, Dict
import warnings

import torch
from torch import nn, Tensor

import resnet
import resnet_jaxlike
import vision_transformer as vits


activation_map = {
    "relu": nn.ReLU(inplace=True),
    "leaky_relu": nn.LeakyReLU(inplace=True),
    "sigmoid": nn.Sigmoid(),
    "tanh": nn.Tanh(),
    "gelu": nn.GELU(),
}


def get_mlp(mlp_str: str, input_dim: str | int, activation: str = "relu", use_output_bn: bool = False) -> nn.Sequential:
    assert activation in activation_map, f"Unknown activation function: {activation}"
    mlp_spec = f"{input_dim}-{mlp_str}"
    layers = []
    f = list(map(int, mlp_spec.split("-")))
    for i in range(len(f) - 2):
        layers.append(nn.Linear(f[i], f[i + 1], bias=False))
        layers.append(nn.BatchNorm1d(f[i + 1]))
        layers.append(activation_map[activation])
    layers.append(nn.Linear(f[-2], f[-1], bias=True))  # BYOL does not set bias=False
    if use_output_bn:
        layers.append(nn.BatchNorm1d(f[-1], affine=False))  # no learnable parameters, like in SimSiam
    return nn.Sequential(*layers)


def get_backbone(args: argparse.Namespace, is_target: bool = False) -> Tuple[nn.Module, int]:
    if args.arch in vits.__dict__.keys():
        if args.drop_path_rate > 0. and not is_target:
            print(f"Using {args.drop_path_rate} drop path rate.")
        drop_path_rate = 0. if is_target else args.drop_path_rate  # stochastic depth only for the online network
        backbone = vits.__dict__[args.arch](patch_size=args.patch_size,
                                            stop_grad_conv1=args.stop_grad_conv1,
                                            drop_path_rate=drop_path_rate)
        embedding_dim = backbone.embed_dim
    elif args.arch in resnet.__dict__.keys():
        if args.use_jax_resnet:
            warnings.warn("Using JAX-like ResNet implementation (only meant for evaluation of JAX pre-trained models).")
            backbone, embedding_dim = resnet_jaxlike.__dict__[args.arch]()
        else:
            backbone, embedding_dim = resnet.__dict__[args.arch]()
    else:
        raise ValueError(f"Unknown architecture: {args.arch}")

    return backbone, embedding_dim


class SSLNetwork(nn.Module):
    def __init__(self, args: argparse.Namespace, is_target: bool = False):
        super().__init__()
        self.num_features = int(args.projector_mlp.split("-")[-1])
        self.args = args

        self.backbone, self.backbone_out_dim = get_backbone(args, is_target=is_target)

        self.projector = get_mlp(args.projector_mlp, self.backbone_out_dim, activation=args.projector_activ,
                                 use_output_bn=args.use_proj_output_bn)

    def forward(self, images: Tensor) -> Dict[str, Tensor]:
        """
        Forward pass of the network.
        :param images: Input images tensor of shape (N, C, H, W)
        """
        h = self.backbone(images)
        z = self.projector(h)

        return {
            "backbone_feats": h,
            "embeddings": z
        }


class BYOLNetwork(SSLNetwork):
    """
    Siamese Networks composed of a backbone, a projector, and a predictor as used in BYOL, SimSiam, and related methods.
    """
    def __init__(self, args: argparse.Namespace, is_target: bool = False):
        super().__init__(args, is_target=is_target)

        pred_out_dim = int(args.predictor_mlp.split("-")[-1])
        assert self.num_features == pred_out_dim, \
            f"Projector and predictor output dimensions must match, got {self.num_features} and {pred_out_dim}"

        self.predictor = get_mlp(args.predictor_mlp, self.num_features, activation=args.predictor_activ,
                                 use_output_bn=args.use_pred_output_bn)

    def forward(self, images: Tensor, view_type: Optional[str] = None) -> Dict[str, Tensor]:
        """
        Forward pass of the BYOL network.
        :param images: Input images tensor of shape (N, C, H, W)
        :param view_type: Optional view type, not used in the base BYOLNetwork.
        """
        h = self.backbone(images)
        z = self.projector(h)
        pred = self.predictor(z)

        return {
            "backbone_feats": h,
            "embeddings": z,
            "predictions": pred
        }


class BYOLChimeraNetwork(SSLNetwork):
    """
    Multi-headed Siamese Networks composed of a backbone, a projector and multiple predictors.
    """
    def __init__(self, args: argparse.Namespace, is_target: bool = False):
        super().__init__(args, is_target=is_target)

        pred_out_dim = int(args.predictor_mlp.split("-")[-1])
        assert self.num_features == pred_out_dim, \
            f"Projector and predictor output dimensions must match, got {self.num_features} and {pred_out_dim}"

        predictors = {"identity": nn.Identity()}  # dummy predictor when no prediction is needed
        for task in args.tasks:
            predictors[task] = get_mlp(args.predictor_mlp, self.num_features, activation=args.predictor_activ,
                                       use_output_bn=args.use_pred_output_bn)
        self.predictors = nn.ModuleDict(predictors)

    def forward(self, images: Tensor, view_type: str = "global") -> Dict[str, Tensor]:
        """
        Forward pass of the BYOL chimera network.
        :param images: Input images tensor of shape (N, C, H, W)
        :param view_type: Type of the view, used to select the appropriate predictor.
        """
        assert view_type in self.predictors.keys(), \
            f"view type should be one of {self.predictors.keys()}, but got {view_type}"

        h = self.backbone(images)
        z = self.projector(h)
        pred = self.predictors[view_type](z)

        return {
            "backbone_feats": h,
            "embeddings": z,
            "predictions": pred
        }
