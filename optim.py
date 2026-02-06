# MIT License
# Copyright (c) 2026 MULAN authors
# Inspired by BarlowTwins (https://github.com/facebookresearch/barlowtwins).

import math
import warnings

import torch
from torch import Tensor
from torch import optim


def exclude_bias_and_norm(p: Tensor):
    return p.ndim == 1


def adjust_learning_rate(args, optimizer, loader, step):
    warmup_steps = args.num_warmup_epochs * len(loader)
    scale = args.batch_size / 256
    if step < warmup_steps:
        lr = scale * step / warmup_steps
    else:
        step_for_sched = step - warmup_steps
        max_steps = args.epochs * len(loader) - warmup_steps
        q = 0.5 * (1 + math.cos(math.pi * step_for_sched / max_steps))
        end_lr = scale * 0.001
        lr = scale * q + end_lr * (1 - q)

    for i, param_group in enumerate(optimizer.param_groups):
        if 'fix_lr' in param_group and param_group['fix_lr']:  # fixed LR
            param_group['lr'] = scale * args.base_lr
        elif i == 1 and args.base_lr_biases is not None:  # cosine schedule with base_lr_biases
            param_group['lr'] = lr * args.base_lr_biases
        else:  # default: cosine schedule
            param_group['lr'] = lr * args.base_lr


def adjust_learning_rate_per_epoch(args, optimizer, epoch):
    scale = args.batch_size / 256
    if epoch < args.num_warmup_epochs:
        lr = scale * epoch / args.num_warmup_epochs
    else:
        epoch_for_sched = epoch - args.num_warmup_epochs
        max_epoch = args.epochs - args.num_warmup_epochs

        q = 0.5 * (1 + math.cos(math.pi * epoch_for_sched / max_epoch))
        end_lr = scale * 0.001
        lr = scale * q + end_lr * (1 - q)

    for i, param_group in enumerate(optimizer.param_groups):
        if 'fix_lr' in param_group and param_group['fix_lr']:  # fixed LR
            param_group['lr'] = scale * args.base_lr
        elif i == 1 and args.base_lr_biases is not None:  # cosine schedule with base_lr_biases
            param_group['lr'] = lr * args.base_lr_biases
        else:  # default: cosine schedule
            param_group['lr'] = lr * args.base_lr


class LARS(optim.Optimizer):
    def __init__(
        self,
        params,
        lr,
        weight_decay=0,
        momentum=0.9,
        eta=0.001,
        weight_decay_filter=None,
        lars_adaptation_filter=None,
    ):
        defaults = dict(
            lr=lr,
            weight_decay=weight_decay,
            momentum=momentum,
            eta=eta,
            weight_decay_filter=weight_decay_filter,
            lars_adaptation_filter=lars_adaptation_filter,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self):
        for g in self.param_groups:
            for p in g["params"]:
                dp = p.grad

                if dp is None:
                    continue

                if g["weight_decay_filter"] is None or not g["weight_decay_filter"](p):
                    dp = dp.add(p, alpha=g["weight_decay"])

                if g["lars_adaptation_filter"] is None or not g[
                    "lars_adaptation_filter"
                ](p):
                    param_norm = torch.norm(p)
                    update_norm = torch.norm(dp)
                    one = torch.ones_like(param_norm)
                    q = torch.where(
                        param_norm > 0.0,
                        torch.where(
                            update_norm > 0, (g["eta"] * param_norm / update_norm), one
                        ),
                        one,
                    )
                    dp = dp.mul(q)

                param_state = self.state[p]
                if "mu" not in param_state:
                    param_state["mu"] = torch.zeros_like(p)
                mu = param_state["mu"]
                mu.mul_(g["momentum"]).add_(dp)

                p.add_(mu, alpha=-g["lr"])


def get_optimizer(args, model):
    if args.use_constant_pred_lr:
        # separate predictor parameters
        param_weights = []
        param_biases = []
        param_predictor = []
        for name, param in model.named_parameters():
            if "predictor" in name:
                param_predictor.append(param)
            elif param.ndim == 1:
                param_biases.append(param)
            else:
                param_weights.append(param)
    else:
        # separate weights and biases
        param_weights = []
        param_biases = []
        for param in model.parameters():
            if param.ndim == 1:
                param_biases.append(param)
            else:
                param_weights.append(param)

    # optimizer and scaler
    if args.base_lr_biases is not None:
        print(f"Using different learning rate for biases and BN: {args.base_lr_biases}")

    if args.optimizer == "adamw":  # to use with ViTs
        assert not args.use_constant_pred_lr, "AdamW with constant predictor lr not implemented"
        parameters = [{'params': param_weights}, {'params': param_biases, 'weight_decay': 0.0}]
        optimizer = torch.optim.AdamW(
            parameters,
            lr=0,  # will be set later in adjust_learning_rate()
            weight_decay=args.wd,
        )
    elif args.optimizer == "lars":
        assert not args.use_constant_pred_lr, "LARS with constant predictor lr not implemented"
        parameters = [{'params': param_weights}, {'params': param_biases}]
        optimizer = LARS(
            parameters,
            lr=0,  # will be set later in adjust_learning_rate()
            weight_decay=args.wd,
            weight_decay_filter=exclude_bias_and_norm,
            lars_adaptation_filter=exclude_bias_and_norm
        )
    elif args.optimizer == "sgd":
        if args.use_constant_pred_lr:
            parameters = [
                {'params': param_weights, 'fix_lr': False},
                {'params': param_biases, 'fix_lr': False},
                {'params': param_predictor, 'fix_lr': True}
            ]
        else:
            parameters = [{'params': param_weights}, {'params': param_biases}]
        optimizer = torch.optim.SGD(
            parameters,
            lr=0,  # will be set later in adjust_learning_rate()
            momentum=0.9,
            weight_decay=args.wd,
        )
    else:
        raise ValueError(f"Unknown optimizer: {args.optimizer}")

    return optimizer

