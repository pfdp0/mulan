# MIT License
# Copyright (c) 2026 MULAN authors

import math
from typing import Tuple, Dict, Sequence, Optional
import warnings
from abc import ABC, abstractmethod
import contextlib

import torch
import torch.nn.functional as F
from torch import Tensor

import models as models
import distributed as dist


def norm_cos_sim_loss(pred: Tensor, tgt: Tensor):
    """
    Normalized cosine similarity loss (equivalent to MSE loss)
    Args:
        pred: (batch_size, num_features)
        tgt: (batch_size, num_features)
    """
    pred_norm = F.normalize(pred, dim=-1, p=2)
    tgt_norm = F.normalize(tgt, dim=-1, p=2)

    loss = 2.0 - 2.0 * (pred_norm * tgt_norm).sum(dim=-1).mean()

    return loss


class SSLAlgorithm(ABC):
    """
    Base class for self-supervised learning algorithms.
    Defines the interface for the main methods: update_target and training_step.
    """
    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def update_target(self, step: int) -> None:
        """
        Update the target model parameters based on the online model parameters and the current step.
        :note: For algorithms without a target model, this can be a no-op.
        """
        pass

    @abstractmethod
    def training_step(self, images_tuple: Tuple[Tensor, Tensor], step: int, scaler: Optional = None) -> Dict:
        """
        Perform a training step given a tuple of images (views).
        Runs the forward pass, computes the loss, and performs the backward pass.
        Args:
            images_tuple: A tuple of Tensors, each of shape (N, C, H, W), representing different views of the same images.
            step: The current training step, used for scheduling.
            scaler: Optional GradScaler for mixed precision training.
        Returns:
            A dictionary of logs to be recorded, e.g. losses, metrics, etc.
        """
        pass

    @torch.no_grad()
    def format_logs(self, raw_logs: Dict) -> Dict:
        """
        Method to convert raw logs (which may contain Tensors) into a format suitable for logging.
        Args:
            raw_logs: The raw logs dictionary returned by training_step.
        Returns:
            A formatted logs dictionary for logging.
        """
        formatted_logs = {}
        for key, value in raw_logs.items():
            if isinstance(value, Tensor):
                formatted_logs[key] = value.item()  # convert Tensor to scalar
            else:
                formatted_logs[key] = value  # keep non-Tensor values unchanged

        return formatted_logs


class BYOLAlgorithm(SSLAlgorithm):
    def __init__(
            self,
            online_model: models.BYOLNetwork,
            target_model: models.BYOLNetwork,
            base_target_ema: float,
            num_steps: int,
    ):
        super().__init__()

        assert 0 <= base_target_ema <= 1
        assert num_steps > 0

        self.online_model = online_model
        self.target_model = target_model
        self.base_target_ema = base_target_ema
        self.num_steps = num_steps

    def get_tau(self, step: int) -> float:
        ema_delta = 1.0 - self.base_target_ema
        return 1.0 - ema_delta * (math.cos(math.pi * step / self.num_steps) + 1) / 2

    @torch.no_grad()
    def update_target(self, step: int) -> None:
        tau = self.get_tau(step)
        for param_online, param_target in zip(self.online_model.parameters(), self.target_model.parameters()):
            param_target.data.mul_(tau).add_(param_online.data, alpha=1 - tau)

    def training_step(self, images_tuple: Tuple[Tensor, Tensor], step: int, scaler: Optional = None) -> Dict:
        assert len(images_tuple) == 2, "Image tuple must contain two Tensors"

        # forward pass
        images_1, images_2 = images_tuple

        # targets
        with torch.no_grad():
            target_out_1 = self.target_model(images_1)
            target_out_2 = self.target_model(images_2)

        # predictions and loss computation
        loss_for_logging = torch.tensor(0.0, device=images_tuple[0].device)  # to log the unscaled loss

        online_out_1 = self.online_model(images_1)
        loss_1 = norm_cos_sim_loss(online_out_1["predictions"], target_out_2["embeddings"].detach())
        loss_for_logging = loss_for_logging + loss_1.detach()
        if scaler is not None:
            scaler.scale(loss_1).backward()
        else:
            loss_1.backward()

        online_out_2 = self.online_model(images_2)
        loss_2 = norm_cos_sim_loss(online_out_2["predictions"], target_out_1["embeddings"].detach())
        loss_for_logging = loss_for_logging + loss_2.detach()
        if scaler is not None:
            scaler.scale(loss_2).backward()
        else:
            loss_2.backward()

        log_dict = {
            "loss": loss_for_logging
        }

        return log_dict


class SimSiamAlgorithm(SSLAlgorithm):
    def __init__(self, online_model: models.BYOLNetwork, target_model: models.BYOLNetwork):
        """
        note: having two model copies is wasteful,
        but allows to save GPU memory by having separated forward-backward passes.
        """
        super().__init__()

        self.online_model = online_model
        self.target_model = target_model

        self.loss_fn = torch.nn.CosineSimilarity(dim=1)

    @torch.no_grad()
    def update_target(self, step: int) -> None:
        for param_online, param_target in zip(self.online_model.parameters(), self.target_model.parameters()):
            param_target.data.copy_(param_online.data)
        for buffer_online, buffer_target in zip(self.online_model.buffers(), self.target_model.buffers()):
            buffer_target.data.copy_(buffer_online.data)

    def training_step(self, images_tuple: Tuple[Tensor, Tensor], step: int, scaler: Optional = None) -> Dict:
        assert len(images_tuple) == 2, "Image tuple must contain two Tensors"

        # forward pass
        images_1, images_2 = images_tuple

        # targets
        with torch.no_grad():
            target_out_1 = self.target_model(images_1)
            target_out_2 = self.target_model(images_2)

        # predictions and loss computation
        loss_for_logging = torch.tensor(0.0, device=images_tuple[0].device)  # to log the unscaled loss

        online_out_1 = self.online_model(images_1)
        loss_1 = - 0.5 * self.loss_fn(online_out_1["predictions"], target_out_2["embeddings"].detach()).mean()
        loss_for_logging = loss_for_logging + loss_1.detach()
        if scaler is not None:
            scaler.scale(loss_1).backward()
        else:
            loss_1.backward()

        online_out_2 = self.online_model(images_2)
        loss_2 = - 0.5 * self.loss_fn(online_out_2["predictions"], target_out_1["embeddings"].detach()).mean()
        loss_for_logging = loss_for_logging + loss_2.detach()
        if scaler is not None:
            scaler.scale(loss_2).backward()
        else:
            loss_2.backward()

        log_dict = {
            "loss": loss_for_logging
        }

        return log_dict


class MoCov3Algorithm(BYOLAlgorithm):
    def __init__(
            self,
            online_model: models.BYOLNetwork,
            target_model: models.BYOLNetwork,
            base_target_ema: float,
            num_steps: int,
            loss_temp: float = 1.0
    ):
        super().__init__(online_model, target_model, base_target_ema, num_steps)
        self.loss_temp = loss_temp

    def contrastive_loss(self, q, k):
        # normalize
        q = F.normalize(q, dim=1)
        k = F.normalize(k, dim=1)
        # gather all targets
        k = dist.concat_all_gather(k)
        # Einstein sum is more intuitive
        logits = torch.einsum('nc,mc->nm', [q, k]) / self.loss_temp
        batch_size_per_gpu = logits.shape[0]  # batch size per GPU
        gpu_shift = batch_size_per_gpu * torch.distributed.get_rank()
        labels = (torch.arange(batch_size_per_gpu, dtype=torch.long) + gpu_shift).cuda()
        return F.cross_entropy(logits, labels) * (2.0 * self.loss_temp)

    def training_step(self, images_tuple: Tuple[Tensor, Tensor], step: int, scaler: Optional = None) -> Dict:
        assert len(images_tuple) == 2, "Image tuple must contain two Tensors"

        # forward pass
        images_1, images_2 = images_tuple

        # targets
        with torch.no_grad():
            target_out_1 = self.target_model(images_1)
            target_out_2 = self.target_model(images_2)

        # predictions and loss computation
        loss_for_logging = torch.tensor(0.0, device=images_tuple[0].device)  # to log the unscaled loss

        online_out_1 = self.online_model(images_1)
        loss_1 = self.contrastive_loss(online_out_1["predictions"], target_out_2["embeddings"].detach())
        loss_for_logging = loss_for_logging + loss_1.detach()
        if scaler is not None:
            scaler.scale(loss_1).backward()
        else:
            loss_1.backward()

        online_out_2 = self.online_model(images_2)
        loss_2 = self.contrastive_loss(online_out_2["predictions"], target_out_1["embeddings"].detach())
        loss_for_logging = loss_for_logging + loss_2.detach()
        if scaler is not None:
            scaler.scale(loss_2).backward()
        else:
            loss_2.backward()

        log_dict = {
            "loss": loss_for_logging
        }

        return log_dict


class BYOLAsymmetricAlgorithm(BYOLAlgorithm):
    def __init__(
            self,
            online_model: models.BYOLNetwork,
            target_model: models.BYOLNetwork,
            base_target_ema: float,
            num_steps: int,
    ):
        super().__init__(online_model, target_model, base_target_ema, num_steps)

    def training_step(self, images_tuple: Tuple[Tensor, Tensor], step: int, scaler: Optional = None) -> Dict:
        assert len(images_tuple) == 2, "Image tuple must contain two Tensors"

        # forward pass
        images_1, images_2 = images_tuple

        # target
        with torch.no_grad():
            target_out = self.target_model(images_2)

        # prediction
        online_out = self.online_model(images_1)

        # loss computation
        loss_1 = 2.0 * norm_cos_sim_loss(online_out["predictions"], target_out["embeddings"].detach())
        loss_for_logging = loss_1.detach()

        # backward pass
        if scaler is not None:
            scaler.scale(loss_1).backward()
        else:
            loss_1.backward()

        log_dict = {
            "loss": loss_for_logging
        }

        return log_dict


class BYOLMultiTaskAlgorithm(BYOLAlgorithm):
    def __init__(
            self,
            online_model: models.BYOLNetwork | models.BYOLChimeraNetwork,
            target_model: models.BYOLNetwork | models.BYOLChimeraNetwork,
            base_target_ema: float,
            num_steps: int,
            tasks: Sequence[str] = ("global", "local", "cutout"),
            num_views_per_task: Sequence[int] = (2, 2, 1),
            task_loss_weights: Sequence[float] = (1.0, 1.0, 1.0)
    ):
        super().__init__(online_model, target_model, base_target_ema, num_steps)
        assert len(tasks) == len(num_views_per_task) == len(task_loss_weights), \
            (f"Mismatch between number of tasks and the length of the num_views_per_task tuple, "
             f"got {len(tasks)=} != {len(num_views_per_task)=} != {len(task_loss_weights)=}")
        assert tasks[0] == "global", \
            "The global views are required since also used as targets, and should be the 1st task"
        assert all(num_views > 0 for num_views in num_views_per_task)
        self.tasks = tasks
        self.num_views_per_task = num_views_per_task
        self.task_loss_weights = task_loss_weights

    def training_step(self, images_tuple: Tuple, step: int, scaler: Optional = None) -> Dict:
        num_input_views = len(images_tuple)
        num_global_views = self.num_views_per_task[0]
        assert sum(self.num_views_per_task) == num_input_views, \
            (f"Mismatch in number of views, "
             f"expected {sum(self.num_views_per_task)} views but input has {num_input_views} views")

        # initialize logs
        log_dict = {
            "loss": torch.tensor(0.0, device=images_tuple[0].device)
        }
        log_dict.update({f"loss_{task}": torch.tensor(0.0, device=images_tuple[0].device) for task in self.tasks})

        # targets: global views only
        tgt_list = []
        with torch.no_grad():
            for images in images_tuple[:num_global_views]:
                target_out = self.target_model(images, view_type="identity")
                tgt_list.append(target_out["embeddings"])

        # predictions and loss computation
        view_idx = 0  # index for the current view in images_tuple
        for task, num_views, weight in zip(self.tasks, self.num_views_per_task, self.task_loss_weights):
            num_loss_terms_task = (num_views * num_global_views) if task != "global" else num_views

            for _ in range(num_views):
                # avoid DDP gradient synchronization for intermediate views (i.e., sync in the end)
                is_last_view = (view_idx == num_input_views - 1)
                if is_last_view or not hasattr(self.online_model, "no_sync"):
                    context = contextlib.nullcontext()
                else:
                    context = self.online_model.no_sync()

                with context:
                    # forward pass for the current view
                    online_out = self.online_model(images_tuple[view_idx], view_type=task)

                    # compute losses for the current prediction
                    loss_sub = torch.tensor(0.0, device=online_out["predictions"].device)
                    for tgt_idx in range(num_global_views):
                        if view_idx != tgt_idx:  # skip the case where prediction and target are from the same view
                            loss_sub += norm_cos_sim_loss(online_out["predictions"], tgt_list[tgt_idx].detach())
                    loss_sub = weight * loss_sub / num_loss_terms_task
                    log_dict[f"loss_{task}"] = log_dict[
                                                   f"loss_{task}"] + loss_sub.detach()  # log the loss for the current view

                    # backward pass for the current pred (frees up memory)
                    if scaler is not None:
                        scaler.scale(loss_sub).backward()
                    else:
                        loss_sub.backward()

                view_idx += 1

            # accumulate task loss into the overall loss
            log_dict["loss"] = log_dict["loss"] + log_dict[f"loss_{task}"]

        return log_dict


class SimSiamMultiTaskAlgorithm(SimSiamAlgorithm):
    def __init__(
            self,
            online_model: models.BYOLNetwork | models.BYOLChimeraNetwork,
            target_model: models.BYOLNetwork | models.BYOLChimeraNetwork,
            tasks: Sequence[str] = ("global", "local", "cutout"),
            num_views_per_task: Sequence[int] = (2, 2, 1),
            task_loss_weights: Sequence[float] = (0.5, 0.5, 0.5)
    ):
        super().__init__(online_model, target_model)
        assert len(tasks) == len(num_views_per_task) == len(task_loss_weights), \
            (f"Mismatch between number of tasks and the length of the num_views_per_task tuple, "
             f"got {len(tasks)=} != {len(num_views_per_task)=} != {len(task_loss_weights)=}")
        assert tasks[0] == "global", \
            "The global views are required since also used as targets, and should be the 1st task"
        assert all(num_views > 0 for num_views in num_views_per_task)
        self.tasks = tasks
        self.num_views_per_task = num_views_per_task
        self.task_loss_weights = task_loss_weights

    def training_step(self, images_tuple: Tuple, step: int, scaler: Optional = None) -> Dict:
        num_input_views = len(images_tuple)
        num_global_views = self.num_views_per_task[0]
        assert sum(self.num_views_per_task) == num_input_views, \
            (f"Mismatch in number of views, "
             f"expected {sum(self.num_views_per_task)} views but input has {num_input_views} views")

        # initialize logs
        log_dict = {"loss": torch.tensor(0.0, device=images_tuple[0].device)}
        log_dict.update({f"loss_{task}": torch.tensor(0.0, device=images_tuple[0].device) for task in self.tasks})

        # targets: global views only
        tgt_list = []
        with torch.no_grad():
            for images in images_tuple[:num_global_views]:
                target_out = self.target_model(images, view_type="identity")
                tgt_list.append(target_out["embeddings"])

        # predictions and loss computation
        view_idx = 0  # index for the current view in images_tuple
        for task, num_views, weight in zip(self.tasks, self.num_views_per_task, self.task_loss_weights):
            num_loss_terms_task = (num_views * num_global_views) if task != "global" else num_views

            for _ in range(num_views):
                # avoid DDP gradient synchronization for intermediate views (i.e., sync in the end)
                is_last_view = (view_idx == num_input_views - 1)
                if is_last_view or not hasattr(self.online_model, "no_sync"):
                    context = contextlib.nullcontext()
                else:
                    context = self.online_model.no_sync()

                with context:
                    # forward pass for the current view
                    online_out = self.online_model(images_tuple[view_idx], view_type=task)

                    # compute losses for the current prediction
                    loss_sub = torch.tensor(0.0, device=online_out["predictions"].device)
                    for tgt_idx in range(num_global_views):
                        if view_idx != tgt_idx:  # skip the case where prediction and target are from the same view
                            loss_sub += - self.loss_fn(online_out["predictions"], tgt_list[tgt_idx].detach()).mean()
                    loss_sub = weight * loss_sub / num_loss_terms_task
                    log_dict[f"loss_{task}"] = log_dict[f"loss_{task}"] + loss_sub.detach()  # log the unscaled loss for the current view

                    # backward pass for the current pred (saves memory)
                    if scaler is not None:
                        scaler.scale(loss_sub).backward()
                    else:
                        loss_sub.backward()

                view_idx += 1

            # accumulate task loss into the overall loss
            log_dict["loss"] = log_dict["loss"] + log_dict[f"loss_{task}"]

        return log_dict


class MoCov3MultiTaskAlgorithm(MoCov3Algorithm):
    def __init__(
            self,
            online_model: models.BYOLNetwork | models.BYOLChimeraNetwork,
            target_model: models.BYOLNetwork | models.BYOLChimeraNetwork,
            base_target_ema: float,
            num_steps: int,
            loss_temp: float = 1.0,
            tasks: Sequence[str] = ("global", "local", "cutout"),
            num_views_per_task: Sequence[int] = (2, 2, 1),
            task_loss_weights: Sequence[float] = (1.0, 1.0, 1.0)
    ):
        super().__init__(online_model, target_model, base_target_ema, num_steps, loss_temp)
        assert len(tasks) == len(num_views_per_task) == len(task_loss_weights), \
            (f"Mismatch between number of tasks and the length of the num_views_per_task tuple, "
             f"got {len(tasks)=} != {len(num_views_per_task)=} != {len(task_loss_weights)=}")
        assert tasks[0] == "global", \
            "The global views are required since also used as targets, and should be the 1st task"
        assert all(num_views > 0 for num_views in num_views_per_task)
        self.tasks = tasks
        self.num_views_per_task = num_views_per_task
        self.task_loss_weights = task_loss_weights

    def training_step(self, images_tuple: Tuple, step: int, scaler: Optional = None) -> Dict:
        num_input_views = len(images_tuple)
        num_global_views = self.num_views_per_task[0]
        assert sum(self.num_views_per_task) == num_input_views, \
            (f"Mismatch in number of views, "
             f"expected {sum(self.num_views_per_task)} views but input has {num_input_views} views")

        # initialize logs similarly to BYOLMultiTaskAlgorithm
        log_dict = {"loss": torch.tensor(0.0, device=images_tuple[0].device)}
        log_dict.update({f"loss_{task}": torch.tensor(0.0, device=images_tuple[0].device) for task in self.tasks})

        # targets: global views only
        tgt_list = []
        with torch.no_grad():
            for images in images_tuple[:num_global_views]:
                target_out = self.target_model(images, view_type="identity")
                tgt_list.append(target_out["embeddings"])

        # predictions and loss computation
        view_idx = 0  # index for the current view in images_tuple
        for task, num_views, weight in zip(self.tasks, self.num_views_per_task, self.task_loss_weights):
            num_loss_terms_task = (num_views * num_global_views) if task != "global" else num_views

            for _ in range(num_views):
                # avoid DDP gradient synchronization for intermediate views (i.e., sync in the end)
                is_last_view = (view_idx == num_input_views - 1)
                if is_last_view or not hasattr(self.online_model, "no_sync"):
                    context = contextlib.nullcontext()
                else:
                    context = self.online_model.no_sync()

                with context:
                    # forward pass for the current view
                    online_out = self.online_model(images_tuple[view_idx], view_type=task)

                    # compute losses for the current prediction
                    loss_sub = torch.tensor(0.0, device=online_out["predictions"].device)
                    for tgt_idx in range(num_global_views):
                        if view_idx != tgt_idx:  # skip the case where prediction and target are from the same view
                            loss_sub += self.contrastive_loss(online_out["predictions"], tgt_list[tgt_idx].detach())
                    loss_sub = weight * loss_sub / num_loss_terms_task
                    log_dict[f"loss_{task}"] = log_dict[f"loss_{task}"] + loss_sub.detach()

                    # backward pass for the current pred (saves memory)
                    if scaler is not None:
                        scaler.scale(loss_sub).backward()
                    else:
                        loss_sub.backward()

                view_idx += 1

            # accumulate task loss into the overall loss
            log_dict["loss"] = log_dict["loss"] + log_dict[f"loss_{task}"]

        return log_dict


def get_algorithm(
        online_model: torch.nn.Module,
        target_model: torch.nn.Module,
        total_num_steps: int,
        args
):
    if args.training_fn == "byol_default":
        assert args.transform != "multitask", \
            "BYOL default training function does not support multitask augmentations"
        algorithm = BYOLAlgorithm(
            online_model,
            target_model,
            args.base_target_ema,
            total_num_steps
        )
    elif args.training_fn == "simsiam_default":
        assert args.transform != "multitask", \
            "SimSiam default training function does not support multitask augmentations"
        algorithm = SimSiamAlgorithm(online_model, target_model)
    elif args.training_fn == "mocov3_default":
        assert args.transform != "multitask", \
            "MoCov3 default training function does not support multitask augmentations"
        algorithm = MoCov3Algorithm(
            online_model,
            target_model,
            args.base_target_ema,
            total_num_steps,
            args.moco_loss_temp
        )
    elif args.training_fn == "byol_asymmetric":
        assert args.transform != "multitask", \
            "BYOL asymmetric training function does not support multitask augmentations"
        algorithm = BYOLAsymmetricAlgorithm(
            online_model,
            target_model,
            args.base_target_ema,
            total_num_steps
        )
    elif args.training_fn == "byol_multitask":
        assert args.transform == "multitask", \
            "BYOL multi-task training function requires multitask augmentations"
        print(f"Using multi-task BYOL with tasks={args.tasks} with loss_weights={args.task_weights}")
        algorithm = BYOLMultiTaskAlgorithm(
            online_model,
            target_model,
            args.base_target_ema,
            total_num_steps,
            tasks=args.tasks,
            num_views_per_task=args.num_views_per_task,
            task_loss_weights=args.task_weights
        )
    elif args.training_fn == "simsiam_multitask":
        assert args.transform == "multitask", \
            "SimSiam multi-task training function requires multitask augmentations"
        print(f"Using multi-task SimSiam with tasks={args.tasks} with loss_weights={args.task_weights}")
        algorithm = SimSiamMultiTaskAlgorithm(
            online_model,
            target_model,
            tasks=args.tasks,
            num_views_per_task=args.num_views_per_task,
            task_loss_weights=args.task_weights
        )
    elif args.training_fn == "mocov3_multitask":
        assert args.transform == "multitask", \
            "MoCov3 multi-task training function requires multitask augmentations"
        print(f"Using multi-task MoCov3 with tasks={args.tasks} with loss_weights={args.task_weights}")
        algorithm = MoCov3MultiTaskAlgorithm(
            online_model,
            target_model,
            args.base_target_ema,
            total_num_steps,
            loss_temp=args.moco_loss_temp,
            tasks=args.tasks,
            num_views_per_task=args.num_views_per_task,
            task_loss_weights=args.task_weights
        )
    else:
        raise ValueError(f"Unknown training function: {args.training_fn}")

    return algorithm
