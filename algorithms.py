# MIT License
# Copyright (c) 2026 MULAN authors

import math
from typing import Tuple, Dict, Sequence, Optional

import torch
import torch.nn.functional as F
from torch import Tensor

from models import BYOLNetwork, BYOLChimeraNetwork
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


class BYOLAlgorithm:
    def __init__(
            self,
            online_model: BYOLNetwork,
            target_model: BYOLNetwork,
            base_target_ema: float,
            num_steps: int,
    ):
        assert 0 <= base_target_ema <= 1
        assert num_steps > 0

        self.online_model = online_model
        self.target_model = target_model
        self.base_target_ema = base_target_ema
        self.num_steps = num_steps

    def get_tau(self, step: int) -> float:
        ema_delta = 1.0 - self.base_target_ema
        step = step % self.num_steps
        return 1.0 - ema_delta * (math.cos(math.pi * step / self.num_steps) + 1) / 2

    @torch.no_grad()
    def _update_target(self, tau: float) -> None:
        for param_online, param_target in zip(self.online_model.parameters(), self.target_model.parameters()):
            param_target.data.mul_(tau).add_(param_online.data, alpha=1 - tau)

    def step(self, images_tuple: Tuple[Tensor, Tensor], step: int, scaler: Optional = None) -> Dict:
        assert len(images_tuple) == 2, "Image tuple must contain two Tensors"

        # update target model: EMA of online model weights
        tau = self.get_tau(step)
        self._update_target(tau)

        # forward pass
        images_1, images_2 = images_tuple

        # targets
        with torch.no_grad():
            tgt_1, _ = self.target_model(images_1)
            tgt_2, _ = self.target_model(images_2)

        # predictions and loss computation
        loss_item: float = 0.0

        _, pred_1 = self.online_model(images_1)
        loss_1 = norm_cos_sim_loss(pred_1, tgt_2.detach())
        loss_item += loss_1.item()
        if scaler is not None:
            scaler.scale(loss_1).backward()
        else:
            loss_1.backward()

        _, pred_2 = self.online_model(images_2)
        loss_2 = norm_cos_sim_loss(pred_2, tgt_1.detach())
        loss_item += loss_2.item()
        if scaler is not None:
            scaler.scale(loss_2).backward()
        else:
            loss_2.backward()

        log_dict = {
            "loss": loss_item,
            "tau": tau,
        }

        return log_dict


class SimSiamAlgorithm:
    def __init__(self, online_model: BYOLNetwork, target_model: BYOLNetwork):
        """
        note: having two model copies is wasteful,
        but allows to save GPU memory by having separated forward-backward passes.
        """
        self.online_model = online_model
        self.target_model = target_model

        self.loss_fn = torch.nn.CosineSimilarity(dim=1)

    @torch.no_grad()
    def _update_target(self) -> None:
        for param_online, param_target in zip(self.online_model.parameters(), self.target_model.parameters()):
            param_target.data.copy_(param_online.data)
        for buffer_online, buffer_target in zip(self.online_model.buffers(), self.target_model.buffers()):
            buffer_target.data.copy_(buffer_online.data)

    def step(self, images_tuple: Tuple[Tensor, Tensor], step: int, scaler: Optional = None) -> Dict:
        assert len(images_tuple) == 2, "Image tuple must contain two Tensors"

        self._update_target()  # copy weights from online model to target model

        # forward pass
        images_1, images_2 = images_tuple

        # targets
        with torch.no_grad():
            tgt_1, _ = self.target_model(images_1)
            tgt_2, _ = self.target_model(images_2)

        # predictions and loss computation
        loss_item: float = 0.0

        backup_tgt_1, pred_1 = self.online_model(images_1)
        loss_1 = - 0.5 * self.loss_fn(pred_1, tgt_2.detach()).mean()
        loss_item += loss_1.item()
        if scaler is not None:
            scaler.scale(loss_1).backward()
        else:
            loss_1.backward()

        _, pred_2 = self.online_model(images_2)
        loss_2 = - 0.5 * self.loss_fn(pred_2, tgt_1.detach()).mean()
        loss_item += loss_2.item()
        if scaler is not None:
            scaler.scale(loss_2).backward()
        else:
            loss_2.backward()

        log_dict = {
            "loss": loss_item,
        }

        return log_dict


class MoCov3Algorithm:
    def __init__(
            self,
            online_model: BYOLNetwork,
            target_model: BYOLNetwork,
            base_target_ema: float,
            num_steps: int,
            loss_temp: float = 1.0
    ):
        assert 0 <= base_target_ema <= 1
        assert num_steps > 0

        self.online_model = online_model
        self.target_model = target_model
        self.base_target_ema = base_target_ema
        self.num_steps = num_steps
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

    def get_tau(self, step: int) -> float:
        ema_delta = 1.0 - self.base_target_ema
        return 1.0 - ema_delta * (math.cos(math.pi * step / self.num_steps) + 1) / 2

    @torch.no_grad()
    def _update_target(self, tau: float) -> None:
        for param_online, param_target in zip(self.online_model.parameters(), self.target_model.parameters()):
            param_target.data.mul_(tau).add_(param_online.data, alpha=1 - tau)

    def step(self, images_tuple: Tuple[Tensor, Tensor], step: int, scaler: Optional = None) -> Dict:
        assert len(images_tuple) == 2, "Image tuple must contain two Tensors"

        # update target model: EMA of online model weights
        tau = self.get_tau(step)
        self._update_target(tau)

        # forward pass
        images_1, images_2 = images_tuple

        # targets
        with torch.no_grad():
            tgt_1, _ = self.target_model(images_1)
            tgt_2, _ = self.target_model(images_2)

        # predictions and loss computation
        loss_item: float = 0.0

        _, pred_1 = self.online_model(images_1)
        loss_1 = self.contrastive_loss(pred_1, tgt_2.detach())
        loss_item += loss_1.item()
        if scaler is not None:
            scaler.scale(loss_1).backward()
        else:
            loss_1.backward()

        _, pred_2 = self.online_model(images_2)
        loss_2 = self.contrastive_loss(pred_2, tgt_1.detach())
        loss_item += loss_2.item()
        if scaler is not None:
            scaler.scale(loss_2).backward()
        else:
            loss_2.backward()

        log_dict = {
            "loss": loss_item,
            "tau": tau,
        }

        return log_dict


class BYOLAsymmetricAlgorithm:
    def __init__(
            self,
            online_model: BYOLNetwork,
            target_model: BYOLNetwork,
            base_target_ema: float,
            num_steps: int,
    ):
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
    def _update_target(self, tau: float) -> None:
        for param_online, param_target in zip(self.online_model.parameters(), self.target_model.parameters()):
            param_target.data.mul_(tau).add_(param_online.data, alpha=1 - tau)

    def step(self, images_tuple: Tuple[Tensor, Tensor], step: int, scaler: Optional = None) -> Dict:
        assert len(images_tuple) == 2, "Image tuple must contain two Tensors"

        # update target model: EMA of online model weights
        tau = self.get_tau(step)
        self._update_target(tau)

        # forward pass
        images_1, images_2 = images_tuple

        # target
        with torch.no_grad():
            tgt_2, _ = self.target_model(images_2)

        # prediction
        _, pred_1 = self.online_model(images_1)

        # loss computation
        loss_1 = 2.0 * norm_cos_sim_loss(pred_1, tgt_2.detach())
        loss_item: float = loss_1.item()

        # backward pass
        if scaler is not None:
            scaler.scale(loss_1).backward()
        else:
            loss_1.backward()

        log_dict = {
            "loss": loss_item,
            "tau": tau,
        }

        return log_dict


class BYOLMultiTaskAlgorithm(BYOLAlgorithm):
    def __init__(
            self,
            online_model: BYOLNetwork | BYOLChimeraNetwork,
            target_model: BYOLNetwork | BYOLChimeraNetwork,
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

        self.all_params_except_predictors = [
            param for name, param in self.online_model.named_parameters()
            if not "predictor." in name and not "predictors." in name
        ]
        print(f"Number of encoder parameters: {sum(p.numel() for p in self.all_params_except_predictors)}")

    def step(self, images_tuple: Tuple, step: int, scaler: Optional = None) -> Dict:
        num_input_views = len(images_tuple)
        num_global_views = self.num_views_per_task[0]
        assert sum(self.num_views_per_task) == num_input_views, \
            (f"Mismatch in number of views, "
             f"expected {sum(self.num_views_per_task)} views but input has {num_input_views} views")

        # update target model: EMA of online model weights
        tau = self.get_tau(step)
        self._update_target(tau)

        # targets: global views only
        tgt_list = []
        with torch.no_grad():
            for images in images_tuple[:num_global_views]:
                tgt, _ = self.target_model(images, view_type="identity")
                tgt_list.append(tgt)

        log_dict = {}
        # predictions and loss computation
        view_idx = 0  # index for the current view in images_tuple
        for task, num_views, weight in zip(self.tasks, self.num_views_per_task, self.task_loss_weights):
            loss_task_item: float = 0.0
            num_loss_terms_task = (num_views * num_global_views) if task != "global" else num_views

            for tvid in range(num_views):
                # forward pass for the current view
                feats, pred = self.online_model(images_tuple[view_idx], view_type=task)
                feats = feats.detach()  # use detached features for variance computation

                # compute losses for the current prediction
                loss_sub = torch.tensor(0.0, device=pred.device)
                for tgt_idx in range(num_global_views):
                    if view_idx != tgt_idx:  # skip the case where prediction and target are from the same view
                        loss_sub += norm_cos_sim_loss(pred, tgt_list[tgt_idx].detach())
                loss_sub = weight * loss_sub / num_loss_terms_task
                loss_task_item += loss_sub.item()

                # backward pass for the current pred (saves memory)
                if scaler is not None:
                    scaler.scale(loss_sub).backward()
                else:
                    loss_sub.backward()

                view_idx += 1

            log_dict[f"loss_{task}"] = loss_task_item  # log the loss for each task

        log_dict.update({
            "loss": sum(log_dict[f"loss_{task}"] for task in self.tasks),
            "tau": tau,
        })

        return log_dict


class SimSiamMultiTaskAlgorithm(SimSiamAlgorithm):
    def __init__(
            self,
            online_model: BYOLNetwork | BYOLChimeraNetwork,
            target_model: BYOLNetwork | BYOLChimeraNetwork,
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

    def step(self, images_tuple: Tuple, step: int, scaler: Optional = None) -> Dict:
        num_input_views = len(images_tuple)
        num_global_views = self.num_views_per_task[0]
        assert sum(self.num_views_per_task) == num_input_views, \
            (f"Mismatch in number of views, "
             f"expected {sum(self.num_views_per_task)} views but input has {num_input_views} views")

        self._update_target()  # copy weights from online model to target model

        # targets: global views only
        tgt_list = []
        with torch.no_grad():
            for images in images_tuple[:num_global_views]:
                tgt, _ = self.target_model(images, view_type="identity")
                tgt_list.append(tgt)

        log_dict = {}
        # predictions and loss computation
        view_idx = 0  # index for the current view in images_tuple
        for task, num_views, weight in zip(self.tasks, self.num_views_per_task, self.task_loss_weights):
            loss_task_item: float = 0.0
            num_loss_terms_task = (num_views * num_global_views) if task != "global" else num_views

            for _ in range(num_views):
                # forward pass for the current view
                _, pred = self.online_model(images_tuple[view_idx], view_type=task)

                # compute losses for the current prediction
                loss_sub = torch.tensor(0.0, device=pred.device)
                for tgt_idx in range(num_global_views):
                    if view_idx != tgt_idx:  # skip the case where prediction and target are from the same view
                        loss_sub += - self.loss_fn(pred, tgt_list[tgt_idx].detach()).mean()
                loss_sub = weight * loss_sub / num_loss_terms_task
                loss_task_item += loss_sub.item()

                # backward pass for the current pred (saves memory)
                if scaler is not None:
                    scaler.scale(loss_sub).backward()
                else:
                    loss_sub.backward()

                view_idx += 1

            log_dict[f"loss_{task}"] = loss_task_item  # log the loss for each task

        log_dict.update({
            "loss": sum(log_dict[f"loss_{task}"] for task in self.tasks),
        })

        return log_dict


class MoCov3MultiTaskAlgorithm(MoCov3Algorithm):
    def __init__(
            self,
            online_model: BYOLNetwork | BYOLChimeraNetwork,
            target_model: BYOLNetwork | BYOLChimeraNetwork,
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

    def step(self, images_tuple: Tuple, step: int, scaler: Optional = None) -> Dict:
        num_input_views = len(images_tuple)
        num_global_views = self.num_views_per_task[0]
        assert sum(self.num_views_per_task) == num_input_views, \
            (f"Mismatch in number of views, "
             f"expected {sum(self.num_views_per_task)} views but input has {num_input_views} views")

        # update target model: EMA of online model weights
        tau = self.get_tau(step)
        self._update_target(tau)

        # targets: global views only
        tgt_list = []
        with torch.no_grad():
            for images in images_tuple[:num_global_views]:
                tgt, _ = self.target_model(images, view_type="identity")
                tgt_list.append(tgt)

        log_dict = {}
        # predictions and loss computation
        view_idx = 0  # index for the current view in images_tuple
        for task, num_views, weight in zip(self.tasks, self.num_views_per_task, self.task_loss_weights):
            loss_task_item: float = 0.0
            num_loss_terms_task = (num_views * num_global_views) if task != "global" else num_views

            for _ in range(num_views):
                # forward pass for the current view
                _, pred = self.online_model(images_tuple[view_idx], view_type=task)

                # compute losses for the current prediction
                loss_sub = torch.tensor(0.0, device=pred.device)
                for tgt_idx in range(num_global_views):
                    if view_idx != tgt_idx:  # skip the case where prediction and target are from the same view
                        loss_sub += self.contrastive_loss(pred, tgt_list[tgt_idx].detach())
                loss_sub = weight * loss_sub / num_loss_terms_task
                loss_task_item += loss_sub.item()

                # backward pass for the current pred (saves memory)
                if scaler is not None:
                    scaler.scale(loss_sub).backward()
                else:
                    loss_sub.backward()

                view_idx += 1

            log_dict[f"loss_{task}"] = loss_task_item  # log the loss for each task

        log_dict.update({
            "loss": sum(log_dict[f"loss_{task}"] for task in self.tasks),
            "tau": tau,
        })

        return log_dict


def get_algorithm(
        online_model: BYOLNetwork | BYOLChimeraNetwork,
        target_model: BYOLNetwork | BYOLChimeraNetwork,
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
