# MIT License
# Copyright (c) 2026 MULAN authors

from typing import Union, Sequence, Tuple, Dict, Any
import warnings
import math

import torch
from torch import Tensor
from torchvision.transforms import InterpolationMode
import torchvision.transforms.v2 as transforms_v2
from torchvision.transforms.v2._utils import query_size
from torch.nn import Module


class RandomCutout(Module):
    """
    Randomly mask an image block (cutout).
    The implementation is adapted from torchvision.transforms.RandomResizedCrop.
    """

    def __init__(
        self,
        scale: Tuple[float, float] = (0.2, 0.4),
        ratio: Tuple[float, float] = (3.0 / 4.0, 4.0 / 3.0),
        fill: Union[float, Tuple[float, float, float]] = 0,
    ) -> None:
        super().__init__()

        if not isinstance(scale, Sequence):
            raise TypeError("Scale should be a sequence")
        if not isinstance(ratio, Sequence):
            raise TypeError("Ratio should be a sequence")
        if (scale[0] > scale[1]) or (ratio[0] > ratio[1]):
            warnings.warn("Scale and ratio should be of kind (min, max)")
        if not (isinstance(fill, float) or (isinstance(fill, Sequence) and len(fill) == 3)):
            raise ValueError("Fill should be an integer or a tuple of length 3")

        self.scale = scale
        self.ratio = ratio
        self._log_ratio = torch.log(torch.tensor(self.ratio))

        if isinstance(fill, float):
            fill = (fill, fill, fill)
        self.fill = torch.tensor(fill).view(3, 1, 1)

    def _get_params(self, flat_inputs: Tensor) -> Dict[str, Any]:
        height, width = query_size(flat_inputs)
        area = height * width

        log_ratio = self._log_ratio
        for _ in range(10):
            target_area = area * torch.empty(1).uniform_(self.scale[0], self.scale[1]).item()
            aspect_ratio = torch.exp(
                torch.empty(1).uniform_(
                    log_ratio[0],  # type: ignore[arg-type]
                    log_ratio[1],  # type: ignore[arg-type]
                )
            ).item()

            w = int(round(math.sqrt(target_area * aspect_ratio)))
            h = int(round(math.sqrt(target_area / aspect_ratio)))

            if 0 < w <= width and 0 < h <= height:
                i = torch.randint(0, height - h + 1, size=(1,)).item()
                j = torch.randint(0, width - w + 1, size=(1,)).item()
                break
        else:
            # Fallback to central block
            in_ratio = float(width) / float(height)
            if in_ratio < min(self.ratio):
                w = width
                h = int(round(w / min(self.ratio)))
            elif in_ratio > max(self.ratio):
                h = height
                w = int(round(h * max(self.ratio)))
            else:  # whole image
                w = width
                h = height
            i = (height - h) // 2
            j = (width - w) // 2

        return dict(top=i, left=j, height=h, width=w)

    def forward(self, inpt: Tensor) -> Any:
        if not isinstance(inpt, Tensor):
            NotImplementedError("The transform only supports input Tensors")
        assert inpt.dim() == 3 and inpt.shape[0] == 3, f"Input tensor should be a tensor of (C, H, W) shape"
        params = self._get_params(inpt)

        fill = self.fill.to(inpt.dtype)
        out = inpt.clone()
        out[:, params['top']:params['top']+params['height'], params['left']:params['left']+params['width']] = fill

        return out


class TrainTransformBYOL(object):
    def __init__(self):
        self.transform = transforms_v2.Compose(
            [
                transforms_v2.RandomResizedCrop(
                    224, interpolation=InterpolationMode.BICUBIC
                ),
                transforms_v2.RandomHorizontalFlip(p=0.5),
                transforms_v2.ToDtype(torch.float32, scale=True),  # values are scaled to [0, 1] here
                transforms_v2.RandomApply(
                    [
                        transforms_v2.ColorJitter(
                            brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1
                        )
                    ],
                    p=0.8,
                ),
                transforms_v2.RandomGrayscale(p=0.2),
                transforms_v2.GaussianBlur(kernel_size=23, sigma=(0.1, 2.0)),  
                transforms_v2.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )
        self.transform_prime = transforms_v2.Compose(
            [
                transforms_v2.RandomResizedCrop(
                    224, interpolation=transforms_v2.InterpolationMode.BICUBIC
                ),
                transforms_v2.RandomHorizontalFlip(p=0.5),
                transforms_v2.ToDtype(torch.float32, scale=True),  # values are scaled to [0, 1] here
                transforms_v2.RandomApply(
                    [
                        transforms_v2.ColorJitter(
                            brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1
                        )
                    ],
                    p=0.8,
                ),
                transforms_v2.RandomGrayscale(p=0.2),
                transforms_v2.RandomApply([
                    transforms_v2.GaussianBlur(kernel_size=23, sigma=(0.1, 2.0))
                ], p=0.1),  
                transforms_v2.RandomSolarize(threshold=0.5, p=0.2), 
                transforms_v2.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

    def __call__(self, sample):
        sample_img = transforms_v2.functional.to_image(sample)
        x1 = self.transform(sample_img)
        x2 = self.transform_prime(sample_img)
        return x1, x2


class TrainTransformSimSiam(object):
    def __init__(self):
        self.transform = transforms_v2.Compose(
            [
                transforms_v2.RandomResizedCrop(
                    224, scale=(0.2, 1.), interpolation=InterpolationMode.BICUBIC
                ),
                transforms_v2.RandomHorizontalFlip(p=0.5),
                transforms_v2.ToDtype(torch.float32, scale=True),
                transforms_v2.RandomApply(
                    [
                        transforms_v2.ColorJitter(
                            brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1
                        )
                    ],
                    p=0.8,
                ),  # saturation=0.4 for SimSiam
                transforms_v2.RandomGrayscale(p=0.2),
                transforms_v2.RandomApply([
                    transforms_v2.GaussianBlur(kernel_size=23, sigma=(0.1, 2.0))
                ], p=0.5),
                transforms_v2.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

    def __call__(self, sample):
        sample_img = transforms_v2.functional.to_image(sample)
        x1 = self.transform(sample_img)
        x2 = self.transform(sample_img)
        return x1, x2


class TrainTransformMoCov3(object):
    """
    The only difference with BYOL is the min glob scale
    """
    def __init__(self):
        self.transform = transforms_v2.Compose(
            [
                transforms_v2.RandomResizedCrop(
                    224, scale=(0.2, 1.), interpolation=InterpolationMode.BICUBIC
                ),
                transforms_v2.RandomHorizontalFlip(p=0.5),
                transforms_v2.ToDtype(torch.float32, scale=True),  # values are scaled to [0, 1] here
                transforms_v2.RandomApply(
                    [
                        transforms_v2.ColorJitter(
                            brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1
                        )
                    ],
                    p=0.8,
                ),
                transforms_v2.RandomGrayscale(p=0.2),
                transforms_v2.GaussianBlur(kernel_size=23, sigma=(0.1, 2.0)),  
                transforms_v2.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )
        self.transform_prime = transforms_v2.Compose(
            [
                transforms_v2.RandomResizedCrop(
                    224, scale=(0.2, 1.), interpolation=transforms_v2.InterpolationMode.BICUBIC
                ),
                transforms_v2.RandomHorizontalFlip(p=0.5),
                transforms_v2.ToDtype(torch.float32, scale=True),  # values are scaled to [0, 1] here
                transforms_v2.RandomApply(
                    [
                        transforms_v2.ColorJitter(
                            brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1
                        )
                    ],
                    p=0.8,
                ),
                transforms_v2.RandomGrayscale(p=0.2),
                transforms_v2.RandomApply([
                    transforms_v2.GaussianBlur(kernel_size=23, sigma=(0.1, 2.0))
                ], p=0.1),  
                transforms_v2.RandomSolarize(threshold=0.5, p=0.2), 
                transforms_v2.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

    def __call__(self, sample):
        sample_img = transforms_v2.functional.to_image(sample)
        x1 = self.transform(sample_img)
        x2 = self.transform_prime(sample_img)
        return x1, x2


class TrainTransformNoCrop(object):
    """
    No random cropping, but all other augmentations are kept the same as BYOL
    """
    def __init__(self):
        self.transform = transforms_v2.Compose(
            [
                transforms_v2.Resize((224, 224), interpolation=InterpolationMode.BICUBIC),
                transforms_v2.RandomHorizontalFlip(p=0.5),
                transforms_v2.ToDtype(torch.float32, scale=True),  # values are scaled to [0, 1] here
                transforms_v2.RandomApply(
                    [
                        transforms_v2.ColorJitter(
                            brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1
                        )
                    ],
                    p=0.8,
                ),
                transforms_v2.RandomGrayscale(p=0.2),
                transforms_v2.GaussianBlur(kernel_size=23, sigma=(0.1, 2.0)),  
                transforms_v2.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )
        self.transform_prime = transforms_v2.Compose(
            [
                transforms_v2.Resize((224, 224), interpolation=InterpolationMode.BICUBIC),
                transforms_v2.RandomHorizontalFlip(p=0.5),
                transforms_v2.ToDtype(torch.float32, scale=True),  # values are scaled to [0, 1] here
                transforms_v2.RandomApply(
                    [
                        transforms_v2.ColorJitter(
                            brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1
                        )
                    ],
                    p=0.8,
                ),
                transforms_v2.RandomGrayscale(p=0.2),
                transforms_v2.RandomApply([
                    transforms_v2.GaussianBlur(kernel_size=23, sigma=(0.1, 2.0))
                ], p=0.1),  
                transforms_v2.RandomSolarize(threshold=0.5, p=0.2), 
                transforms_v2.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

    def __call__(self, sample):
        sample_img = transforms_v2.functional.to_image(sample)
        x1 = self.transform(sample_img)
        x2 = self.transform_prime(sample_img)
        return x1, x2


class TrainTransformOnlyCutoutAsym(object):
    """
    Use only cutout augmentation. And only in the 1st view.
    """
    def __init__(self, cutout_range: Tuple[float, float] = (0.2, 0.6)):
        self.transform = transforms_v2.Compose(
            [
                transforms_v2.Resize((224, 224), interpolation=InterpolationMode.BICUBIC),
                transforms_v2.ToDtype(torch.float32, scale=True),  # values are scaled to [0, 1] here
                RandomCutout(scale=cutout_range, fill=(0.485, 0.456, 0.406)),
                transforms_v2.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )
        self.transform_prime = transforms_v2.Compose(
            [
                transforms_v2.Resize((224, 224), interpolation=InterpolationMode.BICUBIC),
                transforms_v2.ToDtype(torch.float32, scale=True),  # values are scaled to [0, 1] here
                transforms_v2.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

    def __call__(self, sample):
        sample_img = transforms_v2.functional.to_image(sample)
        x1 = self.transform(sample_img)
        x2 = self.transform_prime(sample_img)
        return x1, x2


class TrainTransformOnlyCutout(object):
    """
    Use only cutout augmentation. In both views.
    """
    def __init__(self, cutout_range: Tuple[float, float] = (0.2, 0.6)):
        self.transform = transforms_v2.Compose(
            [
                transforms_v2.Resize((224, 224), interpolation=InterpolationMode.BICUBIC),
                transforms_v2.ToDtype(torch.float32, scale=True),  # values are scaled to [0, 1] here
                RandomCutout(scale=cutout_range, fill=(0.485, 0.456, 0.406)),
                transforms_v2.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

    def __call__(self, sample):
        sample_img = transforms_v2.functional.to_image(sample)
        x1 = self.transform(sample_img)
        x2 = self.transform(sample_img)
        return x1, x2


class TrainTransformOnlyCrop(object):
    """
    Use only random cropping in both views
    """
    def __init__(self):
        self.transform = transforms_v2.Compose(
            [
                transforms_v2.RandomResizedCrop(
                    224, interpolation=InterpolationMode.BICUBIC
                ),
                transforms_v2.ToDtype(torch.float32, scale=True),  # values are scaled to [0, 1] here
                transforms_v2.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

    def __call__(self, sample):
        sample_img = transforms_v2.functional.to_image(sample)
        x1 = self.transform(sample_img)
        x2 = self.transform(sample_img)
        return x1, x2


class TrainTransformMultiTask(object):
    def __init__(
            self,
            tasks: Sequence[str] = ("global", "local", "cutout"),
            num_views_per_task: Sequence[int] = (2, 2, 1),
            cutout_range: Tuple[float, float] = (0.2, 0.6),
            min_glob_scale: float = 0.25,
            min_loc_scale: float = 0.08,
            max_loc_scale: float = 0.25,
            loc_crop_size: int = 96
    ):
        assert len(tasks) == len(num_views_per_task), \
            (f"Mismatch between number of tasks and the length of the num_views_per_task tuple, "
             f"got {len(tasks) = } != {len(num_views_per_task) = }")
        assert tasks[0] == "global" and num_views_per_task[0] == 2, \
            "There should be 2 global views, and global should be the 1st task"
        assert all(num_views > 0 for num_views in num_views_per_task)
        assert 0 <= cutout_range[0] <= cutout_range[1] <= 1.0
        assert min_loc_scale <= max_loc_scale <= 1.0, "min_loc_scale must be <= max_loc_scale <= 1.0"
        assert min_glob_scale <= 1.0, "min_glob_scale must be <= 1.0"

        scale_ranges = {
            "global": (min_glob_scale, 1.0),
            "local": (min_loc_scale, max_loc_scale),
            "cutout": (0.25, 1.0),
            "global_more": (min_glob_scale, 1.0),
        }
        description_str = f"Using per-task crops with:"
        for task, num_views in zip(tasks, num_views_per_task):
            sz = loc_crop_size if task == "local" else 224
            s_min, s_max = scale_ranges[task]
            description_str += f"\n\t{num_views} {task} crops of size {sz}x{sz} with scales in range {s_min} -> {s_max}"
        print(description_str)

        self.tasks = tasks
        self.num_views_per_task = num_views_per_task

        # task-specific args
        self.min_glob_scale_threshold = min_glob_scale
        self.max_loc_scale_threshold = max_loc_scale
        self.loc_crop_size = loc_crop_size
        self.cutout_range = cutout_range

        augs_g1 = transforms_v2.Compose(
            [
                transforms_v2.RandomResizedCrop(
                    224, scale=(min_glob_scale, 1.0), interpolation=InterpolationMode.BICUBIC
                ),
                transforms_v2.RandomHorizontalFlip(p=0.5),
                transforms_v2.ToDtype(torch.float32, scale=True),  # values are scaled to [0, 1] here
                transforms_v2.RandomApply(
                    [
                        transforms_v2.ColorJitter(
                            brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1
                        )
                    ],
                    p=0.8,
                ),
                transforms_v2.RandomGrayscale(p=0.2),
                transforms_v2.GaussianBlur(kernel_size=23, sigma=(0.1, 2.0)),  
                transforms_v2.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )
        augs_g2 = transforms_v2.Compose(
            [
                transforms_v2.RandomResizedCrop(
                    224, scale=(min_glob_scale, 1.0), interpolation=transforms_v2.InterpolationMode.BICUBIC
                ),
                transforms_v2.RandomHorizontalFlip(p=0.5),
                transforms_v2.ToDtype(torch.float32, scale=True),  # values are scaled to [0, 1] here
                transforms_v2.RandomApply(
                    [
                        transforms_v2.ColorJitter(
                            brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1
                        )
                    ],
                    p=0.8,
                ),
                transforms_v2.RandomGrayscale(p=0.2),
                transforms_v2.RandomApply([
                    transforms_v2.GaussianBlur(kernel_size=23, sigma=(0.1, 2.0))
                ], p=0.1), 
                transforms_v2.RandomSolarize(threshold=0.5, p=0.2), 
                transforms_v2.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )
        augs_local = transforms_v2.Compose(
            [
                transforms_v2.RandomResizedCrop(
                    loc_crop_size, scale=(min_loc_scale, max_loc_scale), interpolation=InterpolationMode.BICUBIC
                ),
                transforms_v2.RandomHorizontalFlip(p=0.5),
                transforms_v2.ToDtype(torch.float32, scale=True),  # values are scaled to [0, 1] here
                transforms_v2.RandomApply(
                    [
                        transforms_v2.ColorJitter(
                            brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1
                        )
                    ],
                    p=0.8,
                ),
                transforms_v2.RandomGrayscale(p=0.2),
                transforms_v2.RandomApply([
                    transforms_v2.GaussianBlur(kernel_size=23, sigma=(0.1, 2.0))
                ], p=0.5),
                transforms_v2.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )
        augs_cutout = transforms_v2.Compose(
            [
                transforms_v2.RandomResizedCrop(
                    224, scale=(0.25, 1.0), interpolation=transforms_v2.InterpolationMode.BICUBIC
                ),
                transforms_v2.RandomHorizontalFlip(p=0.5),
                transforms_v2.ToDtype(torch.float32, scale=True),  # values are scaled to [0, 1] here
                RandomCutout(scale=cutout_range, fill=(0.485, 0.456, 0.406)),
                transforms_v2.RandomApply(
                    [
                        transforms_v2.ColorJitter(
                            brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1
                        )
                    ],
                    p=0.8,
                ),
                transforms_v2.RandomGrayscale(p=0.2),
                transforms_v2.RandomApply([
                    transforms_v2.GaussianBlur(kernel_size=23, sigma=(0.1, 2.0))
                ], p=0.5),
                transforms_v2.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

        # for ablation study
        augs_global_more = transforms_v2.Compose(
            [
                transforms_v2.RandomResizedCrop(
                    224, scale=(min_glob_scale, 1.0), interpolation=transforms_v2.InterpolationMode.BICUBIC
                ),
                transforms_v2.RandomHorizontalFlip(p=0.5),
                transforms_v2.ToDtype(torch.float32, scale=True),  # values are scaled to [0, 1] here
                transforms_v2.RandomApply(
                    [
                        transforms_v2.ColorJitter(
                            brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1
                        )
                    ],
                    p=0.8,
                ),
                transforms_v2.RandomGrayscale(p=0.2),
                transforms_v2.RandomApply([
                    transforms_v2.GaussianBlur(kernel_size=23, sigma=(0.1, 2.0))
                ], p=0.5),
                transforms_v2.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

        self.transforms_dict = {
            "global_1": augs_g1,
            "global_2": augs_g2,
        }
        if "local" in tasks:
            self.transforms_dict["local"] = augs_local
        if "cutout" in tasks:
            self.transforms_dict["cutout"] = augs_cutout
        if "global_more" in tasks:
            self.transforms_dict["global_more"] = augs_global_more

    def __call__(self, sample):
        sample_img = transforms_v2.functional.to_image(sample)
        views_list = [
            self.transforms_dict["global_1"](sample_img),
            self.transforms_dict["global_2"](sample_img)
        ]
        for task, num_views in zip(self.tasks[1:], self.num_views_per_task[1:]):
            for _ in range(num_views):
                views_list.append(self.transforms_dict[task](sample_img))
        return views_list


def get_train_transforms(args):
    version = args.transform
    if version == "byol":  # BYOL's default
        return TrainTransformBYOL()
    elif version == "simsiam":  # SimSiam's default
        return TrainTransformSimSiam()
    elif version == "mocov3":  # MoCov3's default
        return TrainTransformMoCov3()
    elif version == "no_crop":
        warnings.warn("Using 'no-crop' transform! This is only meant for ablation studies.")
        return TrainTransformNoCrop()
    elif version == "only_cutout_asym":
        warnings.warn("Using only cutout (asym) transform! This is only meant for ablation studies.")
        return TrainTransformOnlyCutoutAsym(
            cutout_range=args.cutout_range,
        )
    elif version == "only_cutout":
        warnings.warn("Using only cutout (symmetric) transform! This is only meant for ablation studies.")
        return TrainTransformOnlyCutout(
            cutout_range=args.cutout_range,
        )
    elif version == "only_crop":
        warnings.warn("Using only crop (symmetric) transform! This is only meant for ablation studies.")
        return TrainTransformOnlyCrop()
    elif version == "multitask":
        return TrainTransformMultiTask(
            tasks=args.tasks,
            num_views_per_task=args.num_views_per_task,
            cutout_range=args.cutout_range,
            min_glob_scale=args.min_glob_scale,
            min_loc_scale=args.min_loc_scale,
            max_loc_scale=args.max_loc_scale,
            loc_crop_size=args.loc_crop_size
        )
    else:
        raise ValueError(f"Unknown version {version}")
