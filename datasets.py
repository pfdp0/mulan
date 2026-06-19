# MIT License
# Copyright (c) 2026 MULAN authors

import torchvision.datasets as datasets
from torchvision.io import decode_image, ImageReadMode

class ReturnIndexDataset(datasets.ImageFolder):
    def __init__(self, *args, loader=None, **kwargs):
        super(ReturnIndexDataset, self).__init__(*args, **kwargs)
        # update default loader to avoid intermediate PIL format
        default_v2_loader = lambda path: decode_image(path, mode=ImageReadMode.RGB)
        self.loader = loader if loader is not None else default_v2_loader

    def __getitem__(self, idx):
        """Returns the image and its corresponding index."""
        img, target = super(ReturnIndexDataset, self).__getitem__(idx)
        return img, idx
