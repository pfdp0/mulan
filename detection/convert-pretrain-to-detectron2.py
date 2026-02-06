# MIT License
# Copyright (c) 2026 MULAN authors
# Inspired by MoCo (https://github.com/facebookresearch/moco).


import pickle as pkl
import sys

import torch


if __name__ == "__main__":
    input = sys.argv[1]

    obj = torch.load(input, map_location="cpu", weights_only=False)
    # obj = obj["online_model"]

    newmodel = {}
    for k, v in obj.items():
        old_k = k
        k = k.replace("module.backbone.", "")
        if "layer" not in k:
            k = "stem." + k
        for t in [1, 2, 3, 4]:
            k = k.replace("layer{}".format(t), "res{}".format(t + 1))
        for t in [1, 2, 3]:
            k = k.replace("bn{}".format(t), "conv{}.norm".format(t))
        k = k.replace("downsample.0", "shortcut")
        k = k.replace("downsample.1", "shortcut.norm")
        print(old_k, "->", k)
        newmodel[k] = v.numpy()

    res = {"model": newmodel, "__author__": "BYOL_multitask", "matching_heuristics": True}

    with open(sys.argv[2], "wb") as f:
        pkl.dump(res, f)
