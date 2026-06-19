# Self-Supervised Learning with a Multi-Task Latent Space Objective (MULAN)

Official PyTorch implementation of the paper  
**"Self-Supervised Learning with a Multi-Task Latent Space Objective"**.

📄 Paper: https://arxiv.org/abs/2602.05845

![MULAN overview](figs/mulan_overview.jpg)

---

# Pretrained Models

Linear evaluation results on **ImageNet-1k**.

| Method             | Backbone  | Epochs | Lin. Acc. (%) | Backbone weights                                                                                        |
|--------------------|-----------|--------|---------------|---------------------------------------------------------------------------------------------------------|
| SimSiam multi-task | ResNet-50 | 200    | 74.7          | [Link](https://github.com/pfdp0/mulan/releases/download/v2.0/simsiam_multitask_resnet50_200ep_ckpt.pth) |
| MoCo v3 multi-task | ResNet-50 | 200    | 75.7          | [Link](https://github.com/pfdp0/mulan/releases/download/v2.0/mocov3_multitask_resnet50_200ep_ckpt.pth)  |
| BYOL multi-task    | ResNet-50 | 200    | 75.6          | [Link](https://github.com/pfdp0/mulan/releases/download/v2.0/byol_multitask_resnet50_200ep_ckpt.pth)    |
| BYOL multi-task    | ResNet-50 | 800    | 76.7          | [Link](https://github.com/pfdp0/mulan/releases/download/v2.0/byol_multitask_resnet50_800ep_ckpt.pth)    |
| BYOL multi-task    | ViT-S     | 200    | 74.5          | [Link](https://github.com/pfdp0/mulan/releases/download/v2.0/byol_multitask_vits_200ep_tgt_ckpt.pth)    |
| BYOL multi-task    | ViT-B     | 200    | 78.3          | [Link](https://github.com/pfdp0/mulan/releases/download/v2.0/byol_multitask_vitb_200ep_tgt_ckpt.pth)    |

---

# Installation

### Option 1: pip

```bash
pip install \
    torch==2.10 \
    torchvision \
    torchaudio \
    detectron2 \
    tqdm \
    matplotlib \
    wandb \
    scipy \
    scikit-learn
```

### Option 2: mamba

```bash
mamba create -n mulan_env python=3.12 pip
mamba activate mulan_env
mamba install pytorch==2.10 torchvision torchaudio detectron2 tqdm matplotlib wandb scipy anaconda::scikit-learn
```

> `detectron2` is only required for COCO detection and instance segmentation experiments.

---

# Setup

- Download ImageNet-1k and define the paths to the data in `config.py` under `IMAGENET_ROOT_DIR`.
- Also define the paths to the output directory in `config.py` under `EXP_ROOT` 
- Optionally set the W&B project in `config.py` under `WANDB_PROJECT` and `WANDB_ENTITY`.

---

# Usage

## Pre-training

Pre-training a ResNet-50 BYOL with multi-task latent space objective on ImageNet-1k for 200 epochs using 4 GPUs:
```bash
torchrun --standalone --nproc_per_node 4 main.py --training-fn byol_multitask --transform multitask --tasks global local cutout --num-views-per-task 2 2 1 --task-weights 1.0 1.0 1.0
```

Or with a single GPU:
```bash
python main.py --batch-size 512 --training-fn byol_multitask --transform multitask --tasks global local cutout --num-views-per-task 2 2 1 --task-weights 1.0 1.0 1.0
```

> Expected peak GPU memory usage: ~24 GB.

For additional configuration options, see [CONFIG.md](CONFIG.md).

---

## Linear evaluation

Evaluate a pretrained ResNet-50 on ImageNet-1k:

```bash
python linear_eval.py --backbone resnet50 --pretrained PATH/TO/PRETRAINED/WEIGHTS.pth --normalize-features --batch-size 512 --wd 0.0 --lr-head 0.005
```

---
## Transfer Learning on COCO

### Step 1: Convert weights to Detectron2 format
```bash
python detection/convert-pretrain-to-detectron2.py PATH/TO/PRETRAINED/WEIGHTS.pth PATH/TO/OUTPUT/DETECTRON2_WEIGHTS.pkl
```

### Step 2: Fine-tune Mask R-CNN
```bash
python detection/train_coco.py --config-file ./configs/coco_R50_MaskRCNN-FPN_ssl_L.yaml --num-gpus 4 MODEL.WEIGHTS PATH/TO/DETECTRON2_WEIGHTS.pkl
```

---
## Citation

If you find this repository useful in your research, please cite:

```
@article{deplaen2026mulan,
      title={Self-Supervised Learning with a Multi-Task Latent Space Objective}, 
      author={De Plaen, Pierre-François and Jha, Abhishek and Van Gool, Luc and Tuytelaars, Tinne and Proesmans, Marc},
      journal={arXiv preprint arXiv:2602.05845},
      year={2026}
}
```
