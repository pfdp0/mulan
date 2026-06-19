# Configurations

## Baselines

<details>
<summary>BYOL ResNet-50 baseline (200 epochs)</summary>

```bash
torchrun --standalone --nproc_per_node 4 main.py
```
</details>

<details>
<summary>SimSiam ResNet-50 baseline (200 epochs)</summary>

```bash
torchrun --standalone --nproc_per_node 4 main.py --batch-size 512 --training-fn simsiam_default --transform simsiam --use-proj-output-bn --use-constant-pred-lr --use-per-epoch-sched --num-warmup-epochs 0 --optimizer sgd --base-lr 0.05 --wd 1e-4 --projector-mlp 2048-2048-2048 --predictor-mlp 512-2048
```
</details>

<details>
<summary>MoCo v3 ResNet-50 baseline (200 epochs)</summary>

```bash
torchrun --standalone --nproc_per_node 4 main.py --training-fn mocov3_default --transform mocov3 --use-proj-output-bn --use-pred-output-bn
```
</details>

## Multi-task Latent Space Objective

<details>
<summary>BYOL ResNet-50 multi-task (200 epochs)</summary>

```bash
torchrun --standalone --nproc_per_node 4 main.py --training-fn byol_multitask --transform multitask --tasks global local cutout --num-views-per-task 2 2 1 --task-weights 1.0 1.0 1.0
```
</details>

<details>
<summary>BYOL ResNet-50 multi-task (800 epochs)</summary>

```bash
torchrun --standalone --nproc_per_node 8 main.py --training-fn byol_multitask --transform multitask --tasks global local cutout --num-views-per-task 2 4 1 --task-weights 1.0 1.0 1.0 --epochs 800
```
> Note that this run uses 8 GPUs to speed up training. 

> We used 4 local views for this run. However, it does not seem to improve performance compared to using 2 local views, so we recommend sticking to 2 local views.
</details>

<details>
<summary>BYOL ViT-S multi-task (200 epochs)</summary>

```bash
torchrun --standalone --nproc_per_node 4 main.py --arch vit_small --projector-mlp 4096-4096-256 --optimizer adamw --adam-beta2 0.98 --base-lr 3e-4 --wd 0.1 --base-target-ema 0.998 --clip-grad 0.5 --drop-path-rate 0.1 --training-fn byol_multitask --transform multitask --tasks global local cutout --num-views-per-task 2 2 1 --task-weights 1.0 1.0 1.0
```
</details>

<details>
<summary>BYOL ViT-B multi-task (200 epochs)</summary>

```bash
torchrun --standalone --nproc_per_node 4 main.py --arch vit_base --projector-mlp 4096-4096-256 --optimizer adamw --adam-beta2 0.98 --base-lr 3e-4 --wd 0.1 --base-target-ema 0.998 --clip-grad 0.5 --drop-path-rate 0.1 --training-fn byol_multitask --transform multitask --tasks global local cutout --num-views-per-task 2 2 1 --task-weights 1.0 1.0 1.0
```
</details>

<details>
<summary>SimSiam ResNet-50 multi-task (200 epochs)</summary>

```bash
torchrun --standalone --nproc_per_node 4 main.py --batch-size 512 --training-fn simsiam_multitask --transform multitask --use-proj-output-bn --use-constant-pred-lr --use-per-epoch-sched --num-warmup-epochs 0 --optimizer sgd --base-lr 0.05 --wd 0.0001 --projector-mlp 2048-2048-2048 --predictor-mlp 512-2048 --tasks global local cutout --num-views-per-task 2 2 1 --task-weights 0.333 0.333 0.333
```
</details>

<details>
<summary>MoCo v3 ResNet-50 multi-task (200 epochs)</summary>

```bash
torchrun --standalone --nproc_per_node 4 main.py --batch-size 1024 --training-fn mocov3_multitask --transform multitask --use-proj-output-bn --use-pred-output-bn --base-target-ema 0.996 --tasks global local cutout --num-views-per-task 2 2 1 --task-weights 1.0 1.0 1.0
```
</details>