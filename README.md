# 3D Micro-Device Segmentation (U-Net + Self-Supervised Pretraining)

A PyTorch 3D U-Net for segmenting small elongated objects ("micro-devices")
from volumetric imaging data, with a self-supervised denoising-reconstruction
pretraining stage and evaluation via Dice coefficient and Hausdorff distance.

This is a generalized, from-scratch version of a 3D segmentation pipeline I
built during an undergraduate research position on micro-device localization,
rebuilt here on synthetic public-style data so the architecture and training
code are fully shareable.

## Problem

Given a noisy 3D volume (e.g., from micro-CT or confocal microscopy),
segment the voxels belonging to a small rod/capsule-shaped device embedded
in a noisy background medium. Framed as dense binary voxel classification.

## Approach

1. **Synthetic data generation** (`data/generate_synthetic_volumes.py`):
   32×32×32 volumes containing a randomly positioned, randomly oriented
   capsule-shaped object with distinct intensity/texture from a noisy
   background, simulating realistic volumetric imaging data. Swap in real
   data (e.g., the [Medical Segmentation
   Decathlon](http://medicaldecathlon.com/)) by matching the same
   `(volume, mask)` array schema.

2. **Self-supervised pretraining** (`src/pretrain.py`): before any labels are
   used, the U-Net encoder is pretrained on a denoising-reconstruction
   pretext task — recovering clean volumes from noise-corrupted input. This
   lets the encoder learn useful volumetric features before segmentation
   fine-tuning, which matters most when labeled data is scarce relative to
   available unlabeled volumes (a common constraint in real imaging pipelines).

3. **Segmentation model** (`src/model.py`): a compact 3D U-Net —
   encoder-decoder with skip connections, the same core architecture used by
   nnU-Net, scaled down (fewer channels, 3 downsampling stages) to train
   quickly on small volumes.

4. **Fine-tuning** (`src/train.py`): the pretrained encoder is loaded into
   the full U-Net and fine-tuned end-to-end on the segmentation task using a
   soft Dice loss, which handles the severe foreground/background class
   imbalance typical of small-object segmentation (~1% of voxels are
   foreground here) far better than plain cross-entropy.

5. **Evaluation** (`src/losses_metrics.py`): reproducible benchmarking via

   - **Dice coefficient**: overlap-based accuracy metric, standard for
     segmentation tasks.
   - **Hausdorff distance**: worst-case boundary error (via Euclidean
     distance transform), which catches shape/boundary errors that Dice can
     miss — important when precise device localization matters, not just
     approximate overlap.

## Project structure

```
microdevice-3d-segmentation/
├── data/
│   └── generate_synthetic_volumes.py   # synthetic 3D volume + mask generator
├── src/
│   ├── dataset.py                      # PyTorch Dataset for volumes/masks
│   ├── model.py                        # 3D U-Net + reconstruction decoder
│   ├── losses_metrics.py               # Dice loss, Dice coefficient, Hausdorff distance
│   ├── pretrain.py                     # self-supervised pretraining
│   ├── train.py                        # supervised fine-tuning + evaluation
│   └── evaluate.py                     # mid-slice prediction visualization
├── models/                             # saved checkpoints (created on run)
├── plots/                              # output visualizations (created on run)
└── requirements.txt
```

## Usage

```bash
pip install -r requirements.txt

# 1. Generate the synthetic dataset
python data/generate_synthetic_volumes.py

# 2. Self-supervised pretraining of the encoder
python src/pretrain.py

# 3. Fine-tune on segmentation + evaluate (Dice, Hausdorff)
python src/train.py

# 4. Visualize predictions vs. ground truth
python src/evaluate.py
```

Training is CPU-feasible but slow (3D convolutions are compute-heavy); a GPU
is recommended for the full epoch counts in `train.py`/`pretrain.py`. Reduce
`EPOCHS` in either script for a faster smoke test.

## Notes / next steps

- The synthetic foreground class is intentionally small (~1% of voxels),
  matching realistic small-object segmentation difficulty. If Dice scores
  plateau, try increasing pretraining epochs or adding data augmentation
  (random rotation/flips) to the training loop.
- `nnU-Net` in practice also handles automatic preprocessing, patch-based
  training for larger volumes, and ensembling across folds — this repo
  focuses on the core encoder-decoder-with-skip-connections architecture and
  training loop rather than nnU-Net's full auto-configuration pipeline.
- A natural extension is multi-class segmentation (e.g., distinguishing
  device sub-components) by increasing `out_channels` in `UNet3D` and
  switching to a multi-class Dice/cross-entropy loss.
