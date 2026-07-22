"""
Trains the 3D U-Net for micro-device segmentation, optionally initializing
the encoder from self-supervised pretrained weights (src/pretrain.py).
Evaluates on a held-out test split using Dice coefficient and Hausdorff
distance -- standard metrics for 3D medical/scientific image segmentation.

Usage:
    python src/pretrain.py        # optional but recommended first
    python src/train.py
"""

import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

sys.path.append(str(Path(__file__).parent))
from dataset import MicroDeviceVolumeDataset
from model import UNet3D
from losses_metrics import DiceLoss, dice_coefficient, hausdorff_distance

MODEL_DIR = Path(__file__).parent.parent / "models"
PRETRAINED_PATH = MODEL_DIR / "pretrained_encoder.pt"

EPOCHS = 25
BATCH_SIZE = 8
LR = 1e-3
TRAIN_FRACTION = 0.8
BINARIZE_THRESHOLD = 0.5


def main():
    full_dataset = MicroDeviceVolumeDataset()
    n = len(full_dataset)
    indices = np.random.RandomState(0).permutation(n)
    split = int(n * TRAIN_FRACTION)
    train_idx, test_idx = indices[:split], indices[split:]

    train_ds = Subset(full_dataset, train_idx)
    test_ds = Subset(full_dataset, test_idx)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet3D().to(device)

    used_pretrained = False
    if PRETRAINED_PATH.exists():
        model.load_state_dict(torch.load(PRETRAINED_PATH, map_location=device, weights_only=True))
        used_pretrained = True
        print(f"Initialized encoder from self-supervised pretrained weights: {PRETRAINED_PATH}")
    else:
        print("No pretrained weights found -- training from random initialization. "
              "Run src/pretrain.py first to enable self-supervised pretraining.")

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    dice_loss_fn = DiceLoss()

    print(f"Fine-tuning on {len(train_ds)} volumes | testing on {len(test_ds)} | device={device}")

    for epoch in range(1, EPOCHS + 1):
        model.train()
        epoch_loss = 0.0
        for volume, mask in train_loader:
            volume, mask = volume.to(device), mask.to(device)
            optimizer.zero_grad()
            logits = model(volume)
            loss = dice_loss_fn(logits, mask)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * volume.size(0)
        epoch_loss /= len(train_ds)

        if epoch % 5 == 0 or epoch == 1:
            print(f"  epoch {epoch:2d}/{EPOCHS}  dice_loss={epoch_loss:.4f}")

    # --- Evaluation ---
    model.eval()
    dice_scores, hausdorff_scores = [], []
    with torch.no_grad():
        for volume, mask in test_loader:
            volume = volume.to(device)
            logits = model(volume)
            probs = torch.sigmoid(logits).cpu().numpy()[0, 0]
            pred_binary = probs > BINARIZE_THRESHOLD
            true_binary = mask.numpy()[0, 0].astype(bool)

            dice_scores.append(dice_coefficient(pred_binary, true_binary))
            hd = hausdorff_distance(pred_binary, true_binary)
            if not np.isnan(hd):
                hausdorff_scores.append(hd)

    print("\n--- Evaluation on held-out test set ---")
    print(f"  Pretrained encoder used: {used_pretrained}")
    print(f"  Mean Dice coefficient:      {np.mean(dice_scores):.4f}  (std {np.std(dice_scores):.4f})")
    if hausdorff_scores:
        print(f"  Mean Hausdorff distance:    {np.mean(hausdorff_scores):.3f} voxels  (std {np.std(hausdorff_scores):.3f})")
    else:
        print("  Mean Hausdorff distance:    undefined (no valid predictions)")

    MODEL_DIR.mkdir(exist_ok=True)
    torch.save(model.state_dict(), MODEL_DIR / "segmentation_unet3d.pt")
    print(f"\nSaved fine-tuned model to {MODEL_DIR / 'segmentation_unet3d.pt'}")


if __name__ == "__main__":
    main()
