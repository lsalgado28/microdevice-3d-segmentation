"""Generates a mid-slice visualization comparing predicted vs. ground-truth
segmentation for a few test volumes."""

import sys
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt

sys.path.append(str(Path(__file__).parent))
from dataset import MicroDeviceVolumeDataset
from model import UNet3D

MODEL_PATH = Path(__file__).parent.parent / "models" / "segmentation_unet3d.pt"
PLOT_PATH = Path(__file__).parent.parent / "plots" / "segmentation_examples.png"
N_EXAMPLES = 4


def main():
    dataset = MicroDeviceVolumeDataset()
    model = UNet3D()
    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu", weights_only=True))
    model.eval()

    rng = np.random.RandomState(1)
    sample_indices = rng.choice(len(dataset), N_EXAMPLES, replace=False)

    fig, axes = plt.subplots(N_EXAMPLES, 3, figsize=(9, 3 * N_EXAMPLES))

    with torch.no_grad():
        for row, idx in enumerate(sample_indices):
            volume, mask = dataset[idx]
            logits = model(volume.unsqueeze(0))
            probs = torch.sigmoid(logits)[0, 0].numpy()
            pred = probs > 0.5

            mid = volume.shape[1] // 2
            vol_slice = volume[0, mid].numpy()
            mask_slice = mask[0, mid].numpy()
            pred_slice = pred[mid]

            axes[row, 0].imshow(vol_slice, cmap="gray")
            axes[row, 0].set_title("Input volume (mid-slice)")
            axes[row, 1].imshow(mask_slice, cmap="gray")
            axes[row, 1].set_title("Ground truth mask")
            axes[row, 2].imshow(pred_slice, cmap="gray")
            axes[row, 2].set_title("Predicted mask")
            for ax in axes[row]:
                ax.axis("off")

    plt.tight_layout()
    PLOT_PATH.parent.mkdir(exist_ok=True)
    plt.savefig(PLOT_PATH, dpi=150)
    print(f"Saved visualization to {PLOT_PATH}")


if __name__ == "__main__":
    main()
