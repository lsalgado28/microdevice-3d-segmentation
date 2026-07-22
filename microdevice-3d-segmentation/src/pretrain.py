"""
Self-supervised pretraining for the U-Net encoder using a denoising
reconstruction pretext task: the encoder + a lightweight reconstruction
decoder learn to recover clean volumes from noise-corrupted input, with no
segmentation labels used at all.

The pretrained encoder weights are then loaded into the full U-Net before
supervised fine-tuning in train.py -- this is the same self-supervised
pretraining pattern used in medical/scientific imaging when labeled data
is scarce relative to unlabeled volumes.

Usage:
    python src/pretrain.py
"""

import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.append(str(Path(__file__).parent))
from dataset import MicroDeviceVolumeDataset
from model import UNet3D, ReconstructionDecoder

MODEL_DIR = Path(__file__).parent.parent / "models"
EPOCHS = 15
BATCH_SIZE = 8
LR = 1e-3
NOISE_STD = 0.15


def main():
    dataset = MicroDeviceVolumeDataset()
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder = UNet3D().to(device)
    recon_decoder = ReconstructionDecoder().to(device)

    params = list(encoder.parameters()) + list(recon_decoder.parameters())
    optimizer = torch.optim.Adam(params, lr=LR)
    loss_fn = torch.nn.MSELoss()

    print(f"Self-supervised pretraining on {len(dataset)} volumes | device={device}")

    for epoch in range(1, EPOCHS + 1):
        encoder.train()
        recon_decoder.train()
        epoch_loss = 0.0

        for volume, _mask in loader:
            volume = volume.to(device)
            noisy = volume + torch.randn_like(volume) * NOISE_STD
            noisy = noisy.clamp(0, 1)

            optimizer.zero_grad()
            bottleneck, _skips = encoder.encode(noisy)
            reconstruction = recon_decoder(bottleneck)

            loss = loss_fn(reconstruction, volume)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * volume.size(0)

        epoch_loss /= len(dataset)
        if epoch % 3 == 0 or epoch == 1:
            print(f"  epoch {epoch:2d}/{EPOCHS}  reconstruction_mse={epoch_loss:.5f}")

    MODEL_DIR.mkdir(exist_ok=True)
    torch.save(encoder.state_dict(), MODEL_DIR / "pretrained_encoder.pt")
    print(f"\nSaved pretrained encoder weights to {MODEL_DIR / 'pretrained_encoder.pt'}")


if __name__ == "__main__":
    main()
