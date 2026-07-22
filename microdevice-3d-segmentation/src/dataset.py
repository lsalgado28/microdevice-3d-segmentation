"""PyTorch Dataset for synthetic 3D micro-device volumes and segmentation masks."""

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

DATA_DIR = Path(__file__).parent.parent / "data"


class MicroDeviceVolumeDataset(Dataset):
    """
    Returns (volume, mask) pairs, each shaped (1, D, H, W) as float tensors.
    """

    def __init__(self, indices=None):
        self.volumes = np.load(DATA_DIR / "volumes.npy")
        self.masks = np.load(DATA_DIR / "masks.npy")
        self.indices = indices if indices is not None else np.arange(len(self.volumes))

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        real_idx = self.indices[idx]
        volume = torch.from_numpy(self.volumes[real_idx]).unsqueeze(0)  # (1, D, H, W)
        mask = torch.from_numpy(self.masks[real_idx]).unsqueeze(0)      # (1, D, H, W)
        return volume, mask
