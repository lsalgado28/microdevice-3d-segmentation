"""
Generates synthetic 3D volumetric imaging data with embedded "micro-device"
objects and ground-truth segmentation masks.

This simulates the kind of data a real micro-device localization task would
involve: a noisy 3D volume (e.g., from micro-CT, confocal microscopy, or
similar volumetric imaging) containing a small rod/capsule-shaped object
that needs to be segmented from background tissue/medium.

Public real-world alternatives you could swap in instead of synthetic data:
  - Medical Segmentation Decathlon (http://medicaldecathlon.com/)
  - EM segmentation challenge datasets (e.g., ISBI EM Segmentation)
"""

import numpy as np
from pathlib import Path
from scipy.ndimage import gaussian_filter

np.random.seed(42)

VOLUME_SIZE = 32       # 32x32x32 voxel volumes (kept small for fast training)
N_SAMPLES = 240


def make_capsule_mask(size: int, center, length: float, radius: float, axis_dir):
    """Creates a capsule (rod-with-rounded-ends) binary mask -- a reasonable
    stand-in for an elongated micro-device shape."""
    zz, yy, xx = np.meshgrid(np.arange(size), np.arange(size), np.arange(size), indexing="ij")
    coords = np.stack([zz, yy, xx], axis=-1).astype(np.float32) - np.array(center)

    axis_dir = axis_dir / np.linalg.norm(axis_dir)
    proj_len = coords @ axis_dir
    proj_len_clipped = np.clip(proj_len, -length / 2, length / 2)
    closest_point = proj_len_clipped[..., None] * axis_dir
    dist_to_axis = np.linalg.norm(coords - closest_point, axis=-1)

    mask = dist_to_axis <= radius
    return mask


def generate_sample(size: int = VOLUME_SIZE):
    center = np.array([size / 2, size / 2, size / 2]) + np.random.uniform(-3, 3, 3)
    length = np.random.uniform(size * 0.35, size * 0.55)
    radius = np.random.uniform(size * 0.06, size * 0.11)
    axis_dir = np.random.normal(0, 1, 3)

    mask = make_capsule_mask(size, center, length, radius, axis_dir).astype(np.float32)

    # Background: smooth random noise (simulates tissue/medium texture)
    background = gaussian_filter(np.random.normal(0.3, 0.08, (size, size, size)), sigma=1.2)

    # Device intensity: brighter than background, with its own texture/noise
    device_intensity = gaussian_filter(np.random.normal(0.75, 0.05, (size, size, size)), sigma=0.8)

    volume = background * (1 - mask) + device_intensity * mask

    # Additive imaging noise
    volume += np.random.normal(0, 0.04, (size, size, size))
    volume = np.clip(volume, 0, 1).astype(np.float32)

    return volume, mask


def generate_dataset(n_samples: int = N_SAMPLES):
    volumes = np.zeros((n_samples, VOLUME_SIZE, VOLUME_SIZE, VOLUME_SIZE), dtype=np.float32)
    masks = np.zeros((n_samples, VOLUME_SIZE, VOLUME_SIZE, VOLUME_SIZE), dtype=np.float32)
    for i in range(n_samples):
        v, m = generate_sample()
        volumes[i] = v
        masks[i] = m
    return volumes, masks


if __name__ == "__main__":
    volumes, masks = generate_dataset()
    out_dir = Path(__file__).parent
    np.save(out_dir / "volumes.npy", volumes)
    np.save(out_dir / "masks.npy", masks)
    print(f"Generated {len(volumes)} volumes of shape {volumes.shape[1:]}")
    print(f"Saved to {out_dir}/volumes.npy and {out_dir}/masks.npy")
    print(f"Mean foreground voxel fraction: {masks.mean():.4f}")
