"""Dice loss (training) and Dice coefficient / Hausdorff distance (evaluation metrics)."""

import numpy as np
import torch
import torch.nn as nn
from scipy.ndimage import distance_transform_edt


class DiceLoss(nn.Module):
    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, target):
        probs = torch.sigmoid(logits)
        probs_flat = probs.view(probs.size(0), -1)
        target_flat = target.view(target.size(0), -1)

        intersection = (probs_flat * target_flat).sum(dim=1)
        union = probs_flat.sum(dim=1) + target_flat.sum(dim=1)
        dice = (2 * intersection + self.smooth) / (union + self.smooth)
        return 1 - dice.mean()


def dice_coefficient(pred_binary: np.ndarray, true_binary: np.ndarray, smooth: float = 1e-6) -> float:
    intersection = np.logical_and(pred_binary, true_binary).sum()
    total = pred_binary.sum() + true_binary.sum()
    return (2 * intersection + smooth) / (total + smooth)


def hausdorff_distance(pred_binary: np.ndarray, true_binary: np.ndarray) -> float:
    """
    Symmetric Hausdorff distance between two binary masks, computed via
    Euclidean distance transforms (standard approach for 3D segmentation
    evaluation -- avoids the O(n*m) cost of brute-force surface-point
    comparison).

    Returns np.nan if either mask is empty (Hausdorff distance undefined).
    """
    if pred_binary.sum() == 0 or true_binary.sum() == 0:
        return float("nan")

    # Distance from every voxel to nearest true-foreground voxel
    dt_true = distance_transform_edt(~true_binary.astype(bool))
    dt_pred = distance_transform_edt(~pred_binary.astype(bool))

    # Surface (boundary) voxels only, for a meaningful surface distance
    pred_surface = pred_binary.astype(bool)
    true_surface = true_binary.astype(bool)

    d_pred_to_true = dt_true[pred_surface]
    d_true_to_pred = dt_pred[true_surface]

    if len(d_pred_to_true) == 0 or len(d_true_to_pred) == 0:
        return float("nan")

    return float(max(d_pred_to_true.max(), d_true_to_pred.max()))
