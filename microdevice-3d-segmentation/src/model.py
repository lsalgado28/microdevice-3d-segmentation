"""
A compact 3D U-Net for volumetric segmentation.

Scaled down from the standard U-Net/nnU-Net design (fewer channels, 3
downsampling stages instead of 4-5) to train quickly on small synthetic
volumes -- the same encoder-decoder-with-skip-connections architecture
used by full-scale nnU-Net, just lighter weight.
"""

import torch
import torch.nn as nn


def conv_block(in_ch, out_ch):
    return nn.Sequential(
        nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=1),
        nn.InstanceNorm3d(out_ch),
        nn.ReLU(inplace=True),
        nn.Conv3d(out_ch, out_ch, kernel_size=3, padding=1),
        nn.InstanceNorm3d(out_ch),
        nn.ReLU(inplace=True),
    )


class UNet3D(nn.Module):
    def __init__(self, in_channels: int = 1, out_channels: int = 1, base_channels: int = 16):
        super().__init__()
        c = base_channels

        # Encoder
        self.enc1 = conv_block(in_channels, c)
        self.enc2 = conv_block(c, c * 2)
        self.enc3 = conv_block(c * 2, c * 4)
        self.pool = nn.MaxPool3d(2)

        # Bottleneck
        self.bottleneck = conv_block(c * 4, c * 8)

        # Decoder
        self.up3 = nn.ConvTranspose3d(c * 8, c * 4, kernel_size=2, stride=2)
        self.dec3 = conv_block(c * 8, c * 4)
        self.up2 = nn.ConvTranspose3d(c * 4, c * 2, kernel_size=2, stride=2)
        self.dec2 = conv_block(c * 4, c * 2)
        self.up1 = nn.ConvTranspose3d(c * 2, c, kernel_size=2, stride=2)
        self.dec1 = conv_block(c * 2, c)

        self.out_conv = nn.Conv3d(c, out_channels, kernel_size=1)

    def encode(self, x):
        """Returns the bottleneck representation -- used for self-supervised pretraining."""
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        b = self.bottleneck(self.pool(e3))
        return b, (e1, e2, e3)

    def forward(self, x):
        b, (e1, e2, e3) = self.encode(x)

        d3 = self.up3(b)
        d3 = self.dec3(torch.cat([d3, e3], dim=1))

        d2 = self.up2(d3)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))

        d1 = self.up1(d2)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))

        return self.out_conv(d1)  # raw logits, shape (B, 1, D, H, W)


class ReconstructionDecoder(nn.Module):
    """Lightweight decoder used only during self-supervised pretraining, to
    reconstruct the input volume from the encoder's bottleneck features
    (a denoising-autoencoder-style pretext task)."""

    def __init__(self, base_channels: int = 16):
        super().__init__()
        c = base_channels
        self.up3 = nn.ConvTranspose3d(c * 8, c * 4, kernel_size=2, stride=2)
        self.dec3 = conv_block(c * 4, c * 4)
        self.up2 = nn.ConvTranspose3d(c * 4, c * 2, kernel_size=2, stride=2)
        self.dec2 = conv_block(c * 2, c * 2)
        self.up1 = nn.ConvTranspose3d(c * 2, c, kernel_size=2, stride=2)
        self.dec1 = conv_block(c, c)
        self.out_conv = nn.Conv3d(c, 1, kernel_size=1)

    def forward(self, bottleneck):
        d3 = self.dec3(self.up3(bottleneck))
        d2 = self.dec2(self.up2(d3))
        d1 = self.dec1(self.up1(d2))
        return torch.sigmoid(self.out_conv(d1))
