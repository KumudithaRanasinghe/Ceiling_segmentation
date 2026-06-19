"""
U-Net Architecture for Roof Segmentation.

6 output classes by default (set NUM_CLASSES in .env to match your training):
  0 = background
  1 = gypsum_board
  2 = modular_grid
  3 = suspended
  4 = pvc_panel
  5 = wood_panel

If your segmentation_optimized.pth was trained with a different architecture,
replace this class body with your own — keep the same __init__ signature so
the registry can instantiate it.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class Down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.pool_conv = nn.Sequential(nn.MaxPool2d(2), DoubleConv(in_ch, out_ch))

    def forward(self, x):
        return self.pool_conv(x)


class Up(nn.Module):
    def __init__(self, in_ch, out_ch, bilinear=True):
        super().__init__()
        if bilinear:
            self.up   = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
            self.conv = DoubleConv(in_ch, out_ch)
        else:
            self.up   = nn.ConvTranspose2d(in_ch // 2, in_ch // 2, 2, stride=2)
            self.conv = DoubleConv(in_ch, out_ch)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        dh = x2.size(2) - x1.size(2)
        dw = x2.size(3) - x1.size(3)
        x1 = F.pad(x1, [dw // 2, dw - dw // 2, dh // 2, dh - dh // 2])
        return self.conv(torch.cat([x2, x1], dim=1))


class UNet(nn.Module):
    """
    Args:
        in_channels:  always 3 (RGB)
        num_classes:  must match your training — set via NUM_CLASSES in .env
    """

    def __init__(self, in_channels: int = 6, num_classes: int = 6, bilinear: bool = True):
        super().__init__()
        f = 2 if bilinear else 1
        self.inc   = DoubleConv(in_channels, 64)
        self.down1 = Down(64,  128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 1024 // f)
        self.up1   = Up(1024, 512  // f, bilinear)
        self.up2   = Up(512,  256  // f, bilinear)
        self.up3   = Up(256,  128  // f, bilinear)
        self.up4   = Up(128,  64,        bilinear)
        self.outc  = nn.Conv2d(64, num_classes, kernel_size=1)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x  = self.up1(x5, x4)
        x  = self.up2(x,  x3)
        x  = self.up3(x,  x2)
        x  = self.up4(x,  x1)
        return self.outc(x)    # raw logits [B, num_classes, H, W]