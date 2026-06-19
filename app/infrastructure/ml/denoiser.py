"""
Denoiser Model Wrapper.

Replace the DenoiserNet class body with your actual optimized
noise-reduction model architecture.

The registry calls:
    model = DenoiserNet()
    model.load_state_dict(torch.load(path))
    model.eval()
    output = model(input_tensor)   # [B, 3, H, W] → [B, 3, H, W]
"""
import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)          # residual (learn the noise)


class DenoiserNet(nn.Module):
    """
    Lightweight residual denoiser.
    Input:  noisy image  [B, 3, H, W]  float32, values in [0,1]
    Output: clean image  [B, 3, H, W]  float32, values in [0,1]

    Swap this body for your architecture. Keep the same in/out contract.
    """

    def __init__(self, num_residual_blocks: int = 8, features: int = 64):
        super().__init__()
        self.head = nn.Sequential(
            nn.Conv2d(3, features, 3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.body = nn.Sequential(
            *[ResidualBlock(features) for _ in range(num_residual_blocks)]
        )
        self.tail = nn.Conv2d(features, 3, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.head(x)
        feat = self.body(feat)
        noise_estimate = self.tail(feat)
        return torch.clamp(x - noise_estimate, 0.0, 1.0)   # subtract estimated noise
