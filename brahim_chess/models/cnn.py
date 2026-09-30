"""Residual CNN over spatial board tensors (EXP-13)."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.relu(out + residual)


class BrahimCloneCNN(nn.Module):
    def __init__(self, in_channels: int = 14, channels: int = 64, num_blocks: int = 6):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(),
        )
        self.resnet = nn.Sequential(*[ResidualBlock(channels) for _ in range(num_blocks)])
        self.start_head = nn.Conv2d(channels, 1, kernel_size=1)
        self.end_head = nn.Conv2d(channels, 1, kernel_size=1)

    def forward(self, x: torch.Tensor):
        x = self.resnet(self.stem(x))
        start_logits = self.start_head(x).flatten(1)
        end_logits = self.end_head(x).flatten(1)
        return start_logits, end_logits
