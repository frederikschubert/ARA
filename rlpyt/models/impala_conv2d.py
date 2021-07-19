from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from nfnets import WSConv2d


class ResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv1 = WSConv2d(
            in_channels, out_channels, kernel_size=3, stride=1, padding=1
        )
        self.conv2 = WSConv2d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1
        )

    def forward(self, x: torch.Tensor):
        y = F.leaky_relu(x)
        y = self.conv1(y)
        y = F.leaky_relu(y)
        y = self.conv2(y)
        return y + x


class Block(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv = WSConv2d(
            in_channels, out_channels, kernel_size=3, stride=1, padding=1
        )
        self.max = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.res1 = ResidualBlock(out_channels, out_channels)
        self.res2 = ResidualBlock(out_channels, out_channels)

    def forward(self, x: torch.Tensor):
        y = self.conv(x)
        y = self.max(y)
        y = self.res1(y)
        y = self.res2(y)
        return y


class ImpalaConv2d(nn.Module):
    def __init__(
        self,
        image_shape: Tuple[int, int, int],
        output_size: int = 512,
        depths: List[int] = [16, 32, 32],
        k: int = 1,
    ):
        super().__init__()
        c, h, w = image_shape
        self.output_size = output_size
        self.depths = depths
        self.k = k
        self.block1 = Block(c, depths[0] * k)
        self.block2 = Block(depths[0] * k, depths[1] * k)
        self.block3 = Block(depths[1] * k, depths[2] * k)
        self.fc1 = nn.Linear(
            (depths[2] * k * h * w) // (2 ** len(depths) * 2 ** len(depths)),
            output_size,
        )

    def forward(self, x: torch.Tensor):
        y = self.block1(x)
        y = self.block2(y)
        y = self.block3(y)
        y = torch.flatten(y, start_dim=1)
        y = F.leaky_relu(y)
        y = self.fc1(y)
        return y

