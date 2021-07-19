from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from rlpyt.models.utils import conv_out_size, initialize_weights_he


class QuantileConv2dModel(nn.Module):
    def __init__(
        self,
        image_shape: Tuple[int, int, int],
        output_size: int,
        convs: List[Tuple[int, int, int, int]] = [
            (32, 8, 4, 0),
            (64, 4, 2, 1),
            (64, 3, 1, 1),
        ],
        hidden_sizes: List[int] = [512],
        embedding_size: int = 64,
        layer_norm: bool = True,
    ):
        super().__init__()
        c, h, w = image_shape
        channels, kernel_sizes, strides, paddings = zip(*convs)
        self.conv = nn.Sequential(
            *[
                nn.Sequential(
                    nn.Conv2d(
                        c if i == 0 else channels[i - 1],
                        channels[i],
                        kernel_sizes[i],
                        strides[i],
                        paddings[i],
                    ),
                    nn.ReLU(),
                )
                for i in range(len(convs))
            ]
        ).apply(initialize_weights_he)
        
        state_embedding_size = conv_out_size(self.conv.modules(), h=h, w=w)
        self.tau_fc = nn.Sequential(
            nn.Linear(embedding_size, state_embedding_size),
            nn.LayerNorm(state_embedding_size) if layer_norm else nn.Identity(),
            nn.Sigmoid() if layer_norm else nn.ReLU(inplace=True),
        )
        self.merge_fc = nn.Sequential(
            *[
                nn.Sequential(
                    nn.Linear(
                        state_embedding_size if i == 0 else hidden_sizes[i - 1],
                        hidden_sizes[i],
                    ),
                    nn.LayerNorm(hidden_sizes[i]) if layer_norm else nn.Identity(),
                    nn.ReLU(inplace=True),
                )
                for i in range(len(hidden_sizes))
            ]
        )
        self.last_fc = nn.Linear(hidden_sizes[-1], output_size)
        self.register_buffer("const_vec", torch.arange(1, 1 + embedding_size))

    def forward(self, observation, tau, prev_action=None, prev_reward=None):

        h = self.conv(observation)
        # shape: (batch_size, state_embedding_size)
        h = torch.flatten(h, start_dim=1)

        # shape: (batch_size, num_quantiles, embedding_size)
        x = torch.cos(tau.unsqueeze(-1) * self.const_vec * np.pi)
        # shape: (batch_size, num_quantiles, state_embedding_size)
        x = self.tau_fc(x)

        # shape: (batch_size, num_quantiles, state_embedding_size)
        h = torch.mul(x, h.unsqueeze(-2))
        # shape: (batch_size, num_quantiles, hidden_size)
        h = self.merge_fc(h)
        # shape: (batch_size, num_quantiles, output_size)
        output = self.last_fc(h)
        return output
