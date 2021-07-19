from rlpyt.utils.tensor import infer_leading_dims, restore_leading_dims
import numpy as np
import torch
from torch import nn as nn
from torch.nn import functional as F


def weight_init(m):
    """Custom weight init for Conv2D and Linear layers."""
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data)
        m.bias.data.fill_(0.0)
    elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        # delta-orthogonal init from https://arxiv.org/pdf/1806.05393.pdf
        assert m.weight.size(2) == m.weight.size(3)
        m.weight.data.fill_(0.0)
        m.bias.data.fill_(0.0)
        mid = m.weight.size(2) // 2
        gain = nn.init.calculate_gain("relu")
        nn.init.orthogonal_(m.weight.data[:, :, mid, mid], gain)


class QuantileMlpModel(nn.Module):
    def __init__(
        self,
        hidden_sizes,
        action_size,
        observation_shape,
        embedding_size=64,
        num_quantiles=32,
        layer_norm=True,
        **kwargs,
    ):
        super().__init__()
        self.layer_norm = layer_norm
        # hidden_sizes[:-2] MLP base
        # hidden_sizes[-2] before merge
        # hidden_sizes[-1] before output

        self.base_fc_layers = []
        last_size = np.prod(observation_shape) + action_size
        for next_size in hidden_sizes[:-1]:
            self.base_fc_layers += [
                nn.Linear(last_size, next_size),
                nn.LayerNorm(next_size) if layer_norm else nn.Identity(),
                nn.ReLU(inplace=True),
            ]
            last_size = next_size
        self.base_fc = nn.Sequential(*self.base_fc_layers)
        self.num_quantiles = num_quantiles
        self.embedding_size = embedding_size
        self.tau_fc = nn.Sequential(
            nn.Linear(embedding_size, last_size),
            nn.LayerNorm(last_size) if layer_norm else nn.Identity(),
            nn.Sigmoid(),
        )
        self.merge_fc = nn.Sequential(
            nn.Linear(last_size, hidden_sizes[-1]),
            nn.LayerNorm(hidden_sizes[-1]) if layer_norm else nn.Identity(),
            nn.ReLU(inplace=True),
        )
        self.last_fc = nn.Linear(hidden_sizes[-1], 1)
        self.register_buffer("const_vec", torch.arange(1, 1 + embedding_size))
        self._obs_ndim = len(observation_shape)
        self.apply(weight_init)

    def forward(self, observation, prev_action, prev_reward, action, tau):
        lead_dim, T, B, _ = infer_leading_dims(observation, self._obs_ndim)
        q_input = torch.cat(
            [observation.view(T * B, -1), action.view(T * B, -1)], dim=1
        )
        h = self.base_fc(q_input)  # (N, C)

        x = torch.cos(tau.unsqueeze(-1) * self.const_vec * np.pi)  # (N, T, E)
        x = self.tau_fc(x)  # (N, T, C)

        h = torch.mul(x, h.unsqueeze(-2))  # (N, T, C)
        h = self.merge_fc(h)  # (N, T, C)
        output = self.last_fc(h).squeeze(-1)  # (N, T)
        output = restore_leading_dims(output, lead_dim, T, B)
        return output
