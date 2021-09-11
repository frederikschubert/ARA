from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from rlpyt.models.rnd.predictor import RndPredictor
from rlpyt.models.rnd.target import RndTarget
from rlpyt.models.running_mean_std import RunningMeanStdModel
from rlpyt.utils.tensor import infer_leading_dims, restore_leading_dims


class RndModel(nn.Module):
    def __init__(
        self,
        input_shape,
        output_size=None,
        obs_clamp_value: float = 100.0,
        min_size: int = 60,
    ):
        super().__init__()
        self.min_size = min_size
        self.resize_input = False
        self.expand_input = False
        if len(input_shape) == 1:
            input_size = input_shape[0]
            self.rnd_obs_mean_std = RunningMeanStdModel(2 * input_size)
            if input_size < min_size:
                self.resize_input = True
            self.register_buffer("const_vec", torch.arange(1, 1 + input_size))
            self.input_shape = (1, input_size, input_size)
            self.expand_input = True
        elif len(input_shape) == 3:
            self.rnd_obs_mean_std = RunningMeanStdModel([2 * input_shape[0], *input_shape[1:]])
            self.input_shape = input_shape
        else:
            raise ValueError("Only 1 or 3 dimensional observations supported")
        self.output_size = output_size
        self.obs_clamp_value = obs_clamp_value
        self.rnd_predictor = RndPredictor(
            self.input_shape if not self.resize_input else (1, min_size, min_size),
            output_size,
        )
        self.rnd_target = RndTarget(self.input_shape if not self.resize_input else (1, min_size, min_size), output_size)
        self.rnd_error_mean_std = RunningMeanStdModel([])

    def forward(
        self, observation: torch.Tensor, prev_action: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        lead_dim, T, B, obs_shape = infer_leading_dims(
            observation, len(self.input_shape) if not self.expand_input else 1
        )
        with torch.no_grad():
            observation = observation.float().view(T * B, *obs_shape)
            observation_normalized = torch.clamp(
                (observation - self.rnd_obs_mean_std.mean)
                / (torch.sqrt(self.rnd_obs_mean_std.var) + 1e-10),
                -self.obs_clamp_value,
                self.obs_clamp_value,
            )
            if self.expand_input:
                observation_normalized = torch.cos(
                    observation_normalized.unsqueeze(-1) * self.const_vec * np.pi
                ).unsqueeze(-3)
            if self.resize_input:
                observation_normalized = F.interpolate(
                    observation_normalized, (self.min_size, self.min_size)
                )
            self.rnd_obs_mean_std.update(observation)

            targets = self.rnd_target(observation_normalized, action=prev_action)

        predictions = self.rnd_predictor(observation_normalized, action=prev_action)
        rnd_error = torch.pow(predictions - targets, 2).mean(dim=-1)
        return restore_leading_dims(rnd_error, lead_dim, T, B)
