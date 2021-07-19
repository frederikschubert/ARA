import torch
import torch.nn as nn
import torch.nn.functional as F
from rlpyt.models.quantile_conv2d_model import QuantileConv2dModel
from rlpyt.utils.tensor import infer_leading_dims, restore_leading_dims


class AtariIqnModel(QuantileConv2dModel):
    def forward(self, observation, tau, prev_action=None, prev_reward=None):
        img = observation.type(torch.float)  # Expect torch.uint8 inputs
        img = img / 255.0  # From [0-255] to [0-1], in place.

        # Infer (presence of) leading dimensions: [T,B], [B], or [].
        lead_dim, T, B, img_shape = infer_leading_dims(img, 3)

        quantile_values = super().forward(
            img.view(T * B, *img_shape),
            tau=tau,
            prev_action=prev_action,
            prev_reward=prev_reward,
        )

        quantile_values = restore_leading_dims(quantile_values, lead_dim, T, B)

        return quantile_values
