import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from rlpyt.models.utils import conv_out_size


class RndTarget(nn.Module):
    def __init__(self, input_shape, output_size):
        super().__init__()
        self.input_shape = input_shape
        self.output_size = output_size
        c, h, w = input_shape
        self.conv1 = nn.Conv2d(c, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.fc1 = nn.Linear(
            conv_out_size([self.conv1, self.conv2, self.conv3], h, w), 512
        )
        if output_size:
            self.fc_action = nn.Sequential(
                nn.Linear(output_size, 256), nn.LeakyReLU(), nn.Linear(256, 512)
            )

        for p in self.modules():
            if isinstance(p, nn.Conv2d):
                init.orthogonal_(p.weight, np.sqrt(2))
                p.bias.data.zero_()

            if isinstance(p, nn.Linear):
                init.orthogonal_(p.weight, np.sqrt(2))
                p.bias.data.zero_()

        for param in self.parameters():
            param.requires_grad = False

    def forward(self, observation, action=None):
        x = F.leaky_relu(self.conv1(observation))
        x = F.leaky_relu(self.conv2(x))
        x = F.leaky_relu(self.conv3(x))
        x = torch.flatten(x, start_dim=1)
        target_value = self.fc1(x)
        if action is not None:
            if action.dim() == 0 or action.shape[-1] != self.output_size:
                action = F.one_hot(action.long(), self.output_size).float()
            action = self.fc_action(action)
            target_value = target_value + action
        return target_value
