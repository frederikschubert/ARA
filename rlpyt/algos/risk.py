from typing import Literal

import numpy as np
import torch


def normal_cdf(value: torch.Tensor, loc=0.0, scale=1.0):
    return 0.5 * (1 + torch.erf((value - loc) / scale / np.sqrt(2)))


def normal_icdf(value: torch.Tensor, loc=0.0, scale=1.0):
    return loc + scale * torch.erfinv(2 * value - 1) * np.sqrt(2)


def normal_pdf(value: torch.Tensor, loc=0.0, scale=1.0):
    return (
        torch.exp(-((value - loc) ** 2) / (2 * scale ** 2)) / scale / np.sqrt(2 * np.pi)
    )


def distortion_fn(tau: torch.Tensor, mode="neutral", param=torch.Tensor([0.0])):
    # Risk distortion function
    tau = tau.clamp(0.0, 1.0)
    if tau.dim() > param.dim():
        param = param.reshape(-1, *[1 for _ in range(tau.dim() - 1)])
    if mode == "neutral":
        tau_ = tau
    elif mode == "wang":
        tau_ = normal_cdf(normal_icdf(tau) + param)
    elif mode == "cvar":
        tau_ = param * tau
    elif mode == "cpw":
        tau_ = torch.pow(tau, param) / torch.pow(
            torch.pow(tau, param) + torch.pow((1 - tau), param), 1 / param
        )
    return tau_.clamp(0.0, 1.0)


def distortion_de(tau, mode="neutral", param=0.0, eps=1e-8):
    # Derivative of Risk distortion function
    tau = tau.clamp(0.0, 1.0)
    if mode == "neutral":
        tau_ = torch.ones_like(tau)
    elif mode == "wang":
        tau_ = normal_pdf(normal_icdf(tau) + param) / (
            normal_pdf(normal_icdf(tau)) + eps
        )
    elif mode == "cvar":
        tau_ = (1.0 / param) * (tau < param)
    elif mode == "cpw":
        g = tau ** param
        h = (tau ** param + (1 - tau) ** param) ** (1 / param)
        g_ = param * tau ** (param - 1)
        h_ = (tau ** param + (1 - tau) ** param) ** (1 / param - 1) * (
            tau ** (param - 1) - (1 - tau) ** (param - 1)
        )
        tau_ = (g_ * h - g * h_) / (h ** 2 + eps)
    return tau_.clamp(0.0, 5.0)
