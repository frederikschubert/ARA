from typing import Literal

import torch
from rlpyt.agents.qpg.sac_agent import SacAgent
from rlpyt.algos.risk import distortion_de, distortion_fn
from rlpyt.models.quantile_mlp_model import QuantileMlpModel
from rlpyt.utils.buffer import buffer_to


class DsacAgent(SacAgent):
    def __init__(
        self,
        num_quantiles=32,
        QModelCls=QuantileMlpModel,
        risk_mode: Literal["neutral", "wang", "cvar", "cpw"] = "neutral",
        risk_param: float = 0.0,
        *args,
        **kwargs
    ):
        super().__init__(QModelCls=QModelCls, *args, **kwargs)
        self.num_quantiles = num_quantiles
        self.risk_mode = risk_mode
        self.risk_param = torch.tensor([risk_param])

    def q(self, observation, prev_action, prev_reward, action, tau):
        model_inputs = buffer_to(
            (observation, prev_action, prev_reward, action, tau), device=self.device
        )
        q1 = self.q1_model(*model_inputs)
        q2 = self.q2_model(*model_inputs)
        return q1, q2

    def target_q(self, observation, prev_action, prev_reward, action, tau):
        model_inputs = buffer_to(
            (observation, prev_action, prev_reward, action, tau), device=self.device
        )
        target_q1 = self.target_q1_model(*model_inputs)
        target_q2 = self.target_q2_model(*model_inputs)
        return target_q1, target_q2

    def get_tau(self, n: int, transform: bool = False):
        presum_tau = torch.rand(n, self.num_quantiles, device=self.device) + 0.1
        presum_tau /= presum_tau.sum(dim=-1, keepdim=True)
        tau = torch.cumsum(
            presum_tau, dim=1
        )  # (N, T), note that they are tau1...tauN in the paper
        with torch.no_grad():
            tau_hat = torch.zeros_like(tau)
            tau_hat[:, 0:1] = tau[:, 0:1] / 2.0
            tau_hat[:, 1:] = (tau[:, 1:] + tau[:, :-1]) / 2.0
            if transform:
                risk_weights = distortion_de(
                    tau_hat, mode=self.risk_mode, param=self.risk_param.to(self.device)
                )
                presum_tau = risk_weights * presum_tau
        return tau, tau_hat, presum_tau
