from typing import Literal

import torch
from rlpyt.agents.base import AgentStep
from rlpyt.agents.dqn.dqn_agent import AgentInfo, DqnAgent
from rlpyt.algos.risk import distortion_fn
from rlpyt.utils.buffer import buffer_to
from rlpyt.utils.collections import namedarraytuple

IqnAgentInfo = namedarraytuple("IqnAgentInfo", AgentInfo._fields + ("quantiles",))


def kolmogorov_smirnov_distance(quantile_samples: torch.Tensor):
    # shape: (num_quantiles, num_actions)
    quantile_samples_sorted = quantile_samples.argsort(dim=0)
    ecdf = quantile_samples_sorted.float() / quantile_samples_sorted.max().float()
    # shape: (num_quantiles, num_actions, num_actions)
    differences = ecdf.unsqueeze(-1) - ecdf.unsqueeze(-2)
    # shape: (num_actions, num_actions)
    max_differences = differences.max(dim=0)[0]
    return max_differences.min()


class IqnAgent(DqnAgent):
    def __init__(
        self,
        num_quantiles=32,
        risk_mode: Literal["neutral", "wang", "cvar", "cpw"] = "neutral",
        risk_param: float = 0.0,
        risk_transform_policy: bool = False,
        adaptive_epsilon: bool = False,
        *args,
        **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)
        self.num_quantiles = num_quantiles
        self.risk_mode = risk_mode
        self.risk_param = torch.tensor([risk_param])
        self.risk_transform_policy = risk_transform_policy
        self.adaptive_epsilon = adaptive_epsilon

    @torch.no_grad()
    def step(self, observation, prev_action, prev_reward):
        obs = buffer_to(observation, self.device)
        tau = self.get_tau(transform=self.risk_transform_policy)
        quantile_values = self.model(obs, tau=tau)
        q = quantile_values.mean(dim=-2)
        q = q.cpu()
        if self.adaptive_epsilon:
            ks_distance = kolmogorov_smirnov_distance(quantile_values)
            base_epsilon = self.distribution.epsilon
            adaptive_epsilon = (1 - ks_distance) * base_epsilon
            self.distribution.set_epsilon(adaptive_epsilon.cpu())
        action = self.distribution.sample(q)
        if self.adaptive_epsilon:
            self.distribution.set_epsilon(base_epsilon)
        agent_info = IqnAgentInfo(q=q, quantiles=quantile_values.cpu().numpy())
        return AgentStep(action=action, agent_info=agent_info)

    def get_tau(self, n: int = 1, transform=False):
        tau_fractions = torch.rand(n, self.num_quantiles, device=self.device)
        tau_fractions /= tau_fractions.sum(dim=-1, keepdim=True)
        tau = torch.cumsum(tau_fractions, dim=1)
        if transform:
            tau = distortion_fn(
                tau, mode=self.risk_mode, param=self.risk_param.to(self.device)
            )
        return tau

    def load_state_dict(self, state_dict):
        return super().load_state_dict(state_dict["model"])
