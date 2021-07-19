import torch
from rlpyt.agents.base import AgentStep
from rlpyt.agents.dqn.atari.mixin import AtariMixin
from rlpyt.agents.dqn.iqn_agent import (
    IqnAgent,
    IqnAgentInfo,
    kolmogorov_smirnov_distance,
)
from rlpyt.agents.rnd.rnd_mixin import RndAgentMixin
from rlpyt.algos.risk import distortion_fn
from rlpyt.models.dqn.atari_iqn_model import AtariIqnModel
from rlpyt.utils.buffer import buffer_to
from rlpyt.utils.collections import namedarraytuple

RndIqnAgentInfo = namedarraytuple(
    "RndIqnAgentInfo", IqnAgentInfo._fields + ("rnd_param",)
)


class RndIqnAgent(RndAgentMixin, IqnAgent):
    @torch.no_grad()
    def step(self, observation, prev_action, prev_reward):
        obs, prev_a = buffer_to((observation, prev_action), self.device)
        tau = self.get_tau(obs, prev_a, transform=self.risk_transform_policy)
        if self.risk_transform_policy:
            tau, param = tau
        else:
            param = torch.ones_like(prev_a)
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
        agent_info = RndIqnAgentInfo(
            q=q.numpy(),
            rnd_param=param.cpu().numpy(),
            quantiles=quantile_values.cpu().numpy(),
        )
        return AgentStep(action=action, agent_info=agent_info)

    def get_tau(
        self, observation: torch.Tensor, prev_action: torch.Tensor, transform: bool
    ):
        tau_fractions = torch.rand(
            len(prev_action) if prev_action.dim() > 0 else 1,
            self.num_quantiles,
            device=self.device,
        )
        tau_fractions /= tau_fractions.sum(dim=-1, keepdim=True)
        tau = torch.cumsum(tau_fractions, dim=1)
        if transform:
            param = self.compute_risk_param(observation, prev_action)
            tau_transformed = distortion_fn(tau, mode=self.risk_mode, param=param)
            return tau_transformed, param
        else:
            return tau


class AtariRndIqnAgent(AtariMixin, RndIqnAgent):
    def __init__(self, ModelCls=AtariIqnModel, **kwargs):
        super().__init__(ModelCls=ModelCls, **kwargs)
