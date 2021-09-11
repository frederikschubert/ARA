import torch
from rlpyt.agents.qpg.dsac_agent import DsacAgent
from rlpyt.agents.rnd.rnd_mixin import RndAgentMixin
from rlpyt.algos.risk import distortion_de, distortion_fn


class RndDsacAgent(RndAgentMixin, DsacAgent):
    @property
    def rnd_input_shape(self):
        return (
            self.env_model_kwargs["image_shape"]
            if "image_shape" in self.env_model_kwargs
            else [self.env_model_kwargs["observation_shape"][0]]
        )

    def get_tau(
        self, observation: torch.Tensor, prev_action: torch.Tensor, transform: bool
    ):
        presum_tau = (
            torch.rand(len(prev_action), self.num_quantiles, device=self.device) + 0.1
        )
        presum_tau /= presum_tau.sum(dim=-1, keepdim=True)
        tau = torch.cumsum(presum_tau, dim=1)
        with torch.no_grad():
            tau_hat = torch.zeros_like(tau)
            tau_hat[:, 0:1] = tau[:, 0:1] / 2.0
            tau_hat[:, 1:] = (tau[:, 1:] + tau[:, :-1]) / 2.0
            if transform:
                param = self.compute_risk_param(observation, prev_action=prev_action)
                risk_weights = distortion_de(
                    tau_hat, mode=self.risk_mode, param=param.unsqueeze(-1)
                )
                presum_tau = risk_weights * presum_tau
                return tau, tau_hat, presum_tau, param
            else:
                return tau, tau_hat, presum_tau
