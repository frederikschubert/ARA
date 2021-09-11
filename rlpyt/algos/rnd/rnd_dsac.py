import torch
from rlpyt.algos.qpg.dsac import DSAC
from rlpyt.algos.rnd.rnd import RND
from rlpyt.utils.buffer import buffer_to
from rlpyt.utils.logging import logger


class RNDDSAC(RND, DSAC):

    def get_tau(self, samples, description: str):
        states, next_states, actions = buffer_to(
            (
                samples.agent_inputs.observation,
                samples.target_inputs.observation,
                samples.action,
            ),
            device=self.agent.device,
        )
        if description in self.risk_transform_fields:
            tau, tau_hat, tau_fractions, param = self.agent.get_tau(
                torch.cat([states, states - next_states], dim=-1),
                actions,
                transform=True,
            )
            logger.record_histogram("RND/param", param.detach().cpu().numpy())
            return tau, tau_hat, tau_fractions
        else:
            return self.agent.get_tau(next_states, actions, transform=False)
