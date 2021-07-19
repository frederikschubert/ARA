from rlpyt.algos.dqn.iqn import IQN
from rlpyt.algos.rnd.rnd import RND
from rlpyt.utils.buffer import buffer_to
from rlpyt.utils.logging import logger


class RNDIQN(RND, IQN):
    def get_tau(self, samples, description: str):
        next_states, actions = buffer_to(
            (samples.target_inputs.observation, samples.action),
            device=self.agent.device,
        )
        if description in self.risk_transform_fields:
            tau, param = self.agent.get_tau(next_states, actions, transform=True)
            logger.record_histogram("RND/param", param.detach().cpu().numpy())
        else:
            tau = self.agent.get_tau(next_states, actions, transform=False)
        return tau
