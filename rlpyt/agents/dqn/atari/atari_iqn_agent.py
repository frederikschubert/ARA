from rlpyt.agents.dqn.iqn_agent import IqnAgent
from rlpyt.models.dqn.atari_iqn_model import AtariIqnModel
from rlpyt.agents.dqn.atari.mixin import AtariMixin


class AtariIqnAgent(AtariMixin, IqnAgent):
    def __init__(self, ModelCls=AtariIqnModel, **kwargs):
        super().__init__(ModelCls=ModelCls, **kwargs)
