from collections import namedtuple
from rlpyt.utils.logging import logger


from rlpyt.utils.collections import namedarraytuple, AttrDict


Samples = namedarraytuple("Samples", ["agent", "env"])

AgentSamples = namedarraytuple("AgentSamples", ["action", "prev_action", "agent_info"])
AgentSamplesBsv = namedarraytuple(
    "AgentSamplesBsv", ["action", "prev_action", "agent_info", "bootstrap_value"]
)
EnvSamples = namedarraytuple(
    "EnvSamples", ["observation", "reward", "prev_reward", "done", "env_info"]
)


class BatchSpec(namedtuple("BatchSpec", "T B")):
    """
    T: int  Number of time steps, >=1.
    B: int  Number of separate trajectory segments (i.e. # env instances), >=1.
    """

    __slots__ = ()

    @property
    def size(self):
        return self.T * self.B


class TrajInfo(AttrDict):
    """
    Because it inits as an AttrDict, this has the methods of a dictionary,
    e.g. the attributes can be iterated through by traj_info.items()
    Intent: all attributes not starting with underscore "_" will be logged.
    (Can subclass for more fields.)
    Convention: traj_info fields CamelCase, opt_info fields lowerCamelCase.
    """

    _discount = 1  # Leading underscore, but also class attr not in self.__dict__.
    _record_trajectory = 0

    def __init__(self, **kwargs):
        super().__init__(**kwargs)  # (for AttrDict behavior)
        self.Length = 0
        self.Return = 0
        self.NonzeroRewards = 0
        self.DiscountedReturn = 0
        self.Observations = list()
        self._cur_discount = 1
        self._recording = False

    def append_obs(self, observation):
        if len(observation.shape) == 3 and (
            observation.shape[0] == 3 or observation.shape[-1] == 3
        ):
            if not self.Observations:
                with self._record_trajectory.get_lock():
                    if self._record_trajectory.value > 0:
                        logger.log("Recording trajectory")
                        self._recording = True
                        self._record_trajectory.value -= 1
            if self._recording:
                self.Observations.append(observation.copy())

    def step(self, observation, action, reward, done, agent_info, env_info):
        self.Length += 1
        self.Return += reward
        self.NonzeroRewards += reward != 0
        self.DiscountedReturn += self._cur_discount * reward
        self._cur_discount *= self._discount
        self.append_obs(observation)

    def terminate(self, observation):
        self.append_obs(observation)
        return self
