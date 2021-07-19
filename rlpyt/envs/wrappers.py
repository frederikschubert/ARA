import numpy as np

import gym
from gym.spaces import Box


class PytorchImgWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.observation_space = Box(
            low=0,
            high=255,
            shape=[env.observation_space.shape[-1], *env.observation_space.shape[:-1],],
            dtype=np.uint8,
        )

    def observation(self, observation):
        return np.transpose(observation, axes=[2, 1, 0])

