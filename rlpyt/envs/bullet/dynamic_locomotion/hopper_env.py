from rlpyt.envs.bullet.dynamic_locomotion.dynamic_dynamics_base_env import (
    DynamicDynamicsBaseEnv,
)
from pybulletgym.envs.roboschool.robots.locomotors import Hopper


class HopperBulletEnv(DynamicDynamicsBaseEnv):
    def __init__(self, *args, **kwargs):
        self.robot = Hopper()
        DynamicDynamicsBaseEnv.__init__(self, *args, **kwargs, robot=self.robot)

