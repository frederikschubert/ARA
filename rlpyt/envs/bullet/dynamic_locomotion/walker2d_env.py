from rlpyt.envs.bullet.dynamic_locomotion.dynamic_dynamics_base_env import (
    DynamicDynamicsBaseEnv,
)
from pybulletgym.envs.roboschool.robots.locomotors import Walker2D


class Walker2DBulletEnv(DynamicDynamicsBaseEnv):
    def __init__(self, *args, **kwargs):
        self.robot = Walker2D()
        DynamicDynamicsBaseEnv.__init__(self, *args, **kwargs, robot=self.robot)

