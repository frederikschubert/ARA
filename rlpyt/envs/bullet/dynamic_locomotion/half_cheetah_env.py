from rlpyt.envs.bullet.dynamic_locomotion.dynamic_dynamics_base_env import (
    DynamicDynamicsBaseEnv,
)
from pybulletgym.envs.roboschool.robots.locomotors import HalfCheetah


class HalfCheetahBulletEnv(DynamicDynamicsBaseEnv):
    def __init__(self, *args, **kwargs):
        self.robot = HalfCheetah()
        DynamicDynamicsBaseEnv.__init__(self, *args, **kwargs, robot=self.robot)
