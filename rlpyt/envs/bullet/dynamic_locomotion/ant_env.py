from pybulletgym.envs.roboschool.robots.locomotors import Ant
from rlpyt.envs.bullet.dynamic_locomotion.dynamic_dynamics_base_env import (
    DynamicDynamicsBaseEnv,
)


class AntBulletEnv(DynamicDynamicsBaseEnv):
    def __init__(self, *args, **kwargs):
        self.robot = Ant()
        DynamicDynamicsBaseEnv.__init__(self, *args, **kwargs, robot=self.robot)
