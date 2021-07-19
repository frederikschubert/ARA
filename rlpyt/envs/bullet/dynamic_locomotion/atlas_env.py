from rlpyt.envs.bullet.dynamic_locomotion.dynamic_dynamics_base_env import (
    DynamicDynamicsBaseEnv,
)
from pybulletgym.envs.roboschool.robots.locomotors import Atlas


class AtlasBulletEnv(DynamicDynamicsBaseEnv):
    def __init__(self, *args, **kwargs):
        self.robot = Atlas()
        DynamicDynamicsBaseEnv.__init__(self, *args, **kwargs, robot=self.robot)

    def robot_specific_reset(self):
        self.robot.robot_specific_reset()
