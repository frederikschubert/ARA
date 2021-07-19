from rlpyt.envs.bullet.dynamic_locomotion.dynamic_dynamics_base_env import (
    DynamicDynamicsBaseEnv,
)
from pybulletgym.envs.roboschool.robots.locomotors import Humanoid


class HumanoidBulletEnv(DynamicDynamicsBaseEnv):
    def __init__(
        self, robot=None, *args, **kwargs,
    ):
        self.robot = robot or Humanoid()
        DynamicDynamicsBaseEnv.__init__(self, *args, **kwargs, robot=self.robot)
        self.electricity_cost = 4.25 * DynamicDynamicsBaseEnv.electricity_cost
        self.stall_torque_cost = 4.25 * DynamicDynamicsBaseEnv.stall_torque_cost

