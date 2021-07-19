import os
from rlpyt.samplers.collections import TrajInfo
from typing import List, Tuple
from rlpyt.utils.logging import logger
import multiprocessing as mp
from tap import Tap

from rlpyt.envs.bullet.gym import make_bullet
from rlpyt.experiment_utils.args import ExperimentArgs
from rlpyt.experiment_utils.train import train

# See https://github.com/lerrel/rllab-adv/blob/master/adversarial/scripts/test_robustness_friction.py


class BulletTrajInfo(TrajInfo):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.FailureRate = 0.0

    def step(self, observation, action, reward, done, agent_info, env_info):
        if done and not env_info.timeout:
            self.FailureRate = 1.0
        return super().step(observation, action, reward, done, agent_info, env_info)


class BulletArgs(Tap):
    walker_id: str = "DynamicWalker2DBulletEnv-v0"
    max_broken_axes: int = 1
    break_prob: float = 0.0

    slippery: bool = False
    slippery_limits: Tuple[float, float] = (0.1, 10.0)
    slippery_steps: List[float] = []
    bouncy: bool = False
    bouncy_limits: Tuple[float, float] = (0.1, 10.0)
    bouncy_steps: List[float] = []
    heavy: bool = False
    heavy_limits: Tuple[float, float] = (0.1, 10.0)
    heavy_steps: List[float] = []

    def process_args(self) -> None:
        super().process_args()
        self.frozen_joint_kwargs = dict()
        self.dynamics_kwargs = dict()
        if self.max_broken_axes > 0 and self.break_prob > 0:
            self.frozen_joint_kwargs = dict(
                max_broken_axes=self.max_broken_axes, break_prob=self.break_prob
            )
        if self.slippery:
            if not self.slippery_steps:
                self.dynamics_kwargs["lateralFriction"] = self.dynamics_kwargs[
                    "rollingFriction"
                ] = self.dynamics_kwargs["spinningFriction"] = {
                    "type": "range",
                    "limits": self.slippery_limits,
                }
            else:
                self.dynamics_kwargs["lateralFriction"] = self.dynamics_kwargs[
                    "rollingFriction"
                ] = self.dynamics_kwargs["spinningFriction"] = {
                    "type": "fixed",
                    "value": mp.Value("d", self.slippery_steps[0]),
                }
        if self.bouncy:
            if not self.bouncy_steps:
                self.dynamics_kwargs["restitution"] = {
                    "type": "range",
                    "limits": self.bouncy_limits,
                }
            else:
                self.dynamics_kwargs["restitution"] = {
                    "type": "fixed",
                    "value": mp.Value("d", self.bouncy_steps[0]),
                }
        if self.heavy:
            if not self.heavy_steps:
                self.dynamics_kwargs["mass"] = {
                    "type": "range",
                    "limits": self.heavy_limits,
                }
            else:
                self.dynamics_kwargs["mass"] = {
                    "type": "fixed",
                    "value": mp.Value("d", self.heavy_steps[0]),
                }


class TrainBulletArgs(ExperimentArgs, BulletArgs):
    def process_args(self) -> None:
        super().process_args()
        self.batch_T = 1
        self.log_interval_steps = int(1e4)
        self.sync_sample = True
        self.steps = self.steps or int(1e6)
        self.env_type = "mujoco"
        self.n_envs = 1
        self.eval_n_envs = 4
        self.eval_max_steps = int(51e3)
        self.eval_max_trajectories = 50
        algorithm_name = os.path.basename(self.config).split(".")[0]
        if self.frozen_joint_kwargs:
            self.name = f"{algorithm_name}-{self.walker_id}-max_broken:{self.max_broken_axes}-break_prob:{self.break_prob}"
        else:
            self.name = f"{algorithm_name}-{self.walker_id}"
        if self.slippery:
            self.name += "-slippery"
            if self.slippery_steps:
                self.name += "-" + ",".join([str(s) for s in self.slippery_steps])
        if self.bouncy:
            self.name += "-bouncy"
            if self.bouncy_steps:
                self.name += "-" + ",".join([str(s) for s in self.bouncy_steps])
        if self.heavy:
            self.name += "-heavy"
            if self.heavy_steps:
                self.name += "-" + ",".join([str(s) for s in self.heavy_steps])


def main():
    args = TrainBulletArgs().parse_args()
    env_kwargs = dict(
        walker_id=args.walker_id,
        frozen_joint_kwargs=args.frozen_joint_kwargs,
        dynamics_kwargs=args.dynamics_kwargs,
    )
    for i, friction in enumerate(args.slippery_steps):
        step = args.steps // len(args.slippery_steps) * i

        def set_friction(friction=friction):
            logger.log(f"Setting friction to {friction}")
            args.dynamics_kwargs["lateralFriction"]["value"].value = friction
            args.dynamics_kwargs["rollingFriction"]["value"].value = friction
            args.dynamics_kwargs["spinningFriction"]["value"].value = friction

        logger.register_callback(step, set_friction)
    for i, restitution in enumerate(args.bouncy_steps):
        step = args.steps // len(args.bouncy_steps) * i

        def set_restitution(restitution=restitution):
            logger.log(f"Setting restitution to {restitution}")
            args.dynamics_kwargs["restitution"]["value"].value = restitution

        logger.register_callback(step, set_restitution)
    for i, mass in enumerate(args.heavy_steps):
        step = args.steps // len(args.heavy_steps) * i

        def set_mass(mass=mass):
            logger.log(f"Setting mass to {mass}")
            args.dynamics_kwargs["mass"]["value"].value = mass

        logger.register_callback(step, set_mass)
    if args.walker_id in ["BipedalWalker-v3", "BipedalWalkerHardcore-v3"]:
        algo_kwargs = dict(fixed_alpha=0.002)
    else:
        algo_kwargs = dict()
    train(
        args,
        make_bullet,
        env_kwargs=env_kwargs,
        eval_env_kwargs=env_kwargs,
        init_kwargs=dict(entity="tnt", project="rnd-iqn", group=args.group),
        algo_kwargs=algo_kwargs,
        TrajInfoCls=BulletTrajInfo
    )


if __name__ == "__main__":
    main()
