from rlpyt.utils.logging import logger
from rlpyt.envs.bullet.frozen_joint import FrozenJointEnv
from rlpyt.envs.gym import GymEnvWrapper

import gym
import pybullet_envs


gym.register(
    id="DynamicWalker2DBulletEnv-v0",
    entry_point="rlpyt.envs.bullet.dynamic_locomotion:Walker2DBulletEnv",
    max_episode_steps=1000,
    reward_threshold=2500.0,
)
gym.register(
    id="DynamicHalfCheetahBulletEnv-v0",
    entry_point="rlpyt.envs.bullet.dynamic_locomotion:HalfCheetahBulletEnv",
    max_episode_steps=1000,
    reward_threshold=3000.0,
)

gym.register(
    id="DynamicAntBulletEnv-v0",
    entry_point="rlpyt.envs.bullet.dynamic_locomotion:AntBulletEnv",
    max_episode_steps=1000,
    reward_threshold=2500.0,
)

gym.register(
    id="DynamicHopperBulletEnv-v0",
    entry_point="rlpyt.envs.bullet.dynamic_locomotion:HopperBulletEnv",
    max_episode_steps=1000,
    reward_threshold=2500.0,
)

gym.register(
    id="DynamicHumanoidBulletEnv-v0",
    entry_point="rlpyt.envs.bullet.dynamic_locomotion:HumanoidBulletEnv",
    max_episode_steps=1000,
)

gym.register(
    id="DynamicHumanoidFlagrunBulletEnv-v0",
    entry_point="rlpyt.envs.bullet.dynamic_locomotion:HumanoidFlagrunBulletEnv",
    max_episode_steps=1000,
    reward_threshold=2000.0,
)

gym.register(
    id="DynamicHumanoidFlagrunHarderBulletEnv-v0",
    entry_point="rlpyt.envs.bullet.dynamic_locomotion:HumanoidFlagrunHarderBulletEnv",
    max_episode_steps=1000,
)


def make_bullet(
    walker_id: str, frozen_joint_kwargs=dict(), dynamics_kwargs=dict(), render=False
):
    if walker_id.startswith("Dynamic"):
        env = gym.make(walker_id, changeDynamics_kwargs=dynamics_kwargs, render=render)
    else:
        env = gym.make(walker_id)

    if frozen_joint_kwargs:
        env = FrozenJointEnv(env, **frozen_joint_kwargs)
    return GymEnvWrapper(env)
