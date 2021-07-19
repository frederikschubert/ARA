import multiprocessing as mp
from rlpyt.replays.non_sequence.prioritized import (
    AsyncPrioritizedReplayBuffer,
    PrioritizedReplayBuffer,
)
from typing import Callable, Dict, Optional

import rlpyt.agents
import rlpyt.algos
import torch
import yaml
from rlpyt.agents.base import BaseAgent
from rlpyt.algos.base import RlAlgorithm
from rlpyt.experiment_utils.affinity import get_affinity_sampler_runner
from rlpyt.experiment_utils.args import ExperimentArgs
from rlpyt.experiment_utils.utils import import_submodules
from rlpyt.replays.non_sequence.uniform import (
    AsyncUniformReplayBuffer,
    UniformReplayBuffer,
)
from rlpyt.runners.base import BaseRunner
from rlpyt.samplers.base import BaseSampler
from rlpyt.samplers.collections import TrajInfo
from rlpyt.utils.logging import logger
from rlpyt.utils.logging.context import logger_context
from xvfbwrapper import Xvfb


def train(
    args: ExperimentArgs,
    EnvCls: Callable,
    env_kwargs: Dict,
    eval_env_kwargs: Dict,
    agent_kwargs: Dict = dict(),
    algo_kwargs: Dict = dict(),
    TrajInfoCls=TrajInfo,
    init_kwargs: Optional[Dict] = None,
):
    import_submodules(rlpyt.agents)
    import_submodules(rlpyt.algos)
    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    affinity, SamplerCls, RunnerCls = get_affinity_sampler_runner(args)
    if args.restore:
        logger.log(f"Restoring agent from {args.restore}")
        with open(args.restore, "rb") as f:
            state_dicts = torch.load(f)
        agent_kwargs["initial_model_state_dict"] = state_dicts["agent_state_dict"]
        algo_kwargs["initial_optim_state_dict"] = state_dicts["optimizer_state_dict"]
    agent: BaseAgent = rlpyt.agents.agent_registry[config["agent"]](
        **{**config["agent_config"], **agent_kwargs}
    )
    algo: RlAlgorithm = rlpyt.algos.algo_registry[config["algo"]](
        **{**config["algo_config"], **algo_kwargs}
    )
    if hasattr(algo, "ReplayBufferCls"):
        if not args.frame_stack and not hasattr(algo, "bootstrap_timelimit"):
            if getattr(algo, "prioritized_replay", False):
                setattr(
                    algo,
                    "ReplayBufferCls",
                    PrioritizedReplayBuffer
                    if args.sync_sample
                    else AsyncPrioritizedReplayBuffer,
                )
            else:
                setattr(
                    algo,
                    "ReplayBufferCls",
                    UniformReplayBuffer
                    if args.sync_sample
                    else AsyncUniformReplayBuffer,
                )

    sampler: BaseSampler = SamplerCls(
        EnvCls=EnvCls,
        env_kwargs=env_kwargs,
        eval_env_kwargs=eval_env_kwargs,
        TrajInfoCls=TrajInfoCls,
    )
    record_trajectory_flag = mp.Value("i", 1)
    runner: BaseRunner = RunnerCls(
        algo=algo, agent=agent, sampler=sampler, n_steps=args.steps, affinity=affinity,
    )
    setattr(sampler.TrajInfoCls, "_record_trajectory", record_trajectory_flag)
    logger.set_record_trajectory_flag(record_trajectory_flag)
    with Xvfb():
        with logger_context(
            args.name,
            dict(
                **args.as_dict(),
                **config,
                env_kwargs=env_kwargs,
                eval_env_kwargs=eval_env_kwargs,
            ),
            snapshot_mode="last",
            debug=args.debug,
            init_kwargs=init_kwargs,
        ):
            runner.train()
