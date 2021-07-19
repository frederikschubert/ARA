import os

import wandb
from rlpyt.utils.logging.context import LOG_DIR, logger_context
from rlpyt.models.utils import strip_ddp_state_dict
from rlpyt.samplers.serial.collectors import SerialEvalCollector
from typing import Callable, Dict, List, Optional

import rlpyt.agents
import rlpyt.algos
import torch
import torch.cuda
import yaml
from rlpyt.agents.base import BaseAgent
from rlpyt.experiment_utils.args import ExperimentArgs
from rlpyt.experiment_utils.utils import import_submodules
from rlpyt.samplers.collections import TrajInfo
from rlpyt.utils.logging import logger
from xvfbwrapper import Xvfb


def evaluate(
    args: ExperimentArgs,
    EnvCls: Callable,
    env_kwargs: Dict,
    TrajInfoCls=TrajInfo,
    run_id: Optional[str] = None,
):
    import_submodules(rlpyt.agents)
    import_submodules(rlpyt.algos)
    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    if run_id:
        api = wandb.Api()
        run = api.run(run_id)
        run_path = os.path.join(LOG_DIR, "eval-runs", run_id)
        if not os.path.exists(run_path):
            os.makedirs(run_path)
            for file in run.files():
                file.download(root=run_path)
        args.restore = os.path.join(run_path, "params.pkl")
    envs = [EnvCls(**env_kwargs)]
    agent: BaseAgent = rlpyt.agents.agent_registry[config["agent"]](
        **config["agent_config"]
    )
    agent.initialize(envs[0].spaces)
    if torch.cuda.is_available():
        agent.to_device(0)
    if args.restore:
        logger.log(f"Restoring agent from {args.restore}")
        with open(args.restore, "rb") as f:
            state_dicts = torch.load(f)
        iteration = state_dicts["itr"]
        agent.load_state_dict(state_dicts["agent_state_dict"])
    else:
        iteration = 0
    collector = SerialEvalCollector(
        envs,
        agent,
        TrajInfoCls=TrajInfoCls,
        max_T=args.eval_max_steps,
        max_trajectories=args.eval_max_trajectories,
    )
    with Xvfb():
        with logger_context(
            args.name,
            dict(**args.as_dict(), **config, env_kwargs=env_kwargs),
            snapshot_mode="none",
            debug=args.debug,
        ):
            trajectories: List[TrajInfoCls] = collector.collect_evaluation(iteration)
    return trajectories
