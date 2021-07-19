from functools import partial
from rlpyt.samplers.parallel.gpu.alternating_sampler import AlternatingSampler
from rlpyt.samplers.parallel.gpu.collectors import GpuResetCollector
from typing import Dict

from rlpyt.experiment_utils.args import ExperimentArgs
from rlpyt.runners.async_rl import AsyncRl, AsyncRlEval
from rlpyt.runners.minibatch_rl import MinibatchRl, MinibatchRlEval
from rlpyt.runners.sync_rl import SyncRl, SyncRlEval
from rlpyt.samplers.async_.collectors import DbCpuResetCollector, DbGpuResetCollector
from rlpyt.samplers.async_.cpu_sampler import AsyncCpuSampler
from rlpyt.samplers.async_.gpu_sampler import AsyncGpuSampler
from rlpyt.samplers.parallel.cpu.collectors import CpuResetCollector
from rlpyt.samplers.parallel.cpu.sampler import CpuSampler
from rlpyt.samplers.parallel.gpu.sampler import GpuSampler
from rlpyt.samplers.serial.sampler import SerialSampler
from rlpyt.utils.launching.affinity import (
    affinity_from_code,
    encode_affinity,
    prepend_run_slot,
)


def get_affinity_sampler_runner(args: ExperimentArgs):
    if args.sync_sample:
        if args.cpus == 1 and len(args.gpus) == 1:
            Sampler = SerialSampler
            Runner = MinibatchRlEval if args.eval_n_envs > 0 else MinibatchRl
            affinity = dict(cuda_idx=args.gpus[0], set_affinity=False)
        else:
            Sampler = GpuSampler if args.gpu_sample else CpuSampler
            if len(args.gpus) > 1:
                Runner = SyncRlEval if args.eval_n_envs > 0 else SyncRl
                affinity = [
                    dict(
                        cuda_idx=cuda_idx,
                        workers_cpus=list(range(args.cpus)),
                        set_affinity=False,
                    )
                    for cuda_idx in args.gpus
                ]
            elif len(args.gpus) == 1:
                Runner = MinibatchRlEval if args.eval_n_envs > 0 else MinibatchRl
                affinity = dict(
                    cuda_idx=args.gpus[0],
                    workers_cpus=list(range(args.cpus)),
                    set_affinity=False,
                )
            else:
                affinity = dict(
                    workers_cpus=list(range(args.cpus)), set_affinity=False,
                )
        CollectorCls = GpuResetCollector if args.gpu_sample else CpuResetCollector
    else:
        affinity = affinity_from_code(
            prepend_run_slot(
                0,
                encode_affinity(
                    n_cpu_core=args.cpus,
                    n_gpu=len(args.gpus),
                    gpu_per_run=len(args.gpus),
                    n_socket=1,
                    set_affinity=False,
                    async_sample=True,
                    sample_gpu_per_run=len(args.gpus) if args.gpu_sample else 0,
                    optim_sample_share_gpu=True if args.gpu_sample else False,
                ),
            )
        )
        Sampler = AsyncGpuSampler if args.gpu_sample else AsyncCpuSampler
        Runner = AsyncRlEval if args.eval_n_envs > 0 else AsyncRl
        CollectorCls = DbGpuResetCollector if args.gpu_sample else DbCpuResetCollector
    return (
        affinity,
        partial(
            Sampler,
            batch_T=args.batch_T,
            batch_B=args.batch_B
            if args.batch_B is not None
            else args.n_envs
            * (args.cpus if args.sync_sample else max(1, args.cpus - len(args.gpus))),
            eval_n_envs=args.eval_n_envs,
            eval_max_steps=args.eval_max_steps,
            eval_max_trajectories=args.eval_max_trajectories,
            CollectorCls=CollectorCls,
        ),
        partial(Runner, n_steps=args.steps, log_interval_steps=args.log_interval_steps, seed=args.seed),
    )
