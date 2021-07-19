import os
from typing import List, Optional

from tap import Tap


class ExperimentArgs(Tap):
    config: str
    steps: Optional[int] = None
    name: Optional[str] = None
    group: Optional[str] = None
    debug: bool = False
    sync_sample: bool = False
    gpus: List[int] = [0]
    cpus: int = os.cpu_count() or 1
    gpu_sample: bool = False
    log_interval_steps: int = int(1e5)
    eval_n_envs: int = 0
    eval_max_steps: int = int(1e3)
    eval_max_trajectories: int = 100
    env_type: str = "rl"
    n_envs: int = 1
    batch_T: int = 5
    batch_B: Optional[int] = None
    restore: Optional[str] = None
    frame_stack: bool = False
    seed: Optional[int] = None
    

    def process_args(self) -> None:
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join([str(gpu) for gpu in self.gpus])
        self.eval_max_steps *= self.cpus
        if self.debug:
            self.sync_sample = True
        return super().process_args()
