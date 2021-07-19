import atexit
from gc import callbacks
import os
import sys
from contextlib import contextmanager

import numpy as np
import progressbar as pbar
import torch
import wandb
from loguru import logger as _logger
from tabulate import tabulate
from torchvision.utils import make_grid
from torchvision.transforms.functional import to_pil_image

_logger.remove()
_logger.add(
    sys.stdout,
    colorize=True,
    format="<fg #808080>{time:DD.MM.YYYY HH:mm:ss:SSS}</fg #808080> - <level>{message}</level>",
    backtrace=True,
    diagnose=True,
)
_prefixes = []

_tabular_prefixes = []
_tabular_prefix_str = ""

_snapshot_mode = "all"
_snapshot_gap = 1

_run = None
_record_trajectory_flag = None
_callbacks = {}


def start_run(*args, **kwargs):
    global _run
    _run = wandb.init(*args, **kwargs)
    return _run


def register_callback(steps, cb):
    global _callbacks
    _callbacks[int(steps)] = cb
    log(f"Registered callback {cb} at {steps}")


def progress_bar(total: int):
    widgets = [
        " [",
        pbar.Timer(),
        "] ",
        pbar.Bar(),
        " (",
        pbar.AdaptiveETA(),
        ") ",
    ]
    return pbar.ProgressBar(max_value=total, widgets=widgets, is_terminal=True)


def set_record_trajectory_flag(flag):
    global _record_trajectory_flag
    _record_trajectory_flag = flag


def get_snapshot_dir():
    return _run.dir


def get_snapshot_mode():
    return _snapshot_mode


def set_snapshot_mode(mode):
    global _snapshot_mode
    _snapshot_mode = mode


def log(message: str):
    _logger.info(message)


def exception(message: str):
    _logger.exception(message)


def record_video(key, video, **kwargs):
    wandb_video = video if type(video) == wandb.Video else wandb.Video(video, **kwargs)
    _run.log({key: wandb_video}, commit=False)


def record_image(key, image, **kwargs):
    if type(image) == torch.Tensor and image.dim() == 4:
        if image.shape[0] > 64:
            image = image[::32].contiguous()
        if image.shape[1] != 3:
            image = to_pil_image(
                make_grid(image.view(-1, 1, *image.shape[-2:]), nrow=image.shape[1],)
            )
        else:
            image = to_pil_image(make_grid(image, nrow=8))
    wandb_image = image if type(image) == wandb.Image else wandb.Image(image, **kwargs)
    _run.log({key: wandb_image}, commit=False)


def record_histogram(key, vals):
    _run.log({key: wandb.Histogram(vals)}, commit=False)


def record_tabular(key, val):
    global _callbacks
    if key == "CumSteps":
        for steps in list(sorted(_callbacks.keys())):
            if steps < val:
                _callbacks[steps]()
                del _callbacks[steps]
    key = _tabular_prefix_str + str(key)
    _run.log({key: val}, commit=False)


def watch(model):
    wandb.watch(model, log_freq=1, log="all")


def record_tabular_misc_stat(key: str, values, placement="back"):
    if placement == "front":
        prefix = ""
        suffix = key
    else:
        prefix = key
        suffix = ""
    if key == "Observations":
        trajectories = [trajectory for trajectory in values if trajectory]
        if trajectories:
            num_trajectories = 0
            for trajectory in trajectories:
                trajectory = np.array(trajectory)
                if len(trajectory.shape) == 4 and (
                    trajectory.shape[1] == 3 or trajectory.shape[-1] == 3
                ):
                    if trajectory.shape[1] != 3 and trajectory.shape[-1] == 3:
                        trajectory = trajectory.transpose(0, 3, 1, 2)
                    num_trajectories += 1
                    record_video(key, trajectory, format="gif", fps=30)
    elif key.endswith("_hist"):
        record_histogram(key, values)
    elif len(values) > 0:
        prefix += "/"
        record_tabular(prefix + "Average" + suffix, np.average(values))
        record_tabular(prefix + "Std" + suffix, np.std(values))
        record_tabular(prefix + "Median" + suffix, np.median(values))
        record_tabular(prefix + "Min" + suffix, np.min(values))
        record_tabular(prefix + "Max" + suffix, np.max(values))
    else:
        prefix += "/"
        record_tabular(prefix + "Average" + suffix, np.nan)
        record_tabular(prefix + "Std" + suffix, np.nan)
        record_tabular(prefix + "Median" + suffix, np.nan)
        record_tabular(prefix + "Min" + suffix, np.nan)
        record_tabular(prefix + "Max" + suffix, np.nan)


def push_prefix(prefix):
    _prefixes.append(prefix)
    global _logger
    _logger = _logger.bind(prefix=" - " + " ".join(_prefixes))


def pop_prefix():
    del _prefixes[-1]
    global _logger
    _logger = _logger.bind(prefix=" - " + " ".join(_prefixes))


@contextmanager
def prefix(key):
    push_prefix(key)
    try:
        yield
    finally:
        pop_prefix()


def push_tabular_prefix(key):
    _tabular_prefixes.append(key)
    global _tabular_prefix_str
    _tabular_prefix_str = "".join(_tabular_prefixes)


def pop_tabular_prefix():
    del _tabular_prefixes[-1]
    global _tabular_prefix_str
    _tabular_prefix_str = "".join(_tabular_prefixes)


@contextmanager
def tabular_prefix(key):
    push_tabular_prefix(key)
    yield
    pop_tabular_prefix()


def dump_tabular(*args, **kwargs):
    rows = [
        (k, v)
        for k, v in _run.history._data.items()
        if type(v) not in [wandb.Image, wandb.Video, wandb.Histogram]
    ]
    _logger.info("\n" + tabulate(rows, headers=[], tablefmt="presto"))
    _run.log({})


def save_itr_params(itr, params):
    if _record_trajectory_flag:
        _record_trajectory_flag.value = 1
    if _snapshot_mode == "all":
        file_name = os.path.join(get_snapshot_dir(), "itr_%d.pkl" % itr)
    elif _snapshot_mode == "last":
        # override previous params
        file_name = os.path.join(get_snapshot_dir(), "params.pkl")
    elif _snapshot_mode == "gap":
        if itr == 0 or (itr + 1) % _snapshot_gap == 0:
            file_name = os.path.join(get_snapshot_dir(), "itr_%d.pkl" % itr)
        else:
            return
    elif _snapshot_mode == "last+gap":
        if itr == 0 or (itr + 1) % _snapshot_gap == 0:
            file_name = os.path.join(get_snapshot_dir(), "itr_%d.pkl" % itr)
            torch.save(params, file_name)
        file_name = os.path.join(get_snapshot_dir(), "params.pkl")
    elif _snapshot_mode == "none":
        return
    else:
        raise NotImplementedError
    torch.save(params, file_name)
