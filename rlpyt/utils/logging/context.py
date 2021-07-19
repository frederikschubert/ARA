import datetime
import json
import os
from contextlib import contextmanager
from rlpyt.utils.logging import logger


LOG_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../data"))


@contextmanager
def logger_context(
    name,
    log_params=dict(),
    snapshot_mode="none",
    debug=False,
    init_kwargs=None,
):
    """Use as context manager around calls to the runner's ``train()`` method.
    Sets up the logger directory and filenames.

    Input ``snapshot_mode`` refers to how often the logger actually saves the
    snapshot (e.g. may include agent parameters).  The runner calls on the
    logger to save the snapshot at every iteration, but the input
    ``snapshot_mode`` sets how often the logger actually saves (e.g. snapshot
    may include agent parameters). Possible modes include (but check inside
    the logger itself):
        * "none": don't save at all
        * "last": always save and overwrite the previous
        * "all": always save and keep each iteration
        * "gap": save periodically and keep each (will also need to set the gap, not done here) 
    """
    init_kwargs = init_kwargs or dict(entity="tnt", project="rnd-iqn")
    run = logger.start_run(
        **{
            **{
                "name": name,
                "mode": "offline" if debug else "run",
                "config": log_params,
                "dir": LOG_DIR,
            },
            **init_kwargs,
        }
    )
    logger.set_snapshot_mode(snapshot_mode)
    logger.push_prefix(f"{name}")

    yield

    logger.pop_prefix()
