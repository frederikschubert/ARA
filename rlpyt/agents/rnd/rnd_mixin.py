from abc import abstractmethod
import numpy as np

import torch
from torch import distributions
from rlpyt.models.rnd.rnd_model import RndModel
from rlpyt.models.utils import strip_ddp_state_dict
from rlpyt.utils.logging import logger
from torch.nn.parallel import DistributedDataParallel as DDP

class RndAgentMixin:
    def __init__(
        self,
        cvar_lower_bound: float = 0.01,
        cvar_slope: float = 3.0,
        rnd_mapping="standard",
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.rnd_model = None
        self.shared_rnd_model = None
        self.cvar_lower_bound = cvar_lower_bound
        self.cvar_slope = cvar_slope
        self.rnd_mapping = rnd_mapping

    @abstractmethod
    def get_tau(self, *args, **kwargs):
        pass

    def compute_risk_param(self, observation: torch.Tensor, prev_action: torch.Tensor):
        rnd_error = self.rnd_model(observation, prev_action=prev_action)
        rnd_error_distribution = distributions.HalfNormal(scale=torch.ones_like(rnd_error) * self.rnd_error_std)
        rnd_error_prob = 1.0 - rnd_error_distribution.cdf(rnd_error)
        if self.risk_mode == "wang":
            param = torch.cos(rnd_error_prob * np.pi / 2.0)
        elif self.risk_mode == "cvar":
            param = rnd_error_prob
        else:
            raise ValueError("Only risk_mode wang and cvar are supported")
        return param

    @property
    def rnd_error_std(self):
        return torch.sqrt(self.rnd_error_mean_std.var)

    @property
    def rnd_error_mean_std(self):
        if type(self.rnd_model) == DDP:
            return self.rnd_model.module.rnd_error_mean_std
        else:
            return self.rnd_model.rnd_error_mean_std

    @property
    def rnd_input_shape(self):
        return (
            self.env_model_kwargs["image_shape"]
            if "image_shape" in self.env_model_kwargs
            else self.env_model_kwargs["observation_shape"]
        )

    @property
    def rnd_output_size(self):
        return (
            self.env_model_kwargs["output_size"]
            if "output_size" in self.env_model_kwargs
            else self.env_model_kwargs["action_size"]
        )

    def initialize(self, env_spaces, share_memory=False, **kwargs):
        super().initialize(env_spaces, share_memory, **kwargs)
        self.rnd_model = RndModel(self.rnd_input_shape, self.rnd_output_size)
        if share_memory:
            self.rnd_model.share_memory()
            self.shared_rnd_model = self.rnd_model

    def to_device(self, cuda_idx=None):
        super().to_device(cuda_idx)
        if self.shared_rnd_model is not None:
            self.rnd_model = RndModel(self.rnd_input_shape, self.rnd_output_size)
            self.rnd_model.load_state_dict(self.shared_rnd_model.state_dict())
        self.rnd_model.to(self.device)
        logger.log(f"Initialized agent rnd model on device: {self.device}.")

    def data_parallel(self):
        device_id = super().data_parallel()
        self.rnd_model = DDP(
            self.rnd_model,
            device_ids=None if device_id is None else [device_id],
            output_device=device_id,
        )
        logger.log(
            "Initialized DistributedDataParallel agent rnd model on "
            f"device {self.device}."
        )
        return device_id

    def async_cpu(self, share_memory=True):
        super().async_cpu(share_memory)
        if self.device.type != "cpu":
            return
        assert self.shared_rnd_model is not None
        self.rnd_model = RndModel(self.rnd_input_shape, self.rnd_output_size)
        self.rnd_model.load_state_dict(
            strip_ddp_state_dict(self.shared_rnd_model.state_dict())
        )
        if share_memory:
            self.rnd_model.share_memory()
        logger.log("Initialized async CPU agent rnd model.")

    def state_dict(self):
        return dict(model=super().state_dict(), rnd_model=self.rnd_model.state_dict())

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict["model"])
        self.rnd_model.load_state_dict(state_dict["rnd_model"])

    def train_mode(self, itr):
        super().train_mode(itr)
        self.rnd_model.train()

    def sample_mode(self, itr):
        super().sample_mode(itr)
        self.rnd_model.eval()

    def eval_mode(self, itr):
        super().eval_mode(itr)
        self.rnd_model.eval()

    def sync_shared_memory(self):
        super().sync_shared_memory()
        if self.shared_rnd_model is not self.rnd_model:
            self.shared_rnd_model.load_state_dict(
                strip_ddp_state_dict(self.rnd_model.state_dict())
            )

    def send_shared_memory(self):
        if (
            self.shared_model is not self.model
            or self.shared_rnd_model is not self.rnd_model
        ):
            with self._rw_lock.write_lock:
                self.shared_model.load_state_dict(
                    strip_ddp_state_dict(self.model.state_dict())
                )
                self.shared_rnd_model.load_state_dict(
                    strip_ddp_state_dict(self.rnd_model.state_dict())
                )
                self._send_count.value += 1

    def recv_shared_memory(self):
        if (
            self.shared_model is not self.model
            or self.shared_rnd_model is not self.rnd_model
        ):
            with self._rw_lock:
                if self._recv_count < self._send_count.value:
                    self.model.load_state_dict(self.shared_model.state_dict())
                    self.rnd_model.load_state_dict(self.shared_rnd_model.state_dict())
                    self._recv_count = self._send_count.value
