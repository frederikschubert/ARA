from collections import namedtuple
from typing import List

import numpy as np
import torch.nn
from rlpyt.utils.buffer import buffer_to
from rlpyt.utils.logging import logger


class RND:
    def __init__(
        self,
        rnd_learning_rate: float = 1e-3,
        rnd_clip_grad_norm: float = 100,
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.rnd_learning_rate = rnd_learning_rate
        self.rnd_clip_grad_norm = rnd_clip_grad_norm
        self.opt_info_fields = super().opt_info_fields + ("rndLoss", "rndGradNorm")
        self.OptInfoCls = namedtuple("OptInfo", self.opt_info_fields)
        self.itr_samples = []

    def rnd_loss(self, samples):
        states, next_states, actions = buffer_to(
            (
                samples.agent_inputs.observation,
                samples.target_inputs.observation,
                samples.action,
            ),
            device=self.agent.device,
        )
        rnd_errors = self.agent.rnd_model(
            torch.cat([states, states - next_states], dim=-1), prev_action=actions
        )
        return rnd_errors

    def loss(self, samples):
        self.itr_samples.append(samples)
        return super().loss(samples)

    def optim_initialize(self, rank=0):
        super().optim_initialize(rank)
        self.rnd_optimizer = self.OptimCls(
            self.agent.rnd_model.parameters(), lr=self.rnd_learning_rate
        )

    def optimize_agent(self, itr, samples=None, sampler_itr=None):
        opt_info = super().optimize_agent(itr, samples, sampler_itr)
        itr = itr if sampler_itr is None else sampler_itr
        if itr < self.min_itr_learn:
            return self.OptInfoCls(
                **opt_info._asdict(), **dict(rndLoss=np.nan, rndGradNorm=np.nan)
            )
        # rnd_samples = self.replay_buffer.sample_batch(self.batch_size)
        for rnd_samples in self.itr_samples:
            self.rnd_optimizer.zero_grad()
            rnd_errors = self.rnd_loss(rnd_samples)
            logger.record_histogram("RND/rndErrors", rnd_errors.detach().cpu().numpy())
            rnd_loss = rnd_errors.mean()
            rnd_loss.backward()
            rnd_grad_norm = torch.nn.utils.clip_grad_norm_(
                self.agent.rnd_model.parameters(), self.rnd_clip_grad_norm
            )
            self.rnd_optimizer.step()
            self.agent.rnd_error_mean_std.update(rnd_errors)
        self.itr_samples.clear()
        opt_info_extended = self.OptInfoCls(
            **opt_info._asdict(),
            **dict(rndLoss=rnd_loss.item(), rndGradNorm=rnd_grad_norm.item())
        )
        return opt_info_extended

