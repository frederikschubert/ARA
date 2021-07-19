import numpy as np
from rlpyt.agents.base import AgentInputs
from rlpyt.replays.n_step import BaseNStepReturnBuffer
from rlpyt.replays.non_sequence.n_step import (
    SamplesFromReplay as NStepSamplesFromReplay,
)
from rlpyt.utils.buffer import torchify_buffer
from rlpyt.utils.collections import namedarraytuple

SamplesFromReplay = namedarraytuple(
    "SamplesFromReplay",
    NStepSamplesFromReplay._fields + ("hyper_observation", "hyper_done"),
)


class NStepReturnHyperBuffer(BaseNStepReturnBuffer):
    """Definition of what fields are replayed from basic n-step return buffer."""

    def __init__(self, hyper_steps: int = 5, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.hyper_steps = hyper_steps
        self.off_backward = max(self.hyper_steps, self.n_step_return)

    def extract_batch(self, T_idxs, B_idxs):
        """From buffer locations `[T_idxs,B_idxs]`, extract data needed for
        training, including target values at `T_idxs + n_step_return`.  Returns
        namedarraytuple of torch tensors (see file for all fields).  Each tensor
        has leading batch dimension ``len(T_idxs)==len(B_idxs)``, but individual
        samples are drawn, so no leading time dimension."""
        s = self.samples
        target_T_idxs = (T_idxs + self.n_step_return) % self.T
        hyper_steps = np.random.randint(1, self.hyper_steps)
        hyper_idxs = (T_idxs + hyper_steps) % self.T
        batch = SamplesFromReplay(
            agent_inputs=AgentInputs(
                observation=self.extract_observation(T_idxs, B_idxs),
                prev_action=s.action[T_idxs - 1, B_idxs],
                prev_reward=s.reward[T_idxs - 1, B_idxs],
            ),
            action=s.action[T_idxs, B_idxs],
            return_=self.samples_return_[T_idxs, B_idxs],
            done=self.samples.done[T_idxs, B_idxs],
            done_n=self.samples_done_n[T_idxs, B_idxs],
            target_inputs=AgentInputs(
                observation=self.extract_observation(target_T_idxs, B_idxs),
                prev_action=s.action[target_T_idxs - 1, B_idxs],
                prev_reward=s.reward[target_T_idxs - 1, B_idxs],
            ),
            hyper_observation=self.extract_observation(hyper_idxs, B_idxs),
            hyper_done=self.extract_done(T_idxs, B_idxs, self.hyper_steps),
        )
        t_news = np.where(s.done[T_idxs - 1, B_idxs])[0]
        batch.agent_inputs.prev_action[t_news] = 0
        batch.agent_inputs.prev_reward[t_news] = 0
        return torchify_buffer(batch)

    def extract_done(self, T_idxs, B_idxs, T):
        fm1 = self.hyper_steps - 1
        done = np.zeros((len(T_idxs),), np.bool)
        for i, (t, b) in enumerate(zip(T_idxs, B_idxs)):
            if t - fm1 < 0 or t + T > self.T:  # Wrap.
                done_idxs = np.arange(t - fm1, t + T) % self.T
            else:
                done_idxs = slice(t - fm1, t + T)
            done_fm1 = self.samples.done[done_idxs, b]
            if np.any(done_fm1):
                # where_done_t = np.where(done_fm1)[0]
                done[i] = True
        return done

    def extract_observation(self, T_idxs, B_idxs):
        """Simply ``observation[T_idxs,B_idxs]``; generalization anticipating
        frame-based buffer."""
        return self.samples.observation[T_idxs, B_idxs]
