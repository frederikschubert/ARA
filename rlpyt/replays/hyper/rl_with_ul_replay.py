import numpy as np
from rlpyt.replays.non_sequence.frame import (
    PrioritizedReplayFrameBuffer,
    UniformReplayFrameBuffer,
)
from rlpyt.utils.buffer import buffer_func, torchify_buffer
from rlpyt.utils.collections import namedarraytuple
from rlpyt.utils.misc import extract_sequences

SamplesFromReplay = namedarraytuple(
    "SamplesFromReplay", ["observation", "action", "reward", "done"]
)


class DqnWithUlReplayBufferMixin:
    """Mixes with the replay buffer for DQN. 
    No prioritized for now."""

    def __init__(self, ul_replay_T, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.ul_replay_T = ul_replay_T

    def ul_sample_batch(self, batch_B):
        T_idxs, B_idxs = self.ul_sample_idxs(batch_B)
        return self.ul_extract_batch(T_idxs, B_idxs, self.ul_replay_T)

    def ul_sample_idxs(self, batch_B):
        t, b, f = self.t, self.ul_replay_T, 0  # cursor, off_backward, off_forward
        high = self.T - b - f if self._buffer_full else t - b - f
        T_idxs = np.random.randint(low=0, high=high, size=(batch_B,))
        T_idxs[T_idxs >= t - b] += min(t, b) + f
        B_idxs = np.random.randint(low=0, high=self.B, size=(batch_B,))
        return T_idxs, B_idxs

    def ul_extract_batch(self, T_idxs, B_idxs, T):
        s = self.samples
        batch = SamplesFromReplay(
            observation=self.ul_extract_observation(T_idxs, B_idxs, T),
            action=buffer_func(s.action, extract_sequences, T_idxs, B_idxs, T),
            reward=extract_sequences(s.reward, T_idxs, B_idxs, T),
            done=extract_sequences(s.done, T_idxs, B_idxs, T),
        )
        return torchify_buffer(batch)

    def ul_extract_observation(self, T_idxs, B_idxs, T):
        """Observations are re-assembled from frame-wise buffer as [T,B,C,H,W],
        where C is the frame-history channels, which will have redundancy across the
        T dimension.  Frames are returned OLDEST to NEWEST along the C dimension.

        Frames are zero-ed after environment resets."""
        observation = np.empty(
            shape=(T, len(B_idxs), self.n_frames)
            + self.samples_frames.shape[2:],  # [T,B,C,H,W]
            dtype=self.samples_frames.dtype,
        )
        fm1 = self.n_frames - 1
        for i, (t, b) in enumerate(zip(T_idxs, B_idxs)):
            if t + T > self.T:  # wrap (n_frames duplicated)
                m = self.T - t
                w = T - m
                for f in range(self.n_frames):
                    observation[:m, i, f] = self.samples_frames[t + f : t + f + m, b]
                    observation[m:, i, f] = self.samples_frames[f : w + f, b]
            else:
                for f in range(self.n_frames):
                    observation[:, i, f] = self.samples_frames[t + f : t + f + T, b]

            # Populate empty (zero) frames after environment done.
            if t - fm1 < 0 or t + T > self.T:  # Wrap.
                done_idxs = np.arange(t - fm1, t + T) % self.T
            else:
                done_idxs = slice(t - fm1, t + T)
            done_fm1 = self.samples.done[done_idxs, b]
            if np.any(done_fm1):
                where_done_t = np.where(done_fm1)[0] - fm1  # Might be negative...
                for f in range(1, self.n_frames):
                    t_blanks = where_done_t + f  # ...might be > T...
                    t_blanks = t_blanks[
                        (t_blanks >= 0) & (t_blanks < T)
                    ]  # ..don't let it wrap.
                    observation[t_blanks, i, : self.n_frames - f] = 0

        return observation


class DqnWithUlUniformReplayFrameBuffer(
    DqnWithUlReplayBufferMixin, UniformReplayFrameBuffer
):
    pass


class DqnWithUlPrioritizedReplayFrameBuffer(
    DqnWithUlReplayBufferMixin, PrioritizedReplayFrameBuffer
):
    pass
