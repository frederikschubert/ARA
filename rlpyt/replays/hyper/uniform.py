from rlpyt.replays.async_ import AsyncReplayBufferMixin
from rlpyt.replays.hyper.n_step import NStepReturnHyperBuffer
from rlpyt.replays.non_sequence.uniform import UniformReplay


class UniformHyperReplayBuffer(UniformReplay, NStepReturnHyperBuffer):
    pass
