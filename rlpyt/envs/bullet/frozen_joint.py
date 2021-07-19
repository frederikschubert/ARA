from gym import Wrapper


import numpy as np


class FrozenJointEnv(Wrapper):
    def __init__(
        self,
        wrapped_env,
        max_broken_axes=2,
        break_double_axis=True,
        break_at=None,
        break_prob=0.01,
    ):
        super().__init__(wrapped_env)
        self.joints = np.arange(len(self.env.action_space.low))
        self.broken_axes = []
        self.max_broken = max_broken_axes
        self.break_whole_joint = break_double_axis
        self.num_steps = 0
        if break_at:
            self.break_times = break_at
            self.break_prob = None
        elif break_prob:
            self.break_prob = break_prob
        else:
            raise Exception(
                "Provide either timepoints for joints to break or breakage probabilities."
            )

    def step(self, action):
        self.num_steps += 1
        if len(self.broken_axes) < self.max_broken:
            self.update_breaks()

        if self.broken_axes:
            action[self.broken_axes] = 0
        state, reward, done, info = self.env.step(action)
        return state, reward, done, info

    def update_breaks(self):
        break_joint = False
        if self.break_prob:
            break_joint = np.random.uniform(0, 1) < self.break_prob
        else:
            if (
                self.num_steps
                >= self.break_times[
                    len(self.broken_axes) // (int(self.break_whole_joint) + 1)
                ]
            ):
                break_joint = True

        if break_joint:
            unbroken = [j for j in self.joints if j not in self.broken_axes]
            if self.break_whole_joint:
                unbroken = [j for j in unbroken if j % 2 == 0]
            to_break = np.random.choice(unbroken)
            self.broken_axes.append(to_break)
            if self.break_whole_joint and to_break + 1 in self.joints:
                self.broken_axes.append(to_break + 1)
