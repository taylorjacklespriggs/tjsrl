import numpy as np
from random import randrange, sample

class SampleStore:
    def __init__(self, max_samples):
        self.max_samples = max_samples
        self.samples = []

    def add_sample(self, state, action, new_state, reward):
        if len(self.samples) == self.max_samples:
            self._remove_randomly()
        self.samples.append((state, action, new_state, reward))

    def _remove_randomly(self):
        l = len(self.samples)
        indices = np.arange(l)
        p_dist = np.float32(.99) ** (indices / l)
        p_dist /= p_dist.sum()
        self.samples.pop(np.random.choice(
            np.arange(len(self.samples)),
            p=p_dist
        ))

    def get_batch(self, size=None):
        if size == None:
            size = self.max_samples
        if len(self.samples) <= size:
            samples = self.samples
        else:
            samples = sample(self.samples, size)
        states = [s[0] for s in samples]
        actions = [s[1] for s in samples]
        new_states = [s[2] for s in samples]
        rewards = [s[3] for s in samples]
        return map(np.array, (
            states,
            actions,
            new_states,
            rewards,
        ))

    def get_all(self):
        samples = self.samples
        states = [s[0] for s in samples]
        actions = [s[1] for s in samples]
        new_states = [s[2] for s in samples]
        rewards = [s[3] for s in samples]
        return map(np.array, (
            states,
            actions,
            new_states,
            rewards,
        ))
