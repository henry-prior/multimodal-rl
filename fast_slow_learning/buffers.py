from haiku import PRNGSequence
from typing import Dict
from typing import Sequence
from typing import Tuple

import jax
import numpy as np
from jax import random


class MultiModalReplayBuffer:
    """
    A replay buffer which stores trajectories for whole episodes. Assumes that
    there are states for both language and vision.
    """

    def __init__(
        self,
        max_size: int,
        episode_len: int,
        language_dim: int,
        vision_dim: Tuple[int, int],
        action_dim: int,
        extras: Tuple[str, Sequence[int]] = None,
    ):
        self.max_size = max_size
        self.ptr = 0
        self.size = 0

        self.language_states = np.zeros((max_size, episode_len, language_dim))
        self.vision_states = np.zeros((max_size, episode_len, *vision_dim))
        self.actions = np.zeros((max_size, episode_len, action_dim))
        self.rewards = np.zeros((max_size, episode_len, 1))
        self.discounts = np.ones((max_size, episode_len, 1))
        self.extras = (
            {
                extra_name: np.zeros((max_size, episode_len, *extra_size))
                for extra_name, extra_size in extras.items()
            }
            if extras is not None
            else dict()
        )

    def add(
        self,
        language_states: np.ndarray,
        vision_states: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        discounts: np.ndarray,
        extras: Dict[str, np.ndarray] = None,
    ):
        self.language_states[self.ptr] = language_states
        self.vision_states[self.ptr] = vision_states
        self.actions[self.ptr] = actions
        self.rewards[self.ptr] = rewards
        self.discounts[self.ptr] = discounts

        if extras is not None:
            for k, v in extras.items():
                self.extras[k][self.ptr] = v

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, rng: PRNGSequence, batch_size: int):
        ind = random.randint(rng, (batch_size,), 0, self.size)

        return (
            jax.device_put(self.language_state[ind]),
            jax.device_put(self.vision_state[ind]),
            jax.device_put(self.actions[ind]),
            jax.device_put(self.rewards[ind]),
            jax.device_put(self.discounts[ind]),
            jax.device_put(self.discounts[ind]),
            {k: jax.device_put(v[ind]) for k, v in self.extras.items()},
        )
