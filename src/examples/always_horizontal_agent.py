import math
from collections import defaultdict

import gymnasium
import numpy as np
from gymnasium import spaces
from pettingzoo.utils import BaseWrapper
from pettingzoo.utils.env import AgentID, ObsType


class CustomWrapper(BaseWrapper):
    """Flatten the observation so we can print/debug easily."""

    def __init__(self, env):
        super().__init__(env)
        self.episode_no = 0

    def reset(self, seed=None, options=None):
        out = super().reset(seed=seed, options=options)
        self.episode_no += 1
        return out

    def observation_space(self, agent: AgentID) -> gymnasium.spaces.Space:
        return spaces.flatten_space(super().observation_space(agent))

    def observe(self, agent: AgentID) -> ObsType | None:
        obs = super().observe(agent)
        return None if obs is None else obs.flatten()


class CustomPredictFunction:
    """Debug baseline: try to rotate the archer toward a diagonal and print obs."""

    ACTIONS = {
        "NOOP": 5,
        "FORWARD": 0,
        "BACKWARD": 1,
        "ROTATE_LEFT": 2,
        "ROTATE_RIGHT": 3,
        "ATTACK": 4,
    }

    def __init__(self, env, align_tol_rad: float = 0.10):
        self.env = env
        self.step_count = defaultdict(int)
        self._last_episode_no = getattr(env, "episode_no", 0)

    def reset(self):
        print("done")
        self.step_count = 0

    def __call__(self, observation: np.ndarray, agent: str, *args, **kwargs) -> int:

        current_ep = getattr(self.env, "episode_no", 0)
        if current_ep != self._last_episode_no:
            self.step_count.clear()  # zero all agents
            self._last_episode_no = current_ep

        # --- print full flattened obs ---
        print(f"\n[Agent {agent}] obs len={len(observation)}")
        print(observation)  # first 20 entries for sanity

        if observation is None:
            self.step_count[agent] = 0
            return self.ACTIONS["NOOP"]

        self.step_count[agent] += 1

        if self.step_count[agent] > 37:
            print(" >>> Forcing ATTACK")
            return self.ACTIONS["ATTACK"]

        if self.step_count[agent] > 20:
            print(" forcing right")
            return self.ACTIONS["ROTATE_RIGHT"]

        if self.step_count[agent] > 8:
            print(" >>> Forcing FORWARD")
            return self.ACTIONS["FORWARD"]

        if self.step_count[agent] > 0:
            print(" forcing left")
            return self.ACTIONS["ROTATE_LEFT"]

