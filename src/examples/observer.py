import gymnasium
import numpy as np
from gymnasium import spaces
from pettingzoo.utils import BaseWrapper
from pettingzoo.utils.env import AgentID, ObsType


class CustomWrapper(BaseWrapper):
    def __init__(self, env):
        super().__init__(env)

    def observation_space(self, agent: AgentID) -> gymnasium.spaces.Space:
        return spaces.flatten_space(super().observation_space(agent))

    def observe(self, agent: AgentID) -> ObsType | None:
        obs = super().observe(agent)
        flat_obs = obs.flatten()
        return obs


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
        self.step_count = 0
        self._last_episode_no = getattr(env, "episode_no", 0)

    def reset(self):
        print("done")
        self.step_count = 0

    def __call__(self, observation: np.ndarray, agent: str, *args, **kwargs) -> int:
        print(f"\n[Agent {agent}] obs len={len(observation)}")
        print(observation)

        if observation is None:
            return self.ACTIONS["NOOP"]

        self.step_count += 1

        return self.ACTIONS["NOOP"]
