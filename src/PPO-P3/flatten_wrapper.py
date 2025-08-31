import numpy as np
import gymnasium
from gymnasium import spaces
from pettingzoo.utils import BaseWrapper
from pettingzoo.utils.env import AgentID, ObsType


class FlattenWrapper(BaseWrapper):
    """Flattens the symbolic vector state of the environment.

    IMPORTANT: Use the same (or a consistent) wrapper as in training.
    """
    def __init__(self, env):
        super().__init__(env)

    def observation_space(self, agent: AgentID) -> gymnasium.spaces.Space:
        return spaces.flatten_space(super().observation_space(agent))

    def observe(self, agent: AgentID) -> ObsType | None:
        obs = super().observe(agent)
        if obs is None:
            return None
        return np.asarray(obs, dtype=np.float32).flatten()
