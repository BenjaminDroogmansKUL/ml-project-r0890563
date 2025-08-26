import sys
from pathlib import Path
_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))


"""
This file contains an example of implementation of the CustomWrapper and CustomPredictFunction
that you need to submit.

Here, we are using Ray RLLib to load the trained agents (DQN, new API stack).
"""

from pathlib import Path
from typing import Optional

import gymnasium
import numpy as np
import torch
from gymnasium import spaces
from pettingzoo.utils import BaseWrapper
from pettingzoo.utils.env import AgentID, ObsType
from ray.rllib.core.rl_module import MultiRLModule


class CustomWrapper(BaseWrapper):
    """Flattens the symbolic vector state of the environment.

    IMPORTANT: Use the same (or a consistent) wrapper as in training.
    """

    def observation_space(self, agent: AgentID) -> gymnasium.spaces.Space:
        return spaces.flatten_space(super().observation_space(agent))

    def observe(self, agent: AgentID) -> Optional[ObsType]:
        obs = super().observe(agent)
        flat_obs = obs.flatten()
        return flat_obs


class CustomPredictFunction:
    """Loads a trained DQN MultiRLModule and returns greedy actions via argmax(Q)."""

    def __init__(self, env):
        package_directory = Path(__file__).resolve().parent
        best_checkpoint = (
            package_directory / "results" / "learner_group" / "learner" / "rl_module"
        ).resolve()

        if not best_checkpoint.exists():
            raise FileNotFoundError(
                f"Checkpoint directory not found: {best_checkpoint}"
            )

        self.modules = MultiRLModule.from_checkpoint(best_checkpoint)

    def __call__(self, observation, agent, *args, **kwargs):
        if agent not in self.modules:
            raise ValueError(f"No policy found for agent {agent}")

        rl_module = self.modules[agent]

        if isinstance(observation, np.ndarray):
            obs_tensor = torch.from_numpy(observation).float().unsqueeze(0)
        else:
            obs_tensor = torch.tensor(observation, dtype=torch.float32).unsqueeze(0)

        with torch.no_grad():
            fwd_out = rl_module.forward_inference({"obs": obs_tensor})
            q_values = fwd_out["q_values"]
            action = int(torch.argmax(q_values, dim=1).item())

        return action
