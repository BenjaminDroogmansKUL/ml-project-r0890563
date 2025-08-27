"""
This file contains an example of implementation of the CustomWapper and CustomPredictFunction that you need to submit.

Here, we are using Ray RLLib to load the trained agents.
"""

from pathlib import Path
from typing import Optional

import gymnasium
import torch
from gymnasium import spaces
from pettingzoo.utils import BaseWrapper
from pettingzoo.utils.env import AgentID, ObsType
from ray.rllib.core.rl_module import MultiRLModule


class CustomWrapper(BaseWrapper):
    """An example of a custom wrapper that flattens the symbolic vector state of the environment.

    Wrappers are useful to inject state pre-processing or features that do not need to be learned by the agent.

    Pay attention to submit the same (or consistent) wrapper you used during training.
    """

    def observation_space(self, agent: AgentID) -> gymnasium.spaces.Space:
        return spaces.flatten_space(super().observation_space(agent))

    def observe(self, agent: AgentID) -> Optional[ObsType]:
        obs = super().observe(agent)
        flat_obs = obs.flatten()
        return flat_obs


class CustomPredictFunction:
    """An example of an instantiation of the PredictFunction interface that loads a trained RLLib algorithm from
    a checkpoint and extracts the policies from it.
    """

    def __init__(self, env):
        # Here you should load your trained model(s) from a checkpoint in your folder
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
        fwd_ins = {"obs": torch.Tensor(observation).unsqueeze(0)}
        fwd_outputs = rl_module.forward_inference(fwd_ins)
        action_dist_class = rl_module.get_inference_action_dist_cls()
        action_dist = action_dist_class.from_logits(fwd_outputs["action_dist_inputs"])
        action = action_dist.sample()[0].numpy()
        return action
