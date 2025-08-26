import sys
from pathlib import Path

from ray.rllib.utils.replay_buffers import StorageUnit

_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))



#!/usr/bin/env python3
# encoding: utf-8
"""
Minimal DQN training for PettingZoo KAZ using RLlib's NEW API stack.
This mirrors your PPO template but switches to DQN and the essentials it needs:
- uniform replay buffer
- epsilon-greedy exploration
- target network updates
- default DQN RLModule with a small MLP
"""

from pathlib import Path
from typing import Callable, Optional

import gymnasium
import pettingzoo
from gymnasium import spaces
from pettingzoo.utils import BaseWrapper
from pettingzoo.utils.env import AgentID, ObsType
from ray.rllib.algorithms.dqn import DQNConfig
from ray.rllib.core.rl_module import MultiRLModule
from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv
from ray.tune.registry import register_env
import numpy as np
import torch

from utils import create_environment


# ---------- Wrapper (unchanged) ----------

class CustomWrapper(BaseWrapper):
    # Keep vector observations flattened; DQN requires a fixed-size tensor.
    def observation_space(self, agent: AgentID) -> gymnasium.spaces.Space:
        return spaces.flatten_space(super().observation_space(agent))

    def observe(self, agent: AgentID) -> ObsType | None:
        obs = super().observe(agent)
        flat_obs = obs.flatten()
        return flat_obs


# ---------- Predict function (DQN: argmax Q) ----------

class CustomPredictFunction(Callable):
    """
    Loads RLlib MultiRLModule checkpoint and selects greedy actions via argmax(Q).
    """

    def __init__(self, env):
        best_checkpoint = (Path("results") / "learner_group" / "learner" / "rl_module").resolve()
        self.modules = MultiRLModule.from_checkpoint(best_checkpoint)

    def __call__(self, observation, agent, *args, **kwargs):
        rl_module = self.modules[agent]
        # Forward in NEW API returns algo-specific keys; for DQN we expect "q_values".
        fwd_ins = {"obs": torch.tensor(observation, dtype=torch.float32).unsqueeze(0)}
        fwd_out = rl_module.forward_inference(fwd_ins)
        q_values = fwd_out["q_values"]  # shape: [B, num_actions]
        action = torch.argmax(q_values, dim=1)[0].item()
        return action


# ---------- Config builder (PPO → DQN) ----------

def algo_config(id_env, policies, policies_to_train):
    # Minimal, robust DQN knobs for vector KAZ
    config = (
        DQNConfig()
        .api_stack(
            enable_rl_module_and_learner=True,
            enable_env_runner_and_connector_v2=True,
        )
        .environment(env=id_env, disable_env_checking=True)
        .env_runners(num_env_runners=1,
                     rollout_fragment_length=64)  # keep it simple at first
        .multi_agent(
            # For new API, passing a set of policy IDs auto-creates default modules.
            policies={x for x in policies},
            policy_mapping_fn=lambda agent_id, *args, **kwargs: agent_id,
            policies_to_train=policies_to_train,
        )
        .rl_module(model_config={"fcnet_hiddens": [64, 64]})
        .training(
            lr=1e-4,
            gamma=0.99,
            train_batch_size=256,
            num_steps_sampled_before_learning_starts=5_000,
            target_network_update_freq=1000,
            # Can change for more advanced models
            double_q=True,
            dueling=True,
            # Episode because its new API, non priority because we don't have the facilities yeh
            replay_buffer_config={
                "type": "MultiAgentEpisodeReplayBuffer",
                "capacity": 100_000,
            },
            # n_step can stay at default 1. Try more advanced later
        )

        .debugging(log_level="ERROR")
    )
    return config


# Mainly stolen from TA's
def training(env, checkpoint_path, max_iterations=500):
    rllib_env = ParallelPettingZooEnv(pettingzoo.utils.conversions.aec_to_parallel(env))
    id_env = "knights_archers_zombies_v10"
    register_env(id_env, lambda cfg: rllib_env)

    # Reproducibility
    np.random.seed(42)
    torch.manual_seed(42)

    policies = [x for x in env.agents]
    policies_to_train = policies
    config = algo_config(id_env, policies, policies_to_train)

    algo = config.build()
    for i in range(max_iterations):
        result = algo.train()
        result.pop("config", None)

        if "env_runners" in result and "agent_episode_returns_mean" in result["env_runners"]:
            print(i, result["env_runners"]["agent_episode_returns_mean"])
            # Simple early-stop criterion (tune to taste)
            if "archer_0" in result["env_runners"]["agent_episode_returns_mean"]:
                if result["env_runners"]["agent_episode_returns_mean"]["archer_0"] > 5:
                    break

        if i % 5 == 0:
            save_result = algo.save(checkpoint_path)
            path_to_checkpoint = save_result.checkpoint.path
            print(
                "An Algorithm checkpoint has been created inside directory: "
                f"'{path_to_checkpoint}'."
            )


if __name__ == "__main__":
    num_agents = 1
    visual_observation = False

    env = create_environment(num_agents=num_agents, visual_observation=visual_observation)
    env = CustomWrapper(env)

    checkpoint_path = str(Path("results").resolve())
    training(env, checkpoint_path, max_iterations=500)
