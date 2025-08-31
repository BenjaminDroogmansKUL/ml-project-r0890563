import sys
from pathlib import Path

_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

"""
DQN training for PettingZoo KAZ with a sound multi-seed evaluation protocol.

What this adds vs. your version:
- Clear TRAIN vs EVAL env separation via env_config["mode"] so they can use different seed pools.
- RLlib evaluation workers configured to run K=20 episodes at each evaluation checkpoint with explore=False.
- GLOBAL_SEED parameter for running N=5 independent runs externally (one process per seed).
"""

from typing import Callable

import gymnasium
import pettingzoo
from gymnasium import spaces
from pettingzoo.utils import BaseWrapper
from pettingzoo.utils.env import AgentID, ObsType

import numpy as np
import torch

from ray.rllib.algorithms.dqn import DQNConfig
from ray.rllib.core.rl_module import MultiRLModule
from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv
from ray.tune.registry import register_env

from utils import create_environment
from sticky_wrapper import StickyActionWrapper
from seed_cycle_wrapper import SeedCycleWrapper
from survival_bonus_wrapper import SurvivalBonusAECWrapper
from angular_zombie_wrapper import AngularZombieWrapper
from flatten_wrapper import FlattenWrapper
from only_zombies_wrapper import OnlyZombiesWrapper

from run_logger import RunLogger

# Constants for evaluation

# Choose your N external run seeds (you will pass these in one-by-one as GLOBAL_SEED).
# Example you can use: [111, 222, 333, 444, 555]
GLOBAL_SEED = 111  # <-- change per run (N=5 runs total)

# TRAIN seeds (cycled by SeedCycleWrapper during training).
TRAIN_SEEDS = list(range(0, 1000))

# EVAL seeds (disjoint from training). K=20 episodes per checkpoint will be drawn from this list.
# We’ll let the eval env’s SeedCycleWrapper iterate through this list deterministically.
EVAL_SEEDS = list(range(10_000, 10_000 + 200))  # plenty to cover multiple eval checkpoints

# Evaluation cadence and size (K)
EVAL_INTERVAL_ITERS = 5
EVAL_EPISODES_PER_CHECKPOINT = 20  # K

class CustomPredictFunction(Callable):
    """
    Loads RLlib MultiRLModule checkpoint and selects greedy actions via argmax(Q).
    """

    def __init__(self, env):
        best_checkpoint = (Path("results") / "learner_group" / "learner" / "rl_module").resolve()
        self.modules = MultiRLModule.from_checkpoint(best_checkpoint)

    def __call__(self, observation, agent, *args, **kwargs):
        rl_module = self.modules[agent]
        fwd_ins = {"obs": torch.tensor(observation, dtype=torch.float32).unsqueeze(0)}
        fwd_out = rl_module.forward_inference(fwd_ins)
        q_values = fwd_out["q_values"]  # shape: [B, num_actions]
        action = torch.argmax(q_values, dim=1)[0].item()
        return action



def make_env(
    mode: str,
    num_agents: int = 1,
    visual_observation: bool = False,
    max_zombies: int = 4,
    filter_zombies: bool = False,
    angular_features: bool = False,
    survival_bonus: bool = False,
    p_sticky: float = 0.25,
    survival_reward: float = 0.02,
):
    w = create_environment(
        num_agents=num_agents,
        visual_observation=visual_observation,
        max_zombies=max_zombies,
    )


    w = StickyActionWrapper(w, p_sticky=p_sticky)

    if filter_zombies and max_zombies > 0:
        if angular_features:
            w = AngularZombieWrapper(w, k=max_zombies)
        else:
            w = OnlyZombiesWrapper(w, k=max_zombies)
    else:
        w = FlattenWrapper(w)

    if survival_bonus:
        w = SurvivalBonusAECWrapper(w, archer_id="archer_0", survival_reward=survival_reward)

    if mode == "train":
        w = SeedCycleWrapper(w, seed_list=TRAIN_SEEDS)
    elif mode == "eval":
        w = SeedCycleWrapper(w, seed_list=EVAL_SEEDS)
    else:
        raise ValueError(f"Unknown env mode: {mode}")

    from pettingzoo.utils.conversions import aec_to_parallel
    from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv
    return ParallelPettingZooEnv(aec_to_parallel(w))



def algo_config(env_id: str, policies, policies_to_train):
    """
    Minimal baseline DQN (uniform replay; Double/Dueling off).
    """
    cfg = (
        DQNConfig()
        .api_stack(
            enable_rl_module_and_learner=True,
            enable_env_runner_and_connector_v2=True,
        )
        .environment(env=env_id, disable_env_checking=True, env_config={"mode": "train"})
        .env_runners(
            num_env_runners=1,
            rollout_fragment_length=64  # simple at first
        )
        .multi_agent(
            policies={x for x in policies},
            policy_mapping_fn=lambda agent_id, *args, **kwargs: agent_id,
            policies_to_train=policies_to_train,
        )
        .rl_module(model_config={"fcnet_hiddens": [256, 256]})
        .training(
            lr=1e-4,
            gamma=0.99,
            train_batch_size=256,
            num_steps_sampled_before_learning_starts=5_000,
            target_network_update_freq=1000,
            double_q=False,
            dueling=False,
            replay_buffer_config={
                "_enable_replay_buffer_api": True,
                "type": "MultiAgentEpisodeReplayBuffer",
                "capacity": 100_000,  #  original capacity
                "replay_sequence_length": 1,  # nonrecurrent DQN
            },
        )
        # Built-in evaluation
        .evaluation(
            evaluation_parallel_to_training=True,
            evaluation_interval=EVAL_INTERVAL_ITERS,
            evaluation_duration=EVAL_EPISODES_PER_CHECKPOINT,
            evaluation_duration_unit="episodes",
            evaluation_num_workers=1,
            evaluation_num_env_runners=1,
            evaluation_config={
                "explore": False,                 # deterministic evaluation
                "env_config": {"mode": "eval"},   # use EVAL seed pool
            },
        )
        .debugging(log_level="ERROR")
    )

    # Ensure framework
    cfg = cfg.framework("torch")

    return cfg

def training(checkpoint_path: str, max_iterations: int = 500):
    env_id = "knights_archers_zombies_v10"
    register_env(env_id,
                 lambda cfg: make_env(mode=cfg.get("mode", "train"),
                                      num_agents=1,
                                      visual_observation=False,
                                      max_zombies=4,
                                      angular_features=True,
                                      survival_bonus=False,
                                      survival_reward=0,
                                      p_sticky=0.25))

    # Global seed
    np.random.seed(GLOBAL_SEED)
    torch.manual_seed(GLOBAL_SEED)

    tmp_train_env = make_env(mode="train")
    policies = [x for x in tmp_train_env.agents]
    policies_to_train = policies

    config = algo_config(env_id, policies, policies_to_train)
    algo = config.build()

    # Run logger
    meta = {
        "train_seeds_span": [TRAIN_SEEDS[0], TRAIN_SEEDS[-1]],
        "eval_seed_pool_size": len(EVAL_SEEDS),
        "eval_episodes_per_checkpoint": EVAL_EPISODES_PER_CHECKPOINT,
        "eval_interval_iters": EVAL_INTERVAL_ITERS,
        "sticky_action_p": 0.25,
        "model": "MLP(256,256)",
        "double_q": False,
        "dueling": False,
        "distributional": False,
        "n_step": 1,
        "replay": "uniform",
        "replay_capacity": 100_000,
        "lr": 1e-4,
        "gamma": 0.99,
        "target_network_update_freq": 1000,
    }
    logger = RunLogger(
        base_dir=str(Path("results").resolve()),
        global_seed=GLOBAL_SEED,
        algo_name="DQN_BASE",
        meta=meta,
    )

    for i in range(max_iterations):
        result = algo.train()
        result.pop("config", None)

        logger.log_train(i, result)
        train_metrics = result.get("env_runners", {}).get("agent_episode_returns_mean", {})
        if train_metrics:
            print(f"[seed={GLOBAL_SEED}] iter={i} train_returns={train_metrics}")

        eval_sec = result.get("evaluation") or {}
        er = eval_sec.get("env_runners") or {}
        k_episodes = int(er.get("num_episodes", 0) or 0)
        if k_episodes > 0:
            mean_archer = (er.get("agent_episode_returns_mean") or {}).get("archer_0")
            print(f"[seed={GLOBAL_SEED}] iter={i} EVAL: archer_0_mean={mean_archer} over {k_episodes} episodes")
            logger.log_eval(i, result)

        if i % 5 == 0:
            save_result = algo.save(checkpoint_path)
            ckpt_path = save_result.checkpoint.path
            logger.log_checkpoint(i, result, ckpt_path)
            print(f"[seed={GLOBAL_SEED}] Checkpoint saved: '{ckpt_path}'")

    print(f"Run artifacts in: {logger.run_dir}")

if __name__ == "__main__":
    checkpoint_path = str(Path("results").resolve())
    training(checkpoint_path, max_iterations=500)
