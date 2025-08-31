from pathlib import Path

import numpy as np
import torch

from ray.rllib.algorithms.dqn import DQNConfig
from ray.tune.registry import register_env
from run_logger import RunLogger
from make_env import make_env

GLOBAL_SEED = 444
TRAIN_SEEDS = list(range(0, 1000))
EVAL_SEEDS  = list(range(10_000, 10_000 + 200))
EVAL_INTERVAL_ITERS = 5
EVAL_EPISODES_PER_CHECKPOINT = 20

N_STEP = 3
PER_ALPHA = 0.6
PER_BETA  = 0.4
PER_EPS   = 1e-6

def algo_config(env_id: str, policies, policies_to_train):
    cfg = (
        DQNConfig()
        .api_stack(
            enable_rl_module_and_learner=True,
            enable_env_runner_and_connector_v2=True,
        )
        .environment(env=env_id, disable_env_checking=True, env_config={"mode": "train"})
        .env_runners(
            num_env_runners=1,
            rollout_fragment_length=64
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

            # Improvements
            double_q=True,                # DDQN
            dueling=False,
            n_step=N_STEP,                # nstep

            # PER
            replay_buffer_config={
                "_enable_replay_buffer_api": True,
                "type": "MultiAgentPrioritizedEpisodeReplayBuffer",
                "capacity": 100_000,
                "replay_sequence_length": 1,
                "alpha": PER_ALPHA,
                "beta": PER_BETA,
                "eps": PER_EPS,
            },

        )
        .evaluation(
            evaluation_parallel_to_training=True,
            evaluation_interval=EVAL_INTERVAL_ITERS,
            evaluation_duration=EVAL_EPISODES_PER_CHECKPOINT,
            evaluation_duration_unit="episodes",
            evaluation_num_workers=1,
            evaluation_num_env_runners=1,
            evaluation_config={
                "explore": False,
                "env_config": {"mode": "eval"},
            },
        )
        .debugging(log_level="ERROR")
    )
    return cfg.framework("torch")

def training(checkpoint_path: str, max_iterations: int = 500):
    env_id = "knights_archers_zombies_v10"

    register_env(
        env_id,
        lambda cfg: make_env(
            mode=cfg.get("mode", "train"),
            num_agents=1,
            visual_observation=False,
            max_zombies=4,
            filter_zombies=True,
            angular_features=True,
            survival_bonus=False,
            survival_reward=0,
            p_sticky=0.25,
            train_seeds=TRAIN_SEEDS,
            eval_seeds=EVAL_SEEDS,
        ),
    )

    # Repro
    np.random.seed(GLOBAL_SEED)
    torch.manual_seed(GLOBAL_SEED)

    tmp_train_env = make_env(
        mode="train",
        num_agents=1,
        visual_observation=False,
        max_zombies=4,
        filter_zombies=False,
        angular_features=True,
        survival_bonus=False,
        survival_reward=0,
        p_sticky=0.25,
        train_seeds=TRAIN_SEEDS,
        eval_seeds=EVAL_SEEDS,
    )
    policies = list(tmp_train_env.agents)
    policies_to_train = policies

    config = algo_config(env_id, policies, policies_to_train)
    algo = config.build()

    meta = {
        "train_seeds_span": [TRAIN_SEEDS[0], TRAIN_SEEDS[-1]],
        "eval_seed_pool_size": len(EVAL_SEEDS),
        "eval_episodes_per_checkpoint": EVAL_EPISODES_PER_CHECKPOINT,
        "eval_interval_iters": EVAL_INTERVAL_ITERS,
        "sticky_action_p": 0.25,
        "model": "MLP(256,256)",
        "double_q": True,
        "dueling": False,
        "distributional": False,
        "n_step": N_STEP,
        "replay": "prioritized(episodic)",
        "replay_capacity": 100_000,
        "per_alpha": PER_ALPHA,
        "per_beta": PER_BETA,
        "per_eps": PER_EPS,
        "lr": 1e-4,
        "gamma": 0.99,
        "target_network_update_freq": 1000,
    }
    logger = RunLogger(
        base_dir=str(Path("results").resolve()),
        global_seed=GLOBAL_SEED,
        algo_name="DQN_D3",
        meta=meta,
    )

    for i in range(max_iterations):
        result = algo.train()
        result.pop("config", None)

        logger.log_train(i, result)

        eval_sec = result.get("evaluation") or {}
        er = eval_sec.get("env_runners") or {}
        if int(er.get("num_episodes", 0) or 0) > 0:
            logger.log_eval(i, result)

        if i % 5 == 0:
            ckpt = algo.save(checkpoint_path).checkpoint.path
            logger.log_checkpoint(i, result, ckpt)

    print(f"Run artifacts in: {logger.run_dir}")

if __name__ == "__main__":
    checkpoint_path = str(Path("results").resolve())
    training(checkpoint_path, max_iterations=5000)
