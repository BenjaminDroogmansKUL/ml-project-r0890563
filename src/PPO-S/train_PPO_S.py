from pathlib import Path
import numpy as np
import torch
from ray.rllib.algorithms.ppo import PPOConfig
from ray.tune.registry import register_env
from make_env import make_env
from run_logger import RunLogger


GLOBAL_SEED = 111
TRAIN_SEEDS = list(range(0, 1000))
EVAL_SEEDS = list(range(10_000, 10_000 + 200))

EVAL_INTERVAL_ITERS = 5
EVAL_EPISODES_PER_CHECKPOINT = 20


def algo_config(env_id: str, policies, policies_to_train):
    cfg = (
        PPOConfig()
        .api_stack(
            enable_rl_module_and_learner=True,
            enable_env_runner_and_connector_v2=True,
        )
        .environment(env=env_id, disable_env_checking=True, env_config={"mode": "train"})
        .env_runners(
            num_env_runners=1,
            rollout_fragment_length=128
        )
        .multi_agent(
            policies={x for x in policies},
            policy_mapping_fn=lambda agent_id, *args, **kwargs: agent_id,
            policies_to_train=policies_to_train,
        )
        .rl_module(model_config={"fcnet_hiddens": [64, 64]})
        .training(
            train_batch_size=512,
            lr=1e-4,
            gamma=0.99,

            entropy_coeff=0.01,
            use_kl_loss=True,
            kl_coeff=0.2,
            kl_target=0.01,
            grad_clip=0.5,

            clip_param=0.2,
            vf_clip_param=10.0,
            vf_loss_coeff=1.0,

            use_critic=True,
            use_gae=True,
            lambda_=0.98,
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


def training(checkpoint_path: str, max_iterations: int = 20000):
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

    np.random.seed(GLOBAL_SEED)
    torch.manual_seed(GLOBAL_SEED)

    # Must match the register_env parameters
    tmp_train_env = make_env(
        mode="train",
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
        )
    policies = [x for x in tmp_train_env.agents]
    policies_to_train = policies

    config = algo_config(env_id, policies, policies_to_train)
    algo = config.build()

    meta = {
        "train_seeds_span": [TRAIN_SEEDS[0], TRAIN_SEEDS[-1]],
        "eval_seed_pool_size": len(EVAL_SEEDS),
        "eval_episodes_per_checkpoint": EVAL_EPISODES_PER_CHECKPOINT,
        "eval_interval_iters": EVAL_INTERVAL_ITERS,
        "sticky_action_p": 0.25,
        "model": "MLP(64,64)",
        "lr": 1e-4,
        "gamma": 0.99,
        "S_flags": {
            "entropy_coeff": 0.01,
            "use_kl_loss": True,
            "kl_target": 0.01,
            "kl_coeff": 0.2,
            "grad_clip": 0.5,
            "clip_param": 0.2,
            "vf_clip_param": 10.0,
            "vf_loss_coeff": 1.0,
            "use_critic": True,
            "use_gae": True,
            "gae_lambda": 0.98,
        },
    }
    logger = RunLogger(
        base_dir=str(Path("results").resolve()),
        global_seed=GLOBAL_SEED,
        algo_name="PPO",
        meta=meta,
    )

    for i in range(max_iterations):
        result = algo.train()
        result.pop("config", None)
        print(result)

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
    training(checkpoint_path, max_iterations=10000)
