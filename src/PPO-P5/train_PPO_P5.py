#!/usr/bin/env python3
# encoding: utf-8

from pathlib import Path
from typing import Callable, Dict, Any, List, Optional
import os, csv, json, time, datetime

import gymnasium
import pettingzoo
from gymnasium import spaces
from pettingzoo.utils import BaseWrapper
from pettingzoo.utils.env import AgentID, ObsType

import numpy as np
import torch

from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.core.rl_module import MultiRLModule
from ray.rllib.core.rl_module.default_model_config import DefaultModelConfig
from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv
from ray.tune.registry import register_env

from utils import create_environment
from sticky_wrapper import StickyActionWrapper
from seed_cycle_wrapper import SeedCycleWrapper  # <-- same seed-cycler as in DQN

# ---------------------------
# Run logger (same as DQN)
# ---------------------------

def _extract_env_steps(result: Dict[str, Any]) -> Optional[int]:
    return (
        result.get("env_steps_sampled")
        or result.get("num_env_steps_sampled")
        or result.get("timesteps_total")
        or result.get("counters", {}).get("env_steps_sampled")
        or result.get("env_runners", {}).get("env_steps_sampled")
    )

def _sf(x):
    # safe float for csv (handles np types + None)
    try:
        return "" if x is None else float(x)
    except Exception:
        return ""


class RunLogger:
    def __init__(self, base_dir: str, global_seed: int, algo_name: str, meta: Dict[str, Any]):
        ts = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.run_id = f"{algo_name.lower()}_kaz_{ts}_seed{global_seed}"
        self.dir = os.path.join(base_dir, "metrics", self.run_id)
        os.makedirs(self.dir, exist_ok=True)

        meta_out = {"run_id": self.run_id, "global_seed": global_seed, "algo": algo_name}
        meta_out.update(meta)
        with open(os.path.join(self.dir, "run_meta.json"), "w") as f:
            json.dump(meta_out, f, indent=2)

        self._train_csv = os.path.join(self.dir, "train_log.csv")
        with open(self._train_csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow([
                "run_id", "iter",
                "time_this_iter_s", "time_total_s",
                "delta_env_steps", "cum_env_steps", "episodes_this_iter", "throughput_steps_per_s",
                "train_episode_return_mean", "train_episode_return_min", "train_episode_return_max",
                "train_episode_len_mean", "train_episode_len_min", "train_episode_len_max",
                "train_return_archer_0", "train_return_mean_all_agents",
                # PPO-specific learner stats
                "total_loss", "policy_loss", "vf_loss", "entropy",
                "mean_kl", "curr_kl_coeff", "curr_entropy_coeff", "vf_explained_var",
                "grad_norm", "lr", "module_train_batch_size_mean",
                "num_trainable_parameters", "num_module_steps_trained", "num_module_steps_trained_lifetime",
                "weights_seq_no",
                # System perf
                "cpu_util_percent", "ram_util_percent"
            ])

        self._eval_csv = os.path.join(self.dir, "eval_log.csv")
        with open(self._eval_csv, "w", newline="") as f:
            csv.writer(f).writerow([
                "run_id", "iter",
                "num_episodes",
                "return_mean", "return_min", "return_max",
                "len_mean", "len_min", "len_max",
                "agent_means_json",
                "episode_duration_sec_mean",
                "env_steps", "env_steps_lifetime",
                "sample_time_s", "env_steps_per_s",
                "weights_seq_no"
            ])


        self._ckpt_csv = os.path.join(self.dir, "checkpoints.csv")
        with open(self._ckpt_csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["run_id", "iter", "checkpoint_path", "cum_env_steps"])

        self._t0 = time.time()
        self._last_cum_env_steps = 0

    @staticmethod
    def _safe_float(x: Any) -> Any:
        try:
            return float(x)
        except Exception:
            return ""

    @staticmethod
    def _get(env: Dict[str, Any], path: List[str], default=None):
        cur = env
        for k in path:
            if not isinstance(cur, dict) or k not in cur:
                return default
            cur = cur[k]
        return cur

    def _extract_cum_env_steps(self, result: Dict[str, Any]) -> Optional[int]:
        for path in [
            ["num_env_steps_sampled_lifetime"],
            ["env_runners", "num_env_steps_sampled_lifetime"],
            ["timesteps_total"],
            ["counters", "env_steps_sampled"],
        ]:
            v = self._get(result, path)
            if isinstance(v, (int, float)) and v is not None:
                return int(v)
        return None

    def _extract_delta_env_steps(self, result: Dict[str, Any]) -> Optional[int]:
        v = self._get(result, ["env_runners", "num_env_steps_sampled"])
        if isinstance(v, (int, float)):
            return int(v)
        cum = self._extract_cum_env_steps(result)
        if cum is not None and isinstance(self._last_cum_env_steps, (int, float)):
            delta = int(cum - (self._last_cum_env_steps or 0))
            return max(delta, 0)
        return None

    @staticmethod
    def _mean_of_numeric_dict_values(d: Dict[str, Any]) -> Optional[float]:
        if not isinstance(d, dict) or not d:
            return None
        vals = [float(v) for v in d.values() if isinstance(v, (int, float, np.floating))]
        return float(np.mean(vals)) if vals else None

    @staticmethod
    def _first_learner_key(learners: Dict[str, Any]) -> Optional[str]:
        if not isinstance(learners, dict):
            return None
        for k in learners.keys():
            if k != "__all_modules__":
                return k
        return None

    @staticmethod
    def _extract_eval_seeds(eval_sec: Dict[str, Any]) -> List[Any]:
        hist = eval_sec.get("hist_stats", {}) or {}
        for key, val in hist.items():
            if isinstance(key, str) and "seed" in key.lower() and isinstance(val, list):
                return val
        return []

    def log_train(self, iteration: int, result: Dict[str, Any]):
        time_this_iter_s = self._safe_float(result.get("time_this_iter_s"))
        time_total_s = self._safe_float(result.get("time_total_s"))

        cum_env_steps = self._extract_cum_env_steps(result)
        delta_env_steps = self._extract_delta_env_steps(result)
        episodes_this_iter = self._get(result, ["env_runners", "num_episodes"])

        throughput = ""
        if isinstance(delta_env_steps, (int, float)) and isinstance(time_this_iter_s, float) and time_this_iter_s > 0:
            throughput = float(delta_env_steps) / time_this_iter_s

        er = result.get("env_runners", {}) or {}
        train_team_mean = er.get("episode_return_mean", "")
        train_team_min = er.get("episode_return_min", "")
        train_team_max = er.get("episode_return_max", "")
        train_len_mean = er.get("episode_len_mean", "")
        train_len_min = er.get("episode_len_min", "")
        train_len_max = er.get("episode_len_max", "")
        agent_means = er.get("agent_episode_returns_mean", {}) or {}
        train_archer = agent_means.get("archer_0", "")
        train_agents_overall = self._mean_of_numeric_dict_values(agent_means)

        learners = result.get("learners", {}) or {}
        lm_key = self._first_learner_key(learners)
        lm = learners.get(lm_key, {}) if lm_key else {}

        total_loss = lm.get("total_loss", "")
        policy_loss = lm.get("policy_loss", "")
        vf_loss = lm.get("vf_loss", "")
        entropy = lm.get("entropy", "")
        mean_kl = lm.get("mean_kl_loss", "")
        curr_kl_coeff = lm.get("curr_kl_coeff", "")
        curr_entropy_coeff = lm.get("curr_entropy_coeff", "")
        vf_explained_var = lm.get("vf_explained_var", "")
        grad_norm = lm.get("gradients_default_optimizer_global_norm", "")
        lr = lm.get("default_optimizer_learning_rate", "")
        module_train_batch_size_mean = lm.get("module_train_batch_size_mean", "")
        num_trainable_parameters = lm.get("num_trainable_parameters", "")
        num_module_steps_trained = lm.get("num_module_steps_trained", "")
        num_module_steps_trained_lifetime = lm.get("num_module_steps_trained_lifetime", "")
        weights_seq_no = lm.get("weights_seq_no", "")

        perf = result.get("perf", {}) or {}
        cpu = perf.get("cpu_util_percent", "")
        ram = perf.get("ram_util_percent", "")

        with open(self._train_csv, "a", newline="") as f:
            csv.writer(f).writerow([
                self.run_id, iteration,
                time_this_iter_s, time_total_s,
                delta_env_steps if delta_env_steps is not None else "",
                cum_env_steps if cum_env_steps is not None else "",
                episodes_this_iter if episodes_this_iter is not None else "",
                throughput if throughput != "" else "",
                train_team_mean, train_team_min, train_team_max,
                train_len_mean, train_len_min, train_len_max,
                train_archer,
                train_agents_overall if train_agents_overall is not None else "",
                total_loss, policy_loss, vf_loss, entropy,
                mean_kl, curr_kl_coeff, curr_entropy_coeff, vf_explained_var,
                grad_norm, lr, module_train_batch_size_mean,
                num_trainable_parameters, num_module_steps_trained, num_module_steps_trained_lifetime,
                weights_seq_no,
                self._safe_float(cpu), self._safe_float(ram),
            ])

        if isinstance(cum_env_steps, int):
            self._last_cum_env_steps = cum_env_steps

    def log_eval(self, iteration: int, result: dict):
        import json
        er = (result.get("evaluation") or {}).get("env_runners") or {}

        if not er and "env_runners" in result and not result.get("evaluation"):
            er = result["env_runners"]

        num_episodes = er.get("num_episodes")

        return_mean = er.get("episode_return_mean")
        return_min = er.get("episode_return_min")
        return_max = er.get("episode_return_max")

        agent_means = er.get("agent_episode_returns_mean") \
                      or er.get("module_episode_returns_mean") \
                      or {}

        if return_mean is None and isinstance(agent_means, dict) and agent_means:
            # average across agents if aggregate missing
            try:
                nums = [float(v) for v in agent_means.values()]
                return_mean = sum(nums) / len(nums)
            except Exception:
                return_mean = ""

        len_mean = er.get("episode_len_mean")
        len_min = er.get("episode_len_min")
        len_max = er.get("episode_len_max")

        # throughput
        episode_duration_sec_mean = er.get("episode_duration_sec_mean")
        env_steps = er.get("num_env_steps_sampled")
        env_steps_lifetime = er.get("num_env_steps_sampled_lifetime")
        sample_time_s = er.get("sample")  # seconds spent sampling eval
        env_steps_per_s = er.get("num_env_steps_sampled_per_second")
        if env_steps_per_s is None:
            try:
                if sample_time_s and sample_time_s > 0 and env_steps is not None:
                    env_steps_per_s = float(env_steps) / float(sample_time_s)
            except Exception:
                env_steps_per_s = ""

        weights_seq_no = er.get("weights_seq_no")

        agent_means_json = json.dumps(agent_means or {}, sort_keys=True)

        row = [
            self.run_id, iteration,
            num_episodes if num_episodes is not None else "",
            _sf(return_mean), _sf(return_min), _sf(return_max),
            _sf(len_mean), _sf(len_min), _sf(len_max),
            agent_means_json,
            _sf(episode_duration_sec_mean),
            env_steps if env_steps is not None else "",
            env_steps_lifetime if env_steps_lifetime is not None else "",
            _sf(sample_time_s), _sf(env_steps_per_s),
            _sf(weights_seq_no),
        ]
        with open(self._eval_csv, "a", newline="") as f:
            csv.writer(f).writerow(row)


    def log_checkpoint(self, iteration: int, result: Dict[str, Any], ckpt_path: str):
        cum_env_steps = self._extract_cum_env_steps(result)
        with open(self._ckpt_csv, "a", newline="") as f:
            csv.writer(f).writerow([
                self.run_id, iteration, ckpt_path,
                cum_env_steps if cum_env_steps is not None else ""
            ])

    @property
    def run_dir(self) -> str:
        return self.dir


GLOBAL_SEED = 111                     # change per external run (do N=5 runs)
TRAIN_SEEDS = list(range(0, 1000))    # training seed cycle
EVAL_SEEDS = list(range(10_000, 10_000 + 200))  # disjoint eval pool

EVAL_INTERVAL_ITERS = 5
EVAL_EPISODES_PER_CHECKPOINT = 20

# ---------------------------
# Wrappers and env factory (same as DQN)
# ---------------------------

class CustomWrapper(BaseWrapper):
    def observation_space(self, agent: AgentID) -> gymnasium.spaces.Space:
        return spaces.flatten_space(super().observation_space(agent))

    def observe(self, agent: AgentID) -> ObsType | None:
        obs = super().observe(agent)
        return obs.flatten()

def make_env(mode: str, num_agents: int = 1, visual_observation: bool = False):
    base = create_environment(num_agents=num_agents, visual_observation=visual_observation)
    base = StickyActionWrapper(base, p_sticky=0.25)
    base = CustomWrapper(base)

    if mode == "train":
        base = SeedCycleWrapper(base, seed_list=TRAIN_SEEDS)
    elif mode == "eval":
        base = SeedCycleWrapper(base, seed_list=EVAL_SEEDS)
    else:
        raise ValueError(f"Unknown env mode: {mode}")

    return ParallelPettingZooEnv(pettingzoo.utils.conversions.aec_to_parallel(base))

# ---------------------------
# Deterministic deploy for PPO (argmax over logits)
# ---------------------------

class CustomPredictFunction(Callable):
    def __init__(self, env):
        best_checkpoint = (Path("results") / "learner_group" / "learner" / "rl_module").resolve()
        self.modules = MultiRLModule.from_checkpoint(best_checkpoint)

    def __call__(self, observation, agent, *args, **kwargs):
        rl_module = self.modules[agent]
        fwd_ins = {"obs": torch.tensor(observation, dtype=torch.float32).unsqueeze(0)}
        fwd_out = rl_module.forward_inference(fwd_ins)
        logits = fwd_out["action_dist_inputs"]  # categorical logits for discrete action space
        # Deterministic action = mode of the categorical distribution
        action = torch.argmax(logits, dim=-1)[0].item()
        return action


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
            rollout_fragment_length=128,
            observation_filter="MeanStdFilter",
            use_worker_filter_stats=True,
            update_worker_filter_stats=True,
        )
        .multi_agent(
            policies={x for x in policies},
            policy_mapping_fn=lambda agent_id, *args, **kwargs: agent_id,
            policies_to_train=policies_to_train,
        )
        .rl_module(
            model_config=DefaultModelConfig(
                fcnet_hiddens=[64, 64],
                use_lstm=True,
                lstm_cell_size=128,
                lstm_use_prev_action=True,
                lstm_use_prev_reward=True,
                max_seq_len=32,  # train on fixed sequences of 32
            )
        )
        .training(
            train_batch_size=2048, #bigger then ¨4
            lr=1e-4,
            gamma=0.99,

            # Improvement over BASE
            entropy_coeff=[[0, 0.01], [5_000_000, 0.001]],

            # improvement over P1
            use_kl_loss=True,
            kl_coeff=0.2,
            kl_target=0.01,

            grad_clip=0.5,  # improvement over p2

            clip_param=0.2,
            vf_clip_param=10.0,
            vf_loss_coeff=1.0,


            use_critic=True,
            use_gae=True,
            lambda_=0.98, #improvement because KL control and grad clip P4
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
                "observation_filter": "MeanStdFilter",
                "use_worker_filter_stats": False,
                "update_worker_filter_stats": False,
            },
        )
        .debugging(log_level="ERROR")
    )

    return cfg.framework("torch")


def training(checkpoint_path: str, max_iterations: int = 500):
    env_id = "knights_archers_zombies_v10"
    register_env(env_id, lambda cfg: make_env(mode=cfg.get("mode", "train")))

    np.random.seed(GLOBAL_SEED)
    torch.manual_seed(GLOBAL_SEED)

    tmp_train_env = make_env(mode="train")
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
        "p4_flags": {
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

        #if train_metrics and "archer_0" in train_metrics:
        #    if train_metrics["archer_0"] > 5:
        #        print(f"[seed={GLOBAL_SEED}] Early stop at iter={i} (archer_0 > 5).")
        #        break

        if i % 5 == 0:
            save_result = algo.save(checkpoint_path)
            ckpt_path = save_result.checkpoint.path
            logger.log_checkpoint(i, result, ckpt_path)
            print(f"[seed={GLOBAL_SEED}] Checkpoint saved: '{ckpt_path}'")

    print(f"Run artifacts in: {logger.run_dir}")

# ---------------------------
# Main
# ---------------------------

if __name__ == "__main__":
    checkpoint_path = str(Path("results").resolve())
    training(checkpoint_path, max_iterations=500)
