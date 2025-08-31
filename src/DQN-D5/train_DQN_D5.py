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

import supersuit as ss


import csv, json, os, time, datetime
from typing import Dict, Any, List, Optional

def _extract_env_steps(result: Dict[str, Any]) -> Optional[int]:
    # Try common fields across RLlib new/old summaries
    return (
        result.get("env_steps_sampled")
        or result.get("num_env_steps_sampled")
        or result.get("timesteps_total")
        or result.get("counters", {}).get("env_steps_sampled")
        or result.get("env_runners", {}).get("env_steps_sampled")
    )


class RunLogger:
    def __init__(self, base_dir: str, global_seed: int, algo_name: str, meta: Dict[str, Any]):
        ts = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.run_id = f"{algo_name.lower()}_kaz_{ts}_seed{global_seed}"
        self.dir = os.path.join(base_dir, "metrics", self.run_id)
        os.makedirs(self.dir, exist_ok=True)

        # meta file
        meta_out = {"run_id": self.run_id, "global_seed": global_seed, "algo": algo_name}
        meta_out.update(meta)
        with open(os.path.join(self.dir, "run_meta.json"), "w") as f:
            json.dump(meta_out, f, indent=2)

        # csv headers (iteration-first)
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
                "total_loss", "td_error_mean", "qf_mean", "qf_min", "qf_max",
                "num_target_updates", "last_target_update_ts",
                "grad_norm", "lr", "module_train_batch_size_mean",
                "rb_env_steps_added", "rb_env_steps_sampled", "rb_env_step_utilization",
                "rb_env_steps_stored", "rb_episodes_stored", "rb_evicted_steps_lifetime",
                "cpu_util_percent", "ram_util_percent"
            ])

        self._eval_csv = os.path.join(self.dir, "eval_log.csv")
        with open(self._eval_csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow([
                "run_id", "iter",
                "eval_k_episodes",
                "eval_return_mean", "eval_return_min", "eval_return_max", "eval_return_std",
                "eval_len_mean", "eval_len_min", "eval_len_max",
                "eval_return_archer_0",
                "episode_returns_json", "eval_seed_ids_json",
                "cum_env_steps", "delta_env_steps"
            ])

        self._ckpt_csv = os.path.join(self.dir, "checkpoints.csv")
        with open(self._ckpt_csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["run_id", "iter", "checkpoint_path", "cum_env_steps"])

        self._t0 = time.time()
        self._last_cum_env_steps = 0  # for delta fallback if per-iter steps are absent

    # ---------- helpers ----------
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
        # Prefer explicit lifetime counters; fallback to older RLlib fields
        for path in [
            ["num_env_steps_sampled_lifetime"],
            ["env_runners", "num_env_steps_sampled_lifetime"],
            ["timesteps_total"],  # older field name
            ["counters", "env_steps_sampled"],  # very old fallback
        ]:
            v = self._get(result, path)
            if isinstance(v, (int, float)) and v is not None:
                return int(v)
        return None

    def _extract_delta_env_steps(self, result: Dict[str, Any]) -> Optional[int]:
        # Prefer per-iteration steps if available; else compute via lifetime diff
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
        # Try to find any hist_stats entry that looks like seeds (e.g., "episode_seed" or "seed")
        hist = eval_sec.get("hist_stats", {}) or {}
        for key, val in hist.items():
            if isinstance(key, str) and "seed" in key.lower() and isinstance(val, list):
                return val
        return []

    def log_train(self, iteration: int, result: Dict[str, Any]):
        # Timing
        time_this_iter_s = self._safe_float(result.get("time_this_iter_s"))
        time_total_s = self._safe_float(result.get("time_total_s"))

        # Steps & episodes
        cum_env_steps = self._extract_cum_env_steps(result)
        delta_env_steps = self._extract_delta_env_steps(result)
        episodes_this_iter = self._get(result, ["env_runners", "num_episodes"])
        throughput = ""
        if isinstance(delta_env_steps, (int, float)) and isinstance(time_this_iter_s, float) and time_this_iter_s > 0:
            throughput = float(delta_env_steps) / time_this_iter_s

        # Train returns/lengths
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

        # Learner metrics (pick first actual module, plus some __all_modules__ fields)
        learners = result.get("learners", {}) or {}
        lm_key = self._first_learner_key(learners)
        lm = learners.get(lm_key, {}) if lm_key else {}
        total_loss = lm.get("total_loss", "")
        td_error_mean = lm.get("td_error_mean", "")
        qf_mean = lm.get("qf_mean", "")
        qf_min = lm.get("qf_min", "")
        qf_max = lm.get("qf_max", "")
        num_target_updates = lm.get("num_target_updates", "")
        last_target_update_ts = lm.get("last_target_update_ts", "")
        grad_norm = lm.get("gradients_default_optimizer_global_norm", "")
        lr = lm.get("default_optimizer_learning_rate", "")
        module_train_batch_size_mean = lm.get("module_train_batch_size_mean", "")
        # Some totals under __all_modules__ (keep as-is if you want; not critical to CSV)

        # Replay buffer metrics
        rb = result.get("replay_buffer", {}) or {}
        rb_env_steps_added = rb.get("num_env_steps_added", "")
        rb_env_steps_sampled = rb.get("num_env_steps_sampled", "")
        rb_util = rb.get("env_step_utilization", "")
        rb_env_steps_stored = rb.get("num_env_steps_stored", "")
        rb_episodes_stored = rb.get("num_episodes_stored", "")
        rb_evicted_steps_lifetime = rb.get("num_env_steps_evicted_lifetime", "")

        # System perf
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
                total_loss, td_error_mean, qf_mean, qf_min, qf_max,
                num_target_updates, last_target_update_ts,
                grad_norm, lr, module_train_batch_size_mean,
                rb_env_steps_added, rb_env_steps_sampled, rb_util,
                rb_env_steps_stored, rb_episodes_stored, rb_evicted_steps_lifetime,
                self._safe_float(cpu), self._safe_float(ram),
            ])

        # Update last cumulative steps for delta fallback next iter
        if isinstance(cum_env_steps, int):
            self._last_cum_env_steps = cum_env_steps

    def log_eval(self, iteration: int, result: Dict[str, Any]):
        # Pull the evaluation section (present only when an eval was run this iter)
        eval_sec = result.get("evaluation") or {}
        er = eval_sec.get("env_runners") or {}
        k = int(er.get("num_episodes", 0) or 0)
        if k <= 0:
            return

        # Team returns and episode length stats (from RLlib's eval env_runners section)
        eval_return_mean = er.get("episode_return_mean", "")
        eval_return_min  = er.get("episode_return_min", "")
        eval_return_max  = er.get("episode_return_max", "")
        eval_len_mean    = er.get("episode_len_mean", "")
        eval_len_min     = er.get("episode_len_min", "")
        eval_len_max     = er.get("episode_len_max", "")

        # Agent-specific return (example: archer_0)
        agent_means = er.get("agent_episode_returns_mean", {}) or {}
        eval_return_archer_0 = agent_means.get("archer_0", "")

        # Hist stats: per-episode returns and (optionally) seed IDs; compute std from the list
        hist = eval_sec.get("hist_stats", {}) or {}
        ep_returns = hist.get("episode_reward") or hist.get("episode_return") or []
        if isinstance(ep_returns, list) and len(ep_returns) > 0:
            try:
                eval_return_std = float(np.std(ep_returns))
            except Exception:
                eval_return_std = ""
        else:
            eval_return_std = ""
        episode_returns_json = json.dumps(ep_returns if isinstance(ep_returns, list) else [])
        eval_seed_ids_json   = json.dumps(self._extract_eval_seeds(eval_sec))

        # Training env step counters for traceability (lifetime + per-iter delta)
        cum_env_steps   = self._extract_cum_env_steps(result)
        delta_env_steps = self._extract_delta_env_steps(result)

        # Append a row that matches the CSV header exactly.
        with open(self._eval_csv, "a", newline="") as f:
            csv.writer(f).writerow([
                self.run_id, iteration,
                k,
                eval_return_mean, eval_return_min, eval_return_max, eval_return_std,
                eval_len_mean, eval_len_min, eval_len_max,
                eval_return_archer_0,
                episode_returns_json, eval_seed_ids_json,
                cum_env_steps if cum_env_steps is not None else "",
                delta_env_steps if delta_env_steps is not None else "",
            ])

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

# Constants for evaluation

# Choose your N external run seeds (you will pass these in one-by-one as GLOBAL_SEED).
# Example you can use: [111, 222, 333, 444, 555]
GLOBAL_SEED = 222  # <-- change per run (N=5 runs total)

# TRAIN seeds (cycled by SeedCycleWrapper during training).
TRAIN_SEEDS = list(range(0, 1000))

# EVAL seeds (disjoint from training). K=20 episodes per checkpoint will be drawn from this list.
# We’ll let the eval env’s SeedCycleWrapper iterate through this list deterministically.
EVAL_SEEDS = list(range(10_000, 10_000 + 200))  # plenty to cover multiple eval checkpoints

# Evaluation cadence and size (K)
EVAL_INTERVAL_ITERS = 5
EVAL_EPISODES_PER_CHECKPOINT = 20  # K

# Improvment over base parameter:
D1_N_STEP = 3

#D3 over D2 impr by prioritized replayu
PER_ALPHA = 0.6
PER_BETA = 0.4
PER_EPS = 1e-6           #  epsilon added  priorities

# D4 over D3, distributional
D4_NUM_ATOMS = 51
D4_V_MIN = -20.0
D4_V_MAX =  20.0

# ---- FULL setup toggles ----
FULL_DUELING = True          # turn dueling streams ON
FULL_USE_LSTM = True         # set to False if you want a non-recurrent "full" ablation

# LSTM (only used if FULL_USE_LSTM=True)
FULL_LSTM_CELL_SIZE = 128    #drastically smaller
FULL_LSTM_USE_PREV_ACTION = True
FULL_LSTM_USE_PREV_REWARD = True
FULL_REPLAY_SEQUENCE_LENGTH = 40     # length of sequences sampled from the buffer
FULL_REPLAY_BURN_IN = 10             # prefix timesteps used to warm the hidden state

# (we reuse your existing D4 settings for num_atoms/v_min/v_max and D1’s n_step)


class AngularK4LastRowsKeepShapeAECWrapper(BaseWrapper):
    """
    Keep shape exactly 5x5 (flattened to 25):
      Row 0 (archer) unchanged: [x, y, ?, hx, hy]
      Rows 1..4 = last 4 rows of raw obs, treated as zombies:
          [dist, dx, dy, cosΔ, sinΔ]
    If a 'zombie' row is actually empty (all zeros), we keep zeros (no fake angles).
    """

    def observation_space(self, agent):
        base_space = super().observation_space(agent)  # assume Box(*, 5)
        low = float(np.min(base_space.low))
        high = float(np.max(base_space.high))
        return spaces.Box(low=low, high=high, shape=(25,), dtype=np.float32)

    @staticmethod
    def _unit(v):
        v = np.asarray(v, dtype=np.float32)
        n = np.linalg.norm(v)
        return v / n if n > 1e-8 else np.array([1.0, 0.0], dtype=np.float32)

    def observe(self, agent):
        raw = super().observe(agent)
        if raw is None:
            return None
        raw = np.asarray(raw, dtype=np.float32)

        # Expect matrix with 5 columns. If flat, reshape.
        if raw.ndim == 1:
            assert raw.size % 5 == 0, f"Unexpected flat obs size {raw.size}"
            raw = raw.reshape((-1, 5))

        # Row 0: archer unchanged
        archer = raw[0].copy()  # [x, y, ?, hx, hy]
        hx, hy = archer[3], archer[4]
        h = self._unit([hx, hy])

        # Take the **last 4 rows**, exactly like your original wrapper
        zsrc = raw[-4:].copy()  # shape (4, 5)

        # Build zombie rows: replace last two cols with [cosΔ, sinΔ]
        z_rows = []
        for i in range(4):
            dist, dx, dy = zsrc[i, 0], zsrc[i, 1], zsrc[i, 2]
            if dist == 0.0 and dx == 0.0 and dy == 0.0:
                # empty slot -> keep all zeros
                z_rows.append(np.zeros(5, dtype=np.float32))
            else:
                vhat = self._unit([dx, dy])
                cos_d = float(h @ vhat)
                sin_d = float(h[0]*vhat[1] - h[1]*vhat[0])
                z_rows.append(np.array([dist, dx, dy, cos_d, sin_d], dtype=np.float32))

        final = np.vstack([archer.reshape(1, 5)] + z_rows)  # (5,5)
        out = final.flatten().astype(np.float32, copy=False)

        # Safety checks (remove after first run)
        # assert out.shape[0] == 25, out.shape
        return out


class CustomWrapper(BaseWrapper):
    def observation_space(self, agent: AgentID) -> gymnasium.spaces.Space:
        # New obs is 5 rows x 5 cols = 25 long when flattened
        base_space = super().observation_space(agent)
        sub_space = spaces.Box(low=base_space.low.min(),
                               high=base_space.high.max(),
                               shape=(25,),
                               dtype=np.float32)
        return sub_space

    def observe(self, agent: AgentID) -> ObsType | None:
        obs = super().observe(agent)
        if obs is None:
            return None

        # Keep only first row (archer) + last 4 rows (zombies)
        obs = obs[[0, -4, -3, -2, -1], :]   # shape (5,5)

        flat_obs = obs.flatten().astype(np.float32, copy=False)  # shape (25,)
        return flat_obs


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



def make_env(mode: str, num_agents: int = 1, visual_observation: bool = False):
    base = create_environment(num_agents=num_agents, visual_observation=visual_observation)
    # (Optional but recommended) keep agent set constant for vectorized training:
    base = ss.black_death_v3(base)

    # Keep your sticky action stochasticity if you like
    base = StickyActionWrapper(base, p_sticky=0.25)

    # >>> Replace your CustomWrapper with the angular one <<<
    base = AngularK4LastRowsKeepShapeAECWrapper(base)

    # Seeds, etc.
    if mode == "train":
        base = SeedCycleWrapper(base, seed_list=TRAIN_SEEDS)
    elif mode == "eval":
        base = SeedCycleWrapper(base, seed_list=EVAL_SEEDS)
    else:
        raise ValueError(f"Unknown env mode: {mode}")

    # Convert AEC → parallel for RLlib
    from pettingzoo.utils.conversions import aec_to_parallel
    from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv
    return ParallelPettingZooEnv(aec_to_parallel(base))



def algo_config(env_id: str, policies, policies_to_train):
    """
    D3 DQN
    """

    model_cfg = {
        "fcnet_hiddens": [128],
    }

    if FULL_USE_LSTM:
        model_cfg.update({
            "use_lstm": True,
            "lstm_cell_size": FULL_LSTM_CELL_SIZE,
            "lstm_use_prev_action": FULL_LSTM_USE_PREV_ACTION,
            "lstm_use_prev_reward": FULL_LSTM_USE_PREV_REWARD,
            "max_seq_len": FULL_REPLAY_SEQUENCE_LENGTH,
        })

    cfg = (
        DQNConfig()
        .api_stack(
            enable_rl_module_and_learner=True,
            enable_env_runner_and_connector_v2=True,
        )
        .environment(env=env_id, disable_env_checking=True, env_config={"mode": "train"})
        .env_runners(
            num_env_runners=4,
            num_envs_per_env_runner=4,
            rollout_fragment_length=64  # simple at first
        )
        .multi_agent(
            policies={x for x in policies},
            policy_mapping_fn=lambda agent_id, *args, **kwargs: agent_id,
            policies_to_train=policies_to_train,
        )
        .rl_module(model_config={"fcnet_hiddens": [256,128,128]}) #try a big bigger
        .training(
            lr=1e-4,
            gamma=0.98,
            train_batch_size=1024, #massively increase
            num_steps_sampled_before_learning_starts=5_000,
            target_network_update_freq=1000,
            double_q=True, # Improvement over D1
            dueling=FULL_DUELING,
            n_step=D1_N_STEP,   # Improvement over BASE
            num_atoms=D4_NUM_ATOMS, # improvement over d3 for distr
            v_min=D4_V_MIN,
            v_max=D4_V_MAX,
            replay_buffer_config={
                "_enable_replay_buffer_api": True,
                "type": "MultiAgentPrioritizedEpisodeReplayBuffer", #Still needs to be episodic to work.
                "capacity": 100_000,  # same capacity as D2
                "replay_sequence_length": FULL_REPLAY_SEQUENCE_LENGTH if FULL_USE_LSTM else 1,  # non-recurrent DQN
                "alpha": PER_ALPHA,
                "beta": PER_BETA,
                "eps": PER_EPS,
            },
            grad_clip= 40.0,

            # n_step default = 1
            #epsilon=[[0, 1.0], [200_000, 0.01]],
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


# ---------------------------
# Training loop
# ---------------------------

def training(checkpoint_path: str, max_iterations: int = 500):
    env_id = "knights_archers_zombies_v10"
    register_env(env_id, lambda cfg: make_env(mode=cfg.get("mode", "train")))

    # Global seed (per-run)
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
        "model": "MLP(64,64)" + (" + LSTM" if FULL_USE_LSTM else ""),
        "double_q": True, # Improvement over D1
        "dueling": FULL_DUELING,
        "v_min": D4_V_MIN,
        "v_max": D4_V_MAX,
        "distributional": True,     #improvement over D3
        "num_atoms": D4_NUM_ATOMS,  #  D4 params
        "n_step": D1_N_STEP, # Improvementover BASE
        "use_lstm": FULL_USE_LSTM,
        "lstm_cell_size": FULL_LSTM_CELL_SIZE if FULL_USE_LSTM else None,
        "lstm_use_prev_action": FULL_LSTM_USE_PREV_ACTION if FULL_USE_LSTM else None,
        "lstm_use_prev_reward": FULL_LSTM_USE_PREV_REWARD if FULL_USE_LSTM else None,
        "replay_sequence_length": FULL_REPLAY_SEQUENCE_LENGTH if FULL_USE_LSTM else 1,
        "replay": "PER",
        "replay_capacity": 100_000,
        "lr": 1e-4,
        "gamma": 0.99,
        "target_network_update_freq": 1000,
    }
    logger = RunLogger(
        base_dir=str(Path("results").resolve()),
        global_seed=GLOBAL_SEED,
        algo_name="DQN",
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

if __name__ == "__main__":
    checkpoint_path = str(Path("results").resolve())
    training(checkpoint_path, max_iterations=20000)
