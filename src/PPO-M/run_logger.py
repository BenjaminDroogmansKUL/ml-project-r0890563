from typing import  Dict, Any, List, Optional
import os, csv, json, time, datetime
import numpy as np
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
                "cum_env_steps", "delta_env_steps",
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

    def log_eval(self, iteration: int, result: Dict[str, Any]):
        import json
        eval_sec = result.get("evaluation") or {}
        er = eval_sec.get("env_runners") or {}

        if not er and "env_runners" in result and not result.get("evaluation"):
            er = result["env_runners"]

        num_episodes = er.get("num_episodes")

        return_mean = er.get("episode_return_mean")
        return_min = er.get("episode_return_min")
        return_max = er.get("episode_return_max")

        agent_means = (
                er.get("agent_episode_returns_mean")
                or er.get("module_episode_returns_mean")
                or {}
        )

        if return_mean is None and isinstance(agent_means, dict) and agent_means:
            try:
                vals = [float(v) for v in agent_means.values()]
                return_mean = sum(vals) / len(vals)
            except Exception:
                return_mean = ""

        len_mean = er.get("episode_len_mean")
        len_min = er.get("episode_len_min")
        len_max = er.get("episode_len_max")

        episode_duration_sec_mean = er.get("episode_duration_sec_mean")

        cum_env_steps = self._extract_cum_env_steps(result)
        delta_env_steps = self._extract_delta_env_steps(result)

        sample_time_s = er.get("sample")
        env_steps_per_s = er.get("num_env_steps_sampled_per_second")
        if env_steps_per_s is None:
            try:
                if sample_time_s and sample_time_s > 0 and isinstance(delta_env_steps, (int, float)):
                    env_steps_per_s = float(delta_env_steps) / float(sample_time_s)
            except Exception:
                env_steps_per_s = ""

        weights_seq_no = er.get("weights_seq_no")
        agent_means_json = json.dumps(agent_means or {}, sort_keys=True)

        row = [
            self.run_id, iteration,
            num_episodes if num_episodes is not None else "",
            self._safe_float(return_mean), self._safe_float(return_min), self._safe_float(return_max),
            self._safe_float(len_mean), self._safe_float(len_min), self._safe_float(len_max),
            agent_means_json,
            self._safe_float(episode_duration_sec_mean),
            cum_env_steps if cum_env_steps is not None else "",
            delta_env_steps if delta_env_steps is not None else "",
            self._safe_float(sample_time_s), self._safe_float(env_steps_per_s),
            self._safe_float(weights_seq_no),
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
