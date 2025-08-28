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

from sticky_wrapper import StickyActionWrapper

class CustomWrapper(BaseWrapper):
    """
    Assumes base obs is a small matrix where:
      row 0: [archer_x, archer_y, <unused>, hx, hy]
      zombie rows: [dist, dx, dy, <unused>, <unused>]
    If your base layout differs, adapt the column indices below.
    """
    def __init__(self, env, k_nearest=5, coord_scale=200.0):
        """
        k_nearest: number of zombies to encode
        coord_scale: scale used by tanh-normalization for x,y,dx,dy,dist
        """
        super().__init__(env)
        self.k = k_nearest
        self.scale = float(coord_scale)

        # Archer: [x_tanh, y_tanh, hx, hy] -> 4
        # Each zombie: [dist_tanh, dx_tanh, dy_tanh, cosΔ, sinΔ, Δθ/π] -> 6
        self._feat_dim = 4 + self.k * 6

        # Keep obs finite and normalized to [-1, 1]
        self._obs_space = spaces.Box(
            low=-1.0, high=1.0, shape=(self._feat_dim,), dtype=np.float32
        )

    def observation_space(self, agent):
        return self._obs_space

    # --- helpers ---
    def _tanh_norm(self, v):
        return np.tanh(v / self.scale, dtype=np.float32)

    def _safe_unit(self, v):
        n = np.linalg.norm(v)
        return v / n if n > 1e-8 else np.array([1.0, 0.0], dtype=np.float32)

    def _build_features(self, raw_obs):
        """
        raw_obs expected shape ~ (rows, 5).
        row 0: archer: [x, y, ?, hx, hy]
        zombie rows: [dist, dx, dy, ?, ?]
        """
        # --- Archer base features ---
        ax, ay, _, hx, hy = raw_obs[0].astype(np.float32)
        h = self._safe_unit(np.array([hx, hy], dtype=np.float32))

        # normalize pos; if you know world size, swap tanh for linear scaling
        archer_feats = np.array([
            self._tanh_norm(ax),
            self._tanh_norm(ay),
            h[0], h[1],
        ], dtype=np.float32)

        # --- Collect candidate zombie rows ---
        if raw_obs.shape[0] > 1:
            zombies = raw_obs[1:]  # everything after row 0
        else:
            zombies = np.zeros((0, raw_obs.shape[1]), dtype=np.float32)

        # compute distances from dx, dy (more robust than trusting provided 'dist')
        if len(zombies) > 0:
            dx = zombies[:, 1].astype(np.float32)
            dy = zombies[:, 2].astype(np.float32)
            d = np.sqrt(dx*dx + dy*dy, dtype=np.float32)
            order = np.argsort(d)  # nearest first
            zombies = zombies[order]
            dx = dx[order]; dy = dy[order]; d = d[order]
        else:
            dx = dy = d = np.array([], dtype=np.float32)

        # pad/truncate to k
        K = self.k
        if len(d) < K:
            pad_n = K - len(d)
            dx = np.concatenate([dx, np.zeros(pad_n, dtype=np.float32)])
            dy = np.concatenate([dy, np.zeros(pad_n, dtype=np.float32)])
            d  = np.concatenate([d,  np.zeros(pad_n, dtype=np.float32)])
        elif len(d) > K:
            dx = dx[:K]; dy = dy[:K]; d = d[:K]

        # build per-zombie features
        z_feats = []
        for i in range(K):
            v = np.array([dx[i], dy[i]], dtype=np.float32)
            vhat = self._safe_unit(v)
            cos_d = float(h @ vhat)
            # 2D "cross z" for signed angle
            sin_d = float(h[0]*vhat[1] - h[1]*vhat[0])
            ang = np.arctan2(sin_d, cos_d) / np.pi  # in (-1, 1]

            z_feats.extend([
                self._tanh_norm(d[i]),
                self._tanh_norm(dx[i]),
                self._tanh_norm(dy[i]),
                np.float32(cos_d),
                np.float32(sin_d),
                np.float32(ang),
            ])

        return np.concatenate([archer_feats, np.array(z_feats, dtype=np.float32)], axis=0)

    # --- PettingZoo API overrides ---
    def observe(self, agent):
        raw = super().observe(agent)
        if raw is None:
            return None

        # If earlier wrappers already sliced rows/cols, adapt here.
        # Expecting shape (rows, 5). If it's flat, try to reshape.
        if raw.ndim == 1 and raw.size % 5 == 0:
            raw = raw.reshape((-1, 5))
        assert raw.ndim == 2 and raw.shape[1] >= 5, \
            f"Expected (rows, >=5), got {raw.shape}"

        feats = self._build_features(raw)
        return feats.astype(np.float32, copy=False)


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
            action = int(fwd_out["actions"])

        return action
