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
        print(raw)
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
        assert out.shape[0] == 25, out.shape
        print(out)
        return out


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
