import numpy as np
from gymnasium import spaces
from pettingzoo.utils import BaseWrapper


class OnlyZombiesWrapper(BaseWrapper):
    def __init__(self, env, k: int = 4):
        super().__init__(env)
        self.k = int(k)

    def observation_space(self, agent):
        base_space = super().observation_space(agent)
        low, high = float(np.min(base_space.low)), float(np.max(base_space.high))
        return spaces.Box(low=low, high=high, shape=(5 * (self.k + 1),), dtype=np.float32)

    def _select_last_k_zombies(self, mat: np.ndarray) -> np.ndarray:
        zmask = (mat[:, 3] == 0.0) & (mat[:, 4] == 1.0)
        zmask[0] = False
        idx = np.where(zmask)[0]
        if idx.size == 0:
            return np.zeros((self.k, 5), dtype=np.float32)
        idx = idx[-self.k:]
        rows = mat[idx].astype(np.float32, copy=False)
        if rows.shape[0] < self.k:
            pad = np.zeros((self.k - rows.shape[0], 5), dtype=np.float32)
            rows = np.vstack([rows, pad])
        return rows

    def observe(self, agent):
        obs = super().observe(agent)
        if obs is None:
            return None
        obs = np.asarray(obs, dtype=np.float32)
        if obs.ndim == 1:
            obs = obs.reshape((-1, 5))
        archer = obs[0].astype(np.float32, copy=True)
        z_rows = self._select_last_k_zombies(obs)
        final = np.vstack([archer.reshape(1, 5), z_rows])
        return final.flatten().astype(np.float32, copy=False)
