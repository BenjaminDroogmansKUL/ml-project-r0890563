import numpy as np
from gymnasium import spaces
from pettingzoo.utils import BaseWrapper


class AngularZombieWrapper(BaseWrapper):
    def __init__(self, env, k: int = 4):
        super().__init__(env)
        self.k = int(k)

    def observation_space(self, agent):
        base_space = super().observation_space(agent)
        low, high = float(np.min(base_space.low)), float(np.max(base_space.high))
        return spaces.Box(low=low, high=high, shape=(5 * (self.k + 1),), dtype=np.float32)

    @staticmethod
    def _unit(v: np.ndarray) -> np.ndarray:
        n = np.linalg.norm(v)
        return v / n if n > 1e-8 else np.array([1.0, 0.0], dtype=np.float32)

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
        raw = super().observe(agent)
        if raw is None:
            return None
        raw = np.asarray(raw, dtype=np.float32)
        if raw.ndim == 1:
            raw = raw.reshape((-1, 5))
        archer = raw[0].astype(np.float32, copy=True)
        h = self._unit(np.array([archer[3], archer[4]], dtype=np.float32))
        zsrc = self._select_last_k_zombies(raw)
        z_rows = []
        for dist, dx, dy, _, _ in zsrc:
            if dist == dx == dy == 0.0:
                z_rows.append(np.zeros(5, dtype=np.float32))
            else:
                vhat = self._unit(np.array([dx, dy], dtype=np.float32))
                cos_d, sin_d = float(h @ vhat), float(h[0] * vhat[1] - h[1] * vhat[0])
                z_rows.append(np.array([dist, dx, dy, cos_d, sin_d], dtype=np.float32))
        final = np.vstack([archer.reshape(1, 5)] + z_rows)
        return final.flatten().astype(np.float32, copy=False)
