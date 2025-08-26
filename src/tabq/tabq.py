from __future__ import annotations
import numpy as np
from pathlib import Path
from typing import Tuple


class QTable:
    """
    Minimal dense tabular Q-table for 4-action slot selection.
    Shape: (C**4, 4), where C = A*B + 1 (incl. ABSENT).
    """

    def __init__(self, C: int, num_actions: int = 4):
        self.C = int(C)
        self.num_actions = int(num_actions)
        self.Q = np.zeros((self.C ** 4, self.num_actions), dtype=np.float32)

    def state_to_index(self, s: Tuple[int, int, int, int]) -> int:
        c0, c1, c2, c3 = map(int, s)
        C = self.C
        return (((c0 * C + c1) * C + c2) * C) + c3

    def q(self, s: Tuple[int, int, int, int]) -> np.ndarray:
        return self.Q[self.state_to_index(s)]

    def update(self, s: Tuple[int, int, int, int], a: int, target: float, alpha: float = 1.0) -> None:
        idx = self.state_to_index(s)
        self.Q[idx, a] += alpha * (target - self.Q[idx, a])

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, Q=self.Q, C=np.array(self.C), num_actions=np.array(self.num_actions))

    @classmethod
    def load(cls, path: str | Path) -> "QTable":
        data = np.load(Path(path))
        C = int(data["C"])
        num_actions = int(data["num_actions"])
        obj = cls(C=C, num_actions=num_actions)
        obj.Q = data["Q"]
        return obj
