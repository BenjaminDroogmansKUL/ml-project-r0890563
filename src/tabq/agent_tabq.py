from __future__ import annotations

import sys
from pathlib import Path
_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

from pathlib import Path
import numpy as np
import gymnasium
from pettingzoo.utils import BaseWrapper
from pettingzoo.utils.env import AgentID, ObsType

from config import Config
from tabq import QTable
from featurizer import encode_state
from controller import ShooterController, ACTIONS


def _resolve_qtable_path(cfg_path: Path, qpath_attr: str | Path) -> Path:
    """
    na wat gesuggel iets wat het pad vind
    """
    cfg_dir = Path(cfg_path).parent
    agent_dir = Path(__file__).resolve().parent
    q = Path(qpath_attr)

    candidates = []
    if q.is_absolute():
        candidates = [q]
    else:
        candidates = [
            cfg_dir / q,                          # typical case (qtable.npz)
            agent_dir / q,                        # project-relative path
            cfg_dir / q.name,                     # just the filename next to config
            agent_dir / "results" / "tabq" / q.name,
        ]

    for cand in candidates:
        if cand.exists():
            return cand.resolve()

    return (cfg_dir / q).resolve()


class CustomWrapper(BaseWrapper):
    def observation_space(self, agent: AgentID) -> gymnasium.spaces.Space:
        return super().observation_space(agent)

    def observe(self, agent: AgentID) -> ObsType | None:
        return super().observe(agent)


def _greedy_masked(q_values: np.ndarray, mask: np.ndarray) -> int | None:
    valid = np.where(mask == 1)[0]
    if valid.size == 0:
        return None
    masked = np.full(4, -np.inf, dtype=float)
    masked[valid] = q_values[valid]
    return int(np.argmax(masked))


class CustomPredictFunction:
    def __init__(self, env):
        self.env = env
        self.controller = ShooterController(env)

        agent_dir = Path(__file__).resolve().parent
        cfg_path = (agent_dir / "results" / "tabq" / "config.json").resolve()

        self.config = None
        self.qtab = None
        self.K = 3; self.A = 6; self.B = 3

        try:
            if cfg_path.exists():
                self.config = Config.load(cfg_path)
                self.K = int(getattr(self.config, "K", self.K))
                self.A = int(getattr(self.config, "A", self.A))
                self.B = int(getattr(self.config, "B", self.B))

                qpath_attr = getattr(self.config, "qtable_path", "qtable.npz")
                qpath = _resolve_qtable_path(cfg_path, qpath_attr)

                #  debug
                # print("loaded config from", cfg_path)
                # print("resolved qtable path:", qpath)

                if qpath.exists():
                    self.qtab = QTable.load(qpath)
        except Exception:
            self.config = None
            self.qtab = None

        self._sticky_left = 0
        self._current_slot = None

    def reset(self):
        self.controller.reset()
        self._sticky_left = 0
        self._current_slot = None

    def __call__(self, observation, agent, *args, **kwargs):
        if observation is None or self.qtab is None:
            return ACTIONS["NOOP"]

        state, mask, _ = encode_state(observation, A=self.A, B=self.B, k=4)

        if self._sticky_left > 0 and self._current_slot is not None and mask[self._current_slot] == 1:
            slot = self._current_slot
            self._sticky_left -= 1
        else:
            q_values = self.qtab.q(state)
            slot = _greedy_masked(q_values, mask)
            self._current_slot = slot
            self._sticky_left = max(0, int(self.K) - 1)

        if slot is None:
            return ACTIONS["NOOP"]

        self.controller.set_target_slot(slot)
        return self.controller(observation=observation, agent=agent)
