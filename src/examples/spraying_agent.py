import gymnasium
import numpy as np
from gymnasium import spaces
from pettingzoo.utils import BaseWrapper
from pettingzoo.utils.env import AgentID, ObsType

import math
from collections import defaultdict
import numpy as np

class CustomWrapper(BaseWrapper):
    """Flatten the observation so we can print/debug easily."""

    def __init__(self, env):
        super().__init__(env)
        self.episode_no = 0

    def reset(self, seed=None, options=None):
        out = super().reset(seed=seed, options=options)
        self.episode_no += 1
        return out

    def observation_space(self, agent: AgentID) -> gymnasium.spaces.Space:
        return spaces.flatten_space(super().observation_space(agent))

    def observe(self, agent: AgentID) -> ObsType | None:
        obs = super().observe(agent)
        return None if obs is None else obs.flatten()


class CustomPredictFunction:
    """
    Walk-to-center + adaptive sprinkler.

    Phase 1 (Centering):
      • If x < 0.5 → face RIGHT (θ→0) then FORWARD until |x-0.5| ≤ center_tol.
      • If x > 0.5 → face LEFT  (θ→π) then FORWARD until |x-0.5| ≤ center_tol.
      (We only rotate during orientation; no shooting in Phase 1.)

    Phase 2 (Sprinkler):
      • Sweep LEFT <-> RIGHT while shooting more near horizontal and less near vertical.
    """

    ACTIONS = {
        "FORWARD": 0,
        "BACKWARD": 1,
        "ROTATE_LEFT": 2,
        "ROTATE_RIGHT": 3,
        "ATTACK": 4,
        "NOOP": 5,
    }

    def __init__(
        self,
        env,
        *,
        center_tol: float = 0.05,        # stop centering once |x-0.5| ≤ this
        orient_tol_rad: float = 0.07,    # how close heading must be to desired horizontal
        rotate_tol_rad: float = 0.50,    # sprinkler: flip sweep when near horizontal
        min_stride: int = 1,             # sprinkler: near vertical -> rotate often
        max_stride: int = 8,             # sprinkler: near horizontal -> rotate rarely
        turn_to_walk: int = 12,
    ):
        self.env = env
        self.center_tol = float(center_tol)
        self.orient_tol = float(orient_tol_rad)
        self.rotate_tol = float(rotate_tol_rad)
        self.min_stride = max(1, int(min_stride))
        self.max_stride = max(self.min_stride, int(max_stride))
        self.turn_to_walk = int(turn_to_walk)

        # Per-agent state
        self.step_count = defaultdict(int)
        self.sweep_dir = defaultdict(lambda: +1)   # +1=RIGHT, -1=LEFT
        self.centered = defaultdict(lambda: False)  # False => Phase 1, True => Phase 2
        self._last_episode_no = getattr(env, "episode_no", 0)

        self.turns = 0

    @staticmethod
    def _row_width(flat_obs: np.ndarray) -> int:
        return 11 if (len(flat_obs) % 11 == 0) else 5

    @staticmethod
    def _wrap_pi(a: float) -> float:
        return ((a + math.pi) % (2 * math.pi)) - math.pi

    def _heading_xy(self, flat_obs: np.ndarray) -> tuple[float, float]:
        K = self._row_width(flat_obs)
        if K == 11:
            hx, hy = float(flat_obs[9]), float(flat_obs[10])
        else:
            hx, hy = float(flat_obs[3]), float(flat_obs[4])
        n = (hx * hx + hy * hy) ** 0.5
        if n < 1e-8:
            return (1.0, 0.0)
        return (hx / n, hy / n)

    def _x_position(self, flat_obs: np.ndarray) -> float:
        return float(flat_obs[1])

    def _angle_to_horizontal(self, theta: float) -> float:
        to_right = abs(self._wrap_pi(theta - 0.0))
        to_left1 = abs(self._wrap_pi(theta - math.pi))
        to_left2 = abs(self._wrap_pi(theta + math.pi))
        return min(to_right, to_left1, to_left2)

    def _adaptive_stride(self, theta: float) -> int:
        # 0 rad from horizontal => max_stride (more shots); π/2 => min_stride (fewer shots)
        ang = self._angle_to_horizontal(theta)             # [0, π/2]
        norm = min(1.0, max(0.0, ang / (math.pi / 2)))     # 0..1
        stride = self.max_stride - norm * (self.max_stride - self.min_stride)
        return max(self.min_stride, int(round(stride)))

    def __call__(self, observation: np.ndarray, agent: str, *args, **kwargs) -> int:
        # Reset per-episode via wrapper’s episode counter
        current_ep = getattr(self.env, "episode_no", 0)
        if current_ep != self._last_episode_no:
            self.step_count.clear()
            self.sweep_dir.clear()
            self.centered.clear()
            self._last_episode_no = current_ep
            self.turns = 0

        # If agent is inactive this tick
        if observation is None:
            self.step_count[agent] = 0
            self.sweep_dir[agent] = +1
            self.centered[agent] = False
            return self.ACTIONS["NOOP"]

        self.step_count[agent] += 1

        if not self.centered[agent]:
            x = self._x_position(observation)
            print(f"[{agent}] Centering x={x:.3f} to 0.5")

            # Decide desired horizontal direction toward center
            if abs(x - 0.5) < self.center_tol:
                self.centered[agent] = True
            if abs(self.turns) > self.turn_to_walk: return self.ACTIONS["FORWARD"]
            elif x < 0.5:
                self.turns += 1
                return self.ACTIONS["ROTATE_RIGHT"]
            else:
                self.turns -= 1
                return self.ACTIONS["ROTATE_LEFT"]

        # ---------- Phase 1b: turn straight again ----------
        if abs(self.turns) > self.turn_to_walk-5:
            if self.turns > 0:
                self.turns -= 1
                return self.ACTIONS["ROTATE_LEFT"]
            else:
                self.turns += 1
                return self.ACTIONS["ROTATE_RIGHT"]


        # -------- Spray and Pray ---------
        hx, hy = self._heading_xy(observation)
        theta = math.atan2(hy, hx)  # [-π, π], 0=right, ±π=left

        # narrow flip near horizontal
        if abs(self._wrap_pi(theta - 0.0)) <= self.rotate_tol:
            self.sweep_dir[agent] = -1
        elif abs(self._wrap_pi(theta - math.pi)) <= self.rotate_tol or \
             abs(self._wrap_pi(theta + math.pi)) <= self.rotate_tol:
            self.sweep_dir[agent] = +1

        stride = self._adaptive_stride(theta)

        # Rotate periodically to advance sweep; otherwise attack
        if (self.step_count[agent] % stride) == 0:
            return self.ACTIONS["ROTATE_RIGHT"] if self.sweep_dir[agent] > 0 else self.ACTIONS["ROTATE_LEFT"]
        else:
            return self.ACTIONS["ATTACK"]
