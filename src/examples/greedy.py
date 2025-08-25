import numpy as np
import math
import gymnasium
from gymnasium import spaces
from pettingzoo.utils import BaseWrapper
from pettingzoo.utils.env import AgentID, ObsType

class CustomWrapper(BaseWrapper):
    def observation_space(self, agent: AgentID) -> gymnasium.spaces.Space:
        return spaces.flatten_space(super().observation_space(agent))

    def observe(self, agent: AgentID) -> ObsType | None:
        obs = super().observe(agent)
        return None if obs is None else obs.flatten()

def _row_width(flat_obs: np.ndarray) -> int:
    return 11 if (len(flat_obs) % 11) == 0 else 5

def _rows(flat_obs: np.ndarray, K: int) -> np.ndarray:
    return flat_obs.reshape(-1, K)

def _xy_swapped(row: np.ndarray) -> tuple[float, float]:
    return float(row[1]), float(row[0])

def _is_dead_or_absent(x: float, y: float) -> bool:
    return x == 0.0 and y == 0.0

def agent_angle(flat_obs: np.ndarray) -> float:
    """
    Return agent's facing angle in radians (-pi..pi).
    0 rad = facing right, pi/2 = facing up, -pi/2 = facing down.
    """
    K = _row_width(flat_obs)
    rows = flat_obs.reshape(-1, K)

    if K == 11:
        hx, hy = float(rows[0, 9]), float(rows[0, 10])
    else:  # K == 5
        hx, hy = float(rows[0, 3]), float(rows[0, 4])

    n = math.sqrt(hx * hx + hy * hy)
    if n < 1e-8:
        return 0.0  # default if no heading vector
    hx, hy = hx / n, hy / n

    theta = math.atan2(hy, hx)
    return theta;

def angle_to_entity(flat_obs: np.ndarray, entity_row: int) -> float:
    """
    Return the angle from the agent to a given entity row (in radians).
    """
    K = _row_width(flat_obs)
    rows = flat_obs.reshape(-1, K)

    # agent position (remember x=row[1], y=row[0])
    ax, ay = float(rows[0, 1]), float(rows[0, 0])
    # entity position
    ex, ey = float(rows[entity_row, 1]), float(rows[entity_row, 0])

    dx, dy = ex - ax, ey - ay
    return math.atan2(dy, dx)


def angle_diff(agent_theta: float, target_theta: float) -> float:
    """
    Smallest signed difference agent->target in radians, in [-pi, pi].
    Positive = target is CCW (left) of agent's facing.
    Negative = target is CW (right).
    """
    diff = (target_theta - agent_theta + math.pi) % (2 * math.pi) - math.pi
    return diff

def closest_existing_entity(flat_obs: np.ndarray):
    """
    Returns (index, distance, (x,y), K, rows, agent_xy) for the closest
    non-agent row with coords != (0,0). If none exist, returns (None,...).
    """
    K = _row_width(flat_obs)
    R = _rows(flat_obs, K)
    ax, ay = _xy_swapped(R[0])

    best_i, best_d, best_xy = None, float("inf"), None
    for i in range(1, R.shape[0]):
        ex, ey = _xy_swapped(R[i])
        if _is_dead_or_absent(ex, ey):
            continue
        d = ((ex - ax) ** 2 + (ey - ay) ** 2) ** 0.5
        if d < best_d:
            best_i, best_d, best_xy = i, d, (ex, ey)

    return best_i, best_d, best_xy, K, R, (ax, ay)

import math
import numpy as np

class CustomPredictFunction:
    ACTIONS = {"FORWARD":0, "BACKWARD":1, "ROTATE_LEFT":2, "ROTATE_RIGHT":3, "ATTACK":4, "NOOP":5}

    def __init__(self, env, shoot_prob: float = 0.6, tol_deg: float = 5.0):
        """
        shoot_prob: chance to ATTACK even if not perfectly aligned (0..1).
        tol_deg: if |angle diff| <= tol_deg, always ATTACK.
        """
        self.env = env
        self.shoot_prob = float(np.clip(shoot_prob, 0.0, 1.0))
        self.tol_rad = math.radians(tol_deg)

    def __call__(self, observation: np.ndarray, agent: str, *args, **kwargs) -> int:
        if observation is None:
            return self.ACTIONS["NOOP"]

        # find closest existing entity (your helpers)
        idx, dist, xy, K, R, (ax, ay) = closest_existing_entity(observation)
        if idx is None:
            return self.ACTIONS["NOOP"]

        theta = agent_angle(observation)
        target_theta = angle_to_entity(observation, idx)
        diff = angle_diff(theta, target_theta)

        # Debug (optional)
        # print(f"Facing={math.degrees(theta):.1f}°, target={math.degrees(target_theta):.1f}°, diff={math.degrees(diff):.1f}°")

        # If already aligned -> always shoot
        if abs(diff) <= self.tol_rad:
            return self.ACTIONS["ATTACK"]

        # Otherwise: probabilistic shoot vs rotate-to-aim
        if np.random.random() < self.shoot_prob:
            return self.ACTIONS["ATTACK"]
        else:
            return self.ACTIONS["ROTATE_LEFT"] if diff > 0 else self.ACTIONS["ROTATE_RIGHT"]
