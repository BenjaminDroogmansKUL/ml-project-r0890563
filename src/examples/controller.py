
import gymnasium
import numpy as np
from gymnasium import spaces
from pettingzoo.utils import BaseWrapper
from pettingzoo.utils.env import AgentID, ObsType
import numpy as np
import random

# adapt if your action ids differ
ACTIONS = {
    "FORWARD": 0,
    "BACKWARD": 1,
    "ROTATE_LEFT": 2,
    "ROTATE_RIGHT": 3,
    "ATTACK": 4,
    "NOOP": 5,
}

class CustomWrapper(BaseWrapper):
    def __init__(self, env):
        super().__init__(env)

    def observation_space(self, agent: AgentID) -> gymnasium.spaces.Space:
        return spaces.flatten_space(super().observation_space(agent))

    def observe(self, agent: AgentID) -> ObsType | None:
        obs = super().observe(agent)
        flat_obs = obs.flatten()
        return obs


def get_entities(obs_mat: np.ndarray):
    archer = obs_mat[0]
    zombies = [row for row in obs_mat[2:] if (row[4] == 1 and row[3] == 0)]
    return archer, zombies

def get_closest_zombies(zombies, k=4):
    return sorted(zombies, key=lambda z: z[2], reverse=True)[:k]

def _angle_offset(archer_row, zombie_row):
    hx, hy = archer_row[-2], archer_row[-1]
    dx, dy = zombie_row[1], zombie_row[2]
    dot = hx * dx + hy * dy
    cross = hx * dy - hy * dx
    return np.arctan2(cross, dot)
import numpy as np

def _wrap_to_pi(a):
    return (a + np.pi) % (2 * np.pi) - np.pi

def _angle_sector(offset_rad, A):
    # map (-pi, pi] -> [0..A-1]
    shifted = (offset_rad + np.pi) % (2*np.pi)  # [0, 2pi)
    size = (2*np.pi) / A
    idx = int(shifted // size)
    return min(max(idx, 0), A-1)

def _dist_bin(y_norm, B):
    # equal-width bins on [0,1]; returns 0..B-1
    y_norm = np.clip(y_norm, 0.0, 1.0)
    edges = np.linspace(0.0, 1.0, B+1)[1:-1]   # internal edges
    return int(np.digitize([y_norm], edges)[0])

def encode_state(
    obs_mat: np.ndarray,
    *,
    A: int = 6,          # angle sectors
    B: int = 3,          # distance bins
    k: int = 4,          # number of slots
    min_y: float = 0.162,
    max_y: float = 0.716
):
    archer, zombies = get_entities(np.asarray(obs_mat))

    # sort by closeness to bottom: larger y = closer to bottom
    zombies_sorted = select_bottommost(zombies, k=k)  # already y-desc

    C_ABSENT = A * B
    state = []
    mask = []

    for i in range(k):
        if i >= len(zombies_sorted):
            state.append(C_ABSENT)
            mask.append(0)
            continue

        z = zombies_sorted[i]

        # angle offset (heading -> archer->zombie), in radians (-pi,pi]
        # use y-up conversion to keep east=0, north=+pi/2
        ax, ay = archer[1], archer[0]
        hx, hy = archer[-2], archer[-1]
        zx, zy = z[1], z[0]

        heading_ang = np.arctan2(-hy, hx)
        target_ang  = np.arctan2(-(zy - ay), (zx - ax))
        offset = _wrap_pi(target_ang - heading_ang)

        # discretize
        ang_idx = _angle_sector(offset, A)

        # normalize y to [0,1] using observed min/max, then bin
        y_norm = (z[0] - min_y) / max(1e-8, (max_y - min_y))
        dist_idx = _dist_bin(y_norm, B)

        code = ang_idx * B + dist_idx
        state.append(int(code))
        mask.append(1)

    return tuple(state), np.array(mask, dtype=np.int32), zombies_sorted


class CustomPredictFunction:
    def __init__(
        self,
        env,
        align_tol_rad: float = 0.12,
        fire_every: int = 3,
        lead_pct: float = 0.1,
        dist_scale: float = 0.3,
        k_closest: int = 4,
        max_lead: float = 0.9,
        forward_deadzone: float = 1e-3,
    ):
        self.env = env
        self.align_tol_rad = float(align_tol_rad)
        self.lead_pct = float(lead_pct)
        self.dist_scale = float(dist_scale)
        self.fire_every = int(max(1, fire_every))
        self.k_closest = int(max(1, k_closest))
        self.max_lead = float(max_lead)
        self.forward_deadzone = float(forward_deadzone)

        self._selected_slot = 0
        self._step_idx = 0

    def set_target_slot(self, slot_index: int):
        self._selected_slot = int(slot_index)

    def reset(self):
        self._step_idx = 0

    def __call__(self, observation, agent, *args, **kwargs) -> int:
        self._step_idx += 1
        if observation is None:
            return ACTIONS["NOOP"]

        obs = np.asarray(observation)
        archer, zombies = get_entities(obs)
        if not zombies:
            return ACTIONS["NOOP"]

        slots = get_closest_zombies(zombies, k=self.k_closest)
        if not (0 <= self._selected_slot < len(slots)):
            return ACTIONS["NOOP"]

        target = slots[self._selected_slot]

        hx, hy = archer[-2], archer[-1]
        dist, dx, dy = target[0], target[1], target[2]

        theta_h = np.arctan2(hx, hy)
        theta_z = np.arctan2(dx, dy)

        if abs(theta_z) < self.forward_deadzone:
            theta_z = 0.0

        eff_lead = self.lead_pct * (1.0 + self.dist_scale * dist)
        eff_lead = max(0.0, min(self.max_lead, eff_lead))

        theta_d = (1.0 - eff_lead) * theta_z

        delta = _wrap_to_pi(theta_d - theta_h)

        if delta > self.align_tol_rad:
            return ACTIONS["ROTATE_LEFT"]
        elif delta < -self.align_tol_rad:
            return ACTIONS["ROTATE_RIGHT"]
        else:
            return ACTIONS["ATTACK"] if (self._step_idx % self.fire_every) == 0 else ACTIONS["NOOP"]