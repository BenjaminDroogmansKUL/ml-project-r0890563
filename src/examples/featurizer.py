import gymnasium
import numpy as np
from gymnasium import spaces
from pettingzoo.utils import BaseWrapper
from pettingzoo.utils.env import AgentID, ObsType

def get_entities(obs_mat: np.ndarray):
    """Row 0 = archer. Zombies = rows with last col == 1 (is_zombie)."""
    archer = obs_mat[0]
    zombies = [row for row in obs_mat[2:] if row[4] == 1]  # skip row 1, keep only zombies
    return archer, zombies

def select_bottommost(zombies, k=4):
    """Closest to bottom first: larger dy means lower on screen."""
    return sorted(zombies, key=lambda z: z[2], reverse=True)[:k]

def _angle_offset_from_egocentric(archer_row, zombie_row):
    """
    Signed angle between archer heading (hx, hy) and zombie vector (dx, dy),
    both in screen coords. Returns radians in (-pi, pi].
    """
    hx, hy = archer_row[-2], archer_row[-1]   # heading vector
    dx, dy = zombie_row[1], zombie_row[2]     # egocentric offsets
    dot   = hx * dx + hy * dy
    cross = hx * dy - hy * dx
    return np.arctan2(cross, dot)

def _angle_sector(offset_rad, A: int):
    """Map (-pi, pi] -> {0..A-1}."""
    shifted = (offset_rad + np.pi) % (2 * np.pi)
    return int(shifted // (2 * np.pi / A))

def _dy_bin(dy: float, t1: float, t2: float):
    """
    Simple 3-bin scheme based on dy (how far below the archer):
      dy > t2  -> 0 (imminent)
      t1 < dy <= t2 -> 1 (soon)
      dy <= t1 -> 2 (later / at or above archer)
    """
    if dy > t2:
        return 0
    if dy > t1:
        return 1
    return 2

def encode_state(
    obs_mat: np.ndarray,
    *,
    A: int = 6,              # angle sectors
    B: int = 3,              # distance bins
    k: int = 4,              # number of slots
    dy_thresholds = (0.1, 0.35)  # (t1, t2) in same units as dy
):
    """
    Returns:
      state: tuple length k; codes in [0..A*B] where A*B == ABSENT
      mask:  np.array length k of {0,1}
      slots: list of up to k zombie rows (bottommost first)
    """
    archer, zombies = get_entities(np.asarray(obs_mat))
    slots = select_bottommost(zombies, k=k)

    C_ABSENT = A * B
    state, mask = [], []
    t1, t2 = dy_thresholds

    for i in range(k):
        if i >= len(slots):
            state.append(C_ABSENT)
            mask.append(0)
            continue

        z = slots[i]
        offset = _angle_offset_from_egocentric(archer, z)
        ang_idx = _angle_sector(offset, A)
        dist_idx = _dy_bin(z[2], t1, t2)  # uses dy directly

        code = ang_idx * B + dist_idx
        state.append(int(code))
        mask.append(1)

    return tuple(state), np.array(mask, dtype=np.int32), slots


class CustomWrapper(BaseWrapper):
    def __init__(self, env):
        super().__init__(env)

    def observation_space(self, agent: AgentID) -> gymnasium.spaces.Space:
        return spaces.flatten_space(super().observation_space(agent))

    def observe(self, agent: AgentID) -> ObsType | None:
        obs = super().observe(agent)
        flat_obs = obs.flatten()
        return obs

def _wrap_pi(a):
    return (a + np.pi) % (2*np.pi) - np.pi

def _angle_sector(offset_rad, A):
    shifted = (offset_rad + np.pi) % (2*np.pi)  # [0, 2pi)
    size = (2*np.pi) / A
    idx = int(shifted // size)
    return min(max(idx, 0), A-1)

def _dist_bin(y_norm, B):
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
    """
    Returns:
      state: tuple length k; codes in [0..A*B] where A*B means ABSENT
      mask:  np.array length k; 1 for present, 0 for ABSENT
      sorted_zombies: list of zombie rows (up to k), closest to bottom first
    """
    archer, zombies = get_entities(np.asarray(obs_mat))

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

        ax, ay = archer[1], archer[0]
        hx, hy = archer[-2], archer[-1]
        zx, zy = z[1], z[0]

        heading_ang = np.arctan2(-hy, hx)
        target_ang  = np.arctan2(-(zy - ay), (zx - ax))
        offset = _wrap_pi(target_ang - heading_ang)

        ang_idx = _angle_sector(offset, A)

        y_norm = (z[0] - min_y) / max(1e-8, (max_y - min_y))
        dist_idx = _dist_bin(y_norm, B)

        code = ang_idx * B + dist_idx
        state.append(int(code))
        mask.append(1)

    return tuple(state), np.array(mask, dtype=np.int32), zombies_sorted


class CustomPredictFunction:
    ACTIONS = {
        "NOOP": 5,
        "FORWARD": 0,
        "BACKWARD": 1,
        "ROTATE_LEFT": 2,
        "ROTATE_RIGHT": 3,
        "ATTACK": 4,
    }

    def __init__(self, env, align_tol_rad: float = 0.10):
        self.env = env
        self.step_count = 0
        self._last_episode_no = getattr(env, "episode_no", 0)
        self.shoot = False
        self.min_y = float("inf")
        self.max_y = float("-inf")

    def reset(self):
        print("done")
        self.step_count = 0

    def __call__(self, observation: np.ndarray, agent: str, *args, **kwargs) -> int:
        print(observation)

        if observation is None:
            return self.ACTIONS["NOOP"]

        self.step_count += 1

        obs_mat = np.asarray(observation)
        archer, zombies = get_entities(obs_mat)

        for z in zombies:
            y = z[0]
            if y < self.min_y:
                self.min_y = y
            if y > self.max_y:
                self.max_y = y

        print(f"[step {self.step_count}] min_y={self.min_y:.3f}, max_y={self.max_y:.3f}")
        closest = select_bottommost(zombies,4)
        for z in closest:
            print("offset:", _angle_offset_from_egocentric(archer, z))

        return self.ACTIONS["NOOP"]
