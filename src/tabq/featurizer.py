import numpy as np
from typing import Tuple


def get_entities(obs_mat: np.ndarray):
    """Row 0 = archer. Zombies = rows with last col == 1 (is_zombie)."""
    archer = obs_mat[0]
    zombies = [row for row in obs_mat[2:] if row[4] == 1]
    return archer, zombies


def select_bottommost(zombies, k=4):
    """Closest to bottom first: larger dy means lower on screen."""
    return sorted(zombies, key=lambda z: z[2], reverse=True)[:k]


def _angle_offset_from_egocentric(archer_row, zombie_row):
    """
    Signed angle between archer heading (hx, hy) and zombie vector (dx, dy).
    Returns radians in (-pi, pi].
    """
    hx, hy = archer_row[-2], archer_row[-1]
    dx, dy = zombie_row[1], zombie_row[2]
    dot = hx * dx + hy * dy
    cross = hx * dy - hy * dx
    return np.arctan2(cross, dot)


def _wrap_pi(a: float) -> float:
    return (a + np.pi) % (2 * np.pi) - np.pi


def _angle_sector(offset_rad: float, A: int) -> int:
    """Map (-pi, pi] -> {0..A-1}."""
    shifted = (offset_rad + np.pi) % (2 * np.pi)  # [0, 2pi)
    size = (2 * np.pi) / A
    return int(shifted // size)


def _dy_bin(dy: float, t1: float, t2: float) -> int:
    """3 bins: imminent/soon/later based on dy."""
    if dy > t2:
        return 0
    if dy > t1:
        return 1
    return 2


def encode_state(
    obs_mat: np.ndarray,
    *,
    A: int = 6,               # angle sectors
    B: int = 3,               # distance bins
    k: int = 4,               # number of slots
    dy_thresholds=(-0.5, -0.1) # thresholds for dy binning
) -> Tuple[tuple, np.ndarray, list]:
    """
    Encode observation into discrete state for Q-learning.

    Returns:
      state: tuple length k; codes in [0..A*B] where A*B == ABSENT
      mask:  np.array length k of {0,1}
      slots: list of zombies (bottommost first)
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
        dist_idx = _dy_bin(z[2], t1, t2)  # dy directly

        code = ang_idx * B + dist_idx
        state.append(int(code))
        mask.append(1)

    return tuple(state), np.array(mask, dtype=np.int32), slots
