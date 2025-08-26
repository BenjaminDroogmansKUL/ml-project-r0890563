import numpy as np

ACTIONS = {
    "FORWARD": 0,
    "BACKWARD": 1,
    "ROTATE_LEFT": 2,
    "ROTATE_RIGHT": 3,
    "ATTACK": 4,
    "NOOP": 5,
}


def get_entities(obs_mat: np.ndarray):
    """
    Row 0 = archer.
    Zombies = rows with last col == 1 (is_zombie) and row[3] == 0 (not dead), starting from row 2.
    """
    archer = obs_mat[0]
    zombies = [row for row in obs_mat[2:] if (row[4] == 1 and row[3] == 0)]
    return archer, zombies


def get_closest_zombies(zombies, k=4):
    """Closest to bottom first: larger dy (z[2]) means lower on screen."""
    return sorted(zombies, key=lambda z: z[2], reverse=True)[:k]


def _wrap_to_pi(a: float) -> float:
    """Map angle to (-pi, pi]."""
    return (a + np.pi) % (2 * np.pi) - np.pi


class ShooterController:
    """
    Simple greedy shooter:
    - Rotates toward the currently selected slot's zombie.
    - Fires every `fire_every` steps when aligned within `align_tol_rad`.
    - Adds a small lead proportional to target distance.
    """

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

        eff_lead = self.lead_pct * (1.0 + self.dist_scale * float(dist))
        eff_lead = max(0.0, min(self.max_lead, eff_lead))

        theta_d = (1.0 - eff_lead) * theta_z
        delta = _wrap_to_pi(theta_d - theta_h)

        if delta > self.align_tol_rad:
            return ACTIONS["ROTATE_LEFT"]
        elif delta < -self.align_tol_rad:
            return ACTIONS["ROTATE_RIGHT"]
        else:
            return ACTIONS["ATTACK"] if (self._step_idx % self.fire_every) == 0 else ACTIONS["NOOP"]
