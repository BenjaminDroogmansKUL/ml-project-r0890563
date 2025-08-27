from pettingzoo.utils import BaseWrapper
import numpy as np

class StickyActionWrapper(BaseWrapper):
    """With prob p_sticky, repeat the previous action for the current agent."""
    def __init__(self, env, p_sticky: float = 0.25):
        super().__init__(env)
        assert 0.0 <= p_sticky <= 1.0
        self.p_sticky = p_sticky
        self._last_action = {}           # agent_id -> last action
        self._rng = np.random.default_rng()  # seeded in reset()

        # simple counters (optional; useful for sanity checks)
        self._overrides = 0
        self._steps = 0

    def reset(self, seed=None, options=None):
        # seed our RNG so sticky randomness is reproducible
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self._last_action.clear()
        self._overrides = 0
        self._steps = 0
        return super().reset(seed=seed, options=options)

    def step(self, action):
        agent = self.env.agent_selection  # current agent in AEC
        use_action = action
        # only if we already have a last action for this agent
        if agent in self._last_action and self._rng.random() < self.p_sticky:
            use_action = self._last_action[agent]
            self._overrides += 1
        self._steps += 1
        self._last_action[agent] = action

        # (optional) expose a tiny metric in the agent's info dict
        # so you can verify stickiness is happening.
        # PettingZoo keeps infos per agent; we update AFTER env.step:
        super().step(use_action)
        # best-effort attach metric
        try:
            if agent in self.env.infos:
                self.env.infos[agent] = dict(self.env.infos[agent])
                self.env.infos[agent]["sticky_override_frac"] = (
                    self._overrides / max(1, self._steps)
                )
        except Exception:
            pass
