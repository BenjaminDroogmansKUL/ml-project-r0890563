from pettingzoo.utils import BaseWrapper
import numpy as np

class StickyActionWrapper(BaseWrapper):
    """With prob p_sticky, repeat the previous action for the current agent.
    IMPORTANT: Respect aec_to_parallel's use of step(None).
    """
    def __init__(self, env, p_sticky: float = 0.25):
        super().__init__(env)
        assert 0.0 <= p_sticky <= 1.0
        self.p_sticky = p_sticky
        self._last_action = {}                 #
        self._rng = np.random.default_rng()    # seeded ()

        self._overrides = 0
        self._steps = 0

    def reset(self, seed=None, options=None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self._last_action.clear()
        self._overrides = 0
        self._steps = 0
        return super().reset(seed=seed, options=options)

    def step(self, action):
        if action is None:
            return super().step(action)

        agent = self.agent_selection

        use_action = action
        if agent in self._last_action and self._rng.random() < self.p_sticky:
            use_action = self._last_action[agent]
            self._overrides += 1

        self._steps += 1
        self._last_action[agent] = action

        return super().step(use_action)
