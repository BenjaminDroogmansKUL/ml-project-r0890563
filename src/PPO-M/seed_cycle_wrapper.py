from pettingzoo.utils import BaseWrapper
import itertools

class SeedCycleWrapper(BaseWrapper):
    """Cycles through a fixed list of seeds
    """
    def __init__(self, env, seed_list):
        super().__init__(env)
        self._seed_iter = itertools.cycle(list(seed_list))

    def reset(self, seed=None, options=None):
        forced_seed = next(self._seed_iter)
        return super().reset(seed=forced_seed, options=options)
