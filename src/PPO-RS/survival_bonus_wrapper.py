from pettingzoo.utils import BaseWrapper

class SurvivalBonusAECWrapper(BaseWrapper):
    def __init__(self, env, archer_id: str, survival_reward: float = 0.02):
        super().__init__(env)
        self.archer_id = archer_id
        self.survival_reward = survival_reward

    def step(self, action):
        super().step(action)

        # Only modify rewards if archer is still in the game
        if (
            self.archer_id in self.env.agents
            and not self.env.terminations.get(self.archer_id, False)
            and not self.env.truncations.get(self.archer_id, False)
        ):
            self.env.rewards[self.archer_id] = (
                self.env.rewards.get(self.archer_id, 0.0)  + self.survival_reward
            )
