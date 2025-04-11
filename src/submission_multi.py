"""Template of your submission file for Task 4 (multi agent KAZ).
The template is identical to the one for Task 3, but it will be used in the tournament for Task 4.
"""
import gymnasium
from pettingzoo.utils import BaseWrapper
from pettingzoo.utils.env import AgentID, ObsType


class CustomWrapper(BaseWrapper):
    """
     Wrapper to use to add state pre-processing (feature engineering)
    """

    def observation_space(self, agent: AgentID) -> gymnasium.spaces.Space:
        pass

    def observe(self, agent: AgentID) -> ObsType | None:
        pass



class CustomPredictFunction:
    """ Function to use to load the trained model and predict the action"""

    def __init__(self, env: gymnasium.Env):
        pass

    def __call__(self, observation, agent, *args, **kwargs):
        pass