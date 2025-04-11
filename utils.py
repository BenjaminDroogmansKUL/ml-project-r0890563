import abc
from typing import Callable

import gymnasium
from gymnasium import spaces
from pettingzoo.utils import BaseWrapper
from pettingzoo.utils.env import AgentID, ObsType
import supersuit as ss
from pettingzoo.butterfly import knights_archers_zombies_v10


def create_environment(num_agents=1, max_cycles=1000, render_mode=None, visual_observation=False, max_zombies=4):
    """Create the PettingZoo environment """
    assert num_agents > 0 and num_agents < 3, "Number of agents must be either 1 (one archer) or 2 (two archers)"

    env = knights_archers_zombies_v10.env(max_cycles=max_cycles,
                                          num_archers=num_agents,
                                          num_knights= 0,
                                          max_zombies=max_zombies,
                                          vector_state=not visual_observation,
                                          render_mode = render_mode)
    env = ss.black_death_v3(env)
    if visual_observation:
        # If the observation space is visual, reduce the color channels, resize from 512px to 84px, and apply frame stacking
        env = ss.color_reduction_v0(env, mode="B")
        env = ss.resize_v1(env, x_size=84, y_size=84)
        env = ss.frame_stack_v1(env, 3)

    return env

