from utils import create_environment
from sticky_wrapper import StickyActionWrapper
from seed_cycle_wrapper import SeedCycleWrapper
from survival_bonus_wrapper import SurvivalBonusAECWrapper
from angular_zombie_wrapper import AngularZombieWrapper
from flatten_wrapper import FlattenWrapper
from only_zombies_wrapper import OnlyZombiesWrapper

def make_env(
    mode: str,
    num_agents: int = 1,
    visual_observation: bool = False,
    max_zombies: int = 4,
    filter_zombies: bool = False,
    angular_features: bool = False,
    survival_bonus: bool = False,
    p_sticky: float = 0.25,
    survival_reward: float = 0.02,
    train_seeds=None,
    eval_seeds=None,
):
    if train_seeds is None or eval_seeds is None:
        raise ValueError("make_env requires train_seeds and eval_seeds")

    w = create_environment(
        num_agents=num_agents,
        visual_observation=visual_observation,
        max_zombies=max_zombies,
    )

    w = StickyActionWrapper(w, p_sticky=p_sticky)

    if filter_zombies and max_zombies > 0:
        if angular_features:
            w = AngularZombieWrapper(w, k=max_zombies)
        else:
            w = OnlyZombiesWrapper(w, k=max_zombies)
    else:
        w = FlattenWrapper(w)

    if survival_bonus:
        w = SurvivalBonusAECWrapper(w, archer_id="archer_0", survival_reward=survival_reward)

    if mode == "train":
        w = SeedCycleWrapper(w, seed_list=train_seeds)
    elif mode == "eval":
        w = SeedCycleWrapper(w, seed_list=eval_seeds)
    else:
        raise ValueError(f"Unknown env mode: {mode}")

    from pettingzoo.utils.conversions import aec_to_parallel
    from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv
    return ParallelPettingZooEnv(aec_to_parallel(w))
