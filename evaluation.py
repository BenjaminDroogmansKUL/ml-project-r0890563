#!/usr/bin/env python3
# encoding: utf-8
"""
Code used to load an agent and evaluate its performance.

Usage:
    python3 evaluation.py -h

"""
import sys
import argparse
import logging
import importlib.util

import pygame

from utils import create_environment


logger = logging.getLogger(__name__)


def evaluate(env, predict_function, seed_games):

    rewards = {agent: 0 for agent in env.possible_agents}
    do_terminate = False

    for i in seed_games:
        env.reset(seed=i)
        env.action_space(env.possible_agents[0]).seed(i)

        for agent in env.agent_iter():
            obs, reward, termination, truncation, info = env.last()

            for a in env.agents:
                rewards[a] += env.rewards[a]

            if termination or truncation:
                break
            else:
                action = predict_function(obs, agent)
            if env.render_mode == "human":
                events = pygame.event.get()  # This is required to prevent the window from freezing
                for event in events:
                    if event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_q:
                            pygame.quit()
                            do_terminate = True
                if do_terminate:
                    break
            env.step(action)
            if do_terminate:
                break
        if do_terminate:
            break
    env.close()

    avg_reward = sum(rewards.values()) / len(seed_games)
    avg_reward_per_agent = {
        agent: rewards[agent] / len(seed_games)  for agent in env.possible_agents
    }
    print(f"Avg reward: {avg_reward}")
    print("Avg reward per agent, per game: ", avg_reward_per_agent)
    return avg_reward



def main(argv=None):
    parser = argparse.ArgumentParser(description='Load an agent and play the KAZ game.')
    parser.add_argument('--verbose', '-v', action='count', default=0, help='Verbose output')
    parser.add_argument('--quiet', '-q', action='count', default=0, help='Quiet output')
    parser.add_argument('--load', '-l',
                        help=('Load from the given file, otherwise use '
                              'rllib_student_code_to_submit.'))
    parser.add_argument('--screen', '-s', action='store_true',
                        help='Set render mode to human (show game)')
    args = parser.parse_args(argv)

    logger.setLevel(max(logging.INFO - 10 * (args.verbose - args.quiet), logging.DEBUG))
    logger.addHandler(logging.StreamHandler(sys.stdout))

    num_agents = 1
    visual_observation = False
    render_mode = "human" if args.screen else None # "human" or None
    logger.info(f'Show game: {render_mode}')
    if render_mode == "human":
        logger.info(f'Press q to end game')
    logger.info(f'Use pixels: {visual_observation}')

    # Loading student submitted code
    if args.load is not None:
        spec = importlib.util.spec_from_file_location("KAZ_agent", args.load)
        Agent = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(Agent)
        print(Agent)
        CustomWrapper = Agent.CustomWrapper
        CustomPredictFunction = Agent.CustomPredictFunction
    else:
        from submission_single_example_rllib import CustomWrapper, CustomPredictFunction

    # Create the PettingZoo environment for evaluation (with rendering)
    env = create_environment(num_agents=num_agents, render_mode=render_mode,
                             visual_observation=visual_observation)
    env = CustomWrapper(env)

    # Loading best checkpoint and evaluating
    random_seeds = list(range(100)) # We will be using different seeds for evaluation
    evaluate(env, CustomPredictFunction(env), seed_games=random_seeds) 


if __name__ == "__main__":
     sys.exit(main())

