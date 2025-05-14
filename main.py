import gymnasium as gym
import numpy as np
from gym_robotics_custom import RoboGymObservationWrapper
from model import *

if __name__ == '__main__':

    replay_buffer_size = 1000000
    episodes = 1000
    env_name = "PointMaze_UMaze-v3"
    max_episode_steps = 100

    STRAIGHT_MAZE = [[1, 1, 1, 1, 1],
                     [1, 0, 0, 0, 1],
                     [1, 1, 1, 1, 1]]

    env = gym.make(env_name, max_episode_steps=max_episode_steps, maze_map=STRAIGHT_MAZE)
    env = RoboGymObservationWrapper(env)

    observation, info = env.reset()

    observation_size = observation.shape[0]

    observation, info = env.reset()

    for i in range(1000):
        action = env.action_space.sample()
        env.step(action)

    print(observation)