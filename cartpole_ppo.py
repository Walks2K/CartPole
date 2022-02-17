"""
PPO agent for CartPole-v1 using Stable Baselines 3
"""


import os
import time

import gym
from stable_baselines3 import PPO

EPISODES = 10
TIMESTEPS = 10000
MODELS_DIR = f"models/PPO-{int(time.time())}"
LOGDIR = f"logs/PPO-{int(time.time())}"


def main():
    """
    Main function
    """
    if not os.path.exists(MODELS_DIR):
        os.makedirs(MODELS_DIR)

    if not os.path.exists(LOGDIR):
        os.makedirs(LOGDIR)

    env = gym.make('CartPole-v1')
    model = PPO('MlpPolicy', env, verbose=1,
                tensorboard_log=LOGDIR)

    for i in range(1, 10):
        model.learn(total_timesteps=TIMESTEPS,
                    reset_num_timesteps=False, tb_log_name="PPO")
        model.save(f"{MODELS_DIR}/cartpole_ppo_{TIMESTEPS * i}")


if __name__ == "__main__":
    main()
