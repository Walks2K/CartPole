"""
A2C agent for CartPole-v1 using Stable Baselines 3
"""


import os
import time

import gym
from stable_baselines3 import A2C

EPISODES = 10
TIMESTEPS = 10000
MODELS_DIR = f"models/A2C-{int(time.time())}"
LOGDIR = f"logs/A2C-{int(time.time())}"

def main():
    """
    Main function
    """
    if not os.path.exists(MODELS_DIR):
        os.makedirs(MODELS_DIR)

    if not os.path.exists(LOGDIR):
        os.makedirs(LOGDIR)

    env = gym.make('CartPole-v1')
    model = A2C('MlpPolicy', env, verbose=1,
                tensorboard_log=LOGDIR)

    for i in range(1, 10):
        model.learn(total_timesteps=TIMESTEPS,
                    reset_num_timesteps=False, tb_log_name="A2C")
        model.save(f"{MODELS_DIR}/cartpole_a2c_{TIMESTEPS * i}")


if __name__ == "__main__":
    main()
