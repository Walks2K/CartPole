"""
Load models for CartPole-v1 to test (SB3)
"""


import os

import gym
from stable_baselines3 import A2C, DQN, PPO

EPISODES = 10
MODELS_DIR = "models/PPO-1645100770"
MODEL_TO_LOAD = f"{MODELS_DIR}/cartpole_ppo_90000.zip"


def main():
    """
    Main function
    """
    env = gym.make('CartPole-v1')
    model = PPO.load(MODEL_TO_LOAD, env)

    for episode in range(EPISODES):
        obs = env.reset()
        done = False
        while not done:
            env.render()
            action, _states = model.predict(obs)
            obs, rewards, done, info = env.step(action)

    env.close()


if __name__ == "__main__":
    main()
