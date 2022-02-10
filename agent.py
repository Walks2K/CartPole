import random
from collections import deque

import gym
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

LOAD_MODEL = False


class DQN(nn.Module):
    """
    Deep Q-Learning network
    """

    def __init__(self, state_size, action_size):
        """
        Initialize the agent
        """
        super(DQN, self).__init__()
        self.state_size = state_size
        self.action_size = action_size

        self.fc1 = nn.Linear(state_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_size)

    def forward(self, x):
        """
        Forward pass of the network
        """
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class ReplayBuffer:
    """
    Replay buffer to store the transitions
    """

    def __init__(self, action_size, memory_size, batch_size):
        """
        Initialize the replay buffer
        """
        self.action_size = action_size
        self.memory = []
        self.batch_size = batch_size
        self.memory_size = memory_size

    def add(self, state, action, reward, next_state, done):
        """
        Add a new transition to the replay buffer
        """
        self.memory.append((state, action, reward, next_state, done))
        if len(self.memory) > self.memory_size:
            del self.memory[0]

    def sample(self):
        """
        Sample a random batch of transitions from the replay buffer
        """
        batch = random.sample(self.memory, self.batch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return state, action, reward, next_state, done

    def __len__(self):
        """
        Return the current size of internal memory
        """
        return len(self.memory)


class Agent:
    """
    Deep Q-Learning agent to play CartPole-v1
    """

    def __init__(self, state_size, action_size, lr, gamma, epsilon,
                 epsilon_decay, epsilon_min, batch_size, memory_size, device):
        """
        Initialize the agent
        """
        self.state_size = state_size
        self.action_size = action_size
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.batch_size = batch_size
        self.memory_size = memory_size
        self.device = device

        self.q_eval = DQN(state_size, action_size).to(device)
        self.q_target = DQN(state_size, action_size).to(device)
        self.optimizer = optim.Adam(self.q_eval.parameters(), lr=self.lr)
        self.loss_func = nn.MSELoss()

        self.memory = ReplayBuffer(action_size, self.memory_size,
                                   self.batch_size)

    def choose_action(self, state):
        """
        Choose an action based on the current state
        """
        if np.random.random() < self.epsilon:
            return random.randrange(self.action_size)
        else:
            state = torch.FloatTensor(state).to(self.device)
            return self.q_eval(state).argmax().item()

    def learn(self):
        """
        Learn from the last transition
        """
        if len(self.memory) < self.batch_size:
            return

        state, action, reward, next_state, done = self.memory.sample()
        state = torch.FloatTensor(state).to(self.device)
        next_state = torch.FloatTensor(next_state).to(self.device)
        action = torch.LongTensor(action).to(self.device)
        reward = torch.FloatTensor(reward).to(self.device)
        done = torch.FloatTensor(done).to(self.device)

        q_eval = self.q_eval(state).gather(1, action.unsqueeze(1)).squeeze(1)
        q_next = self.q_eval(next_state).detach().max(1)[0]
        q_target = reward + self.gamma * q_next * (1 - done)

        loss = self.loss_func(q_eval, q_target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def update_target(self):
        """
        Update the target network
        """
        self.q_target.load_state_dict(self.q_eval.state_dict())

    def save(self, path):
        """
        Save the agent to a file
        """
        torch.save(self.q_eval.state_dict(), path)

    def load(self, path):
        """
        Load the agent from a file
        """
        self.q_eval.load_state_dict(torch.load(path))


def train(env, agent, episodes, max_steps, render=False):
    """
    Train the agent
    """
    scores = []
    scores_window = deque(maxlen=100)

    print("- TRAINING - ")

    for episode in range(episodes):
        state = env.reset()
        score = 0
        for step in range(max_steps):
            action = agent.choose_action(state)
            next_state, reward, done, _ = env.step(action)
            agent.memory.add(state, action, reward, next_state, done)
            agent.learn()
            agent.update_target()
            state = next_state
            score += reward
            if done:
                break
        scores_window.append(score)
        scores.append(score)

        if episode % 100 == 0 or episode == episodes - 1:
            print('Episode {}\tAverage Score: {:.2f}'.format(
                episode, np.mean(scores_window)))

        if np.mean(scores_window) >= 200.0:
            print(
                'Environment was solved at episode {:d}!\tAverage Score: {:.2f}'
                .format(episode - 100, np.mean(scores_window)))
            break

    return scores


def test(env, agent, episodes, max_steps, render=False):
    """
    Test the agent
    """
    scores = []
    scores_window = deque(maxlen=100)

    print("- TESTING - ")

    for episode in range(episodes):
        state = env.reset()
        score = 0
        for step in range(max_steps):
            if render:
                env.render()

            action = agent.choose_action(state)
            next_state, reward, done, _ = env.step(action)
            state = next_state
            score += reward
            if done:
                break
        scores_window.append(score)
        scores.append(score)

        print('Episode {}\tScore: {:.2f}'.format(episode, score))

    return scores


def main():
    """
    Main function
    """
    env = gym.make('CartPole-v1')
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    agent = Agent(state_size,
                  action_size,
                  lr=0.001,
                  gamma=0.99,
                  epsilon=1.0,
                  epsilon_decay=0.995,
                  epsilon_min=0.01,
                  batch_size=64,
                  memory_size=100000,
                  device=device)

    if LOAD_MODEL:
        agent.load('cartpole.pth')

    scores = train(env, agent, episodes=500, max_steps=200, render=False)
    agent.save('cartpole.pth')

    plt.plot(np.arange(len(scores)), scores)
    plt.ylabel('Score')
    plt.xlabel('Episode #')
    plt.show()

    # Test the agent
    agent.load('cartpole.pth')
    scores = test(env, agent, episodes=10, max_steps=200, render=True)
    plt.plot(np.arange(len(scores)), scores)
    plt.ylabel('Score')
    plt.xlabel('Episode #')
    plt.show()


if __name__ == '__main__':
    main()