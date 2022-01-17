from statistics import mode
import itertools
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
from torch.nn.init import calculate_gain
from collections import deque
import numpy as np
import random
import gym
import matplotlib.pyplot as plt

class DNQNet(nn.Module):
    """Deep Q-learning network that learns the action-value function"""
    def __init__(self, n_in, n_out, device, n_hidden=128, use_GPU = True):
        super(DNQNet, self).__init__()
        self.device = device
        self.net = nn.Sequential(
            nn.Linear(n_in, n_hidden),
            nn.ReLU(inplace=False),
            nn.Linear(n_hidden, n_out)
        ).to(device)
        self._relu_init(self.net[0])
        self._linear_init(self.net[2])

    def _relu_init(self, m):
        nn.init.kaiming_normal_(
            m.weight.data, 0.,
            mode='fan_in',
            nonlinearity='relu'
        )
        nn.init.constant_(m.bias.data, 0.)
    
    def forward(self, input):
        input = input.to(self.device)
        return self.net(input)

    def _linear_init(self, m):
        nn.init.kaiming_normal_(
            m.weight.data, 0.,
            mode='fan_in',
            nonlinearity='linear'
        )
        nn.init.constant_(m.bias.data, 0.)

    def _softmax_init(self, m):
        nn.init.xavier_normal_(
            m.weight.data, calculate_gain('signmoid')
        )
        nn.init.constant_(m.bias.data, 0.)
    
class Replay_Buffer(object):
    def __init__(self, capacity=50000):
        self.memory = deque(maxlen = capacity)
    
    def push(self, state, action , next_state, reward, done):
        self.memory.append((state, action, next_state, reward, done))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

class DNQAgent(nn.Module):
    def __init__(self, env, use_GPU = True, lr = 5e-4, gamma= .99, episode_num=500, eval_reward_freq=100, epsilon_decay = 10000, epsilon_start = 1., epsilon_end=0.02, target_update_freq= 1000, replay_buffer_size=50000, min_replay_size=1000, batch_size = 32):
        super(DNQAgent, self).__init__()
        self.device = "cuda:0" if use_GPU else "cpu"
        self.gamma = gamma
        self.episode_num = episode_num
        self.eval_reward_freq = eval_reward_freq
        self.epsilon_decay = epsilon_decay
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.min_replay_size = min_replay_size
        self.target_update_freq = target_update_freq
        self.batch_size = batch_size
        self.env = env
        self.replay_buffer = Replay_Buffer(replay_buffer_size)
        n_observations = list(env.observation_space.shape)[0]
        self.qvalue = DNQNet(n_observations, env.action_space.n, self.device)
        self.qtarget = DNQNet(n_observations, env.action_space.n, self.device)
        self.optimizer = optim.Adam(self.qvalue.parameters(), lr = lr)
        self.qtarget.load_state_dict(self.qvalue.state_dict())
        self.initial_state = env.reset()
        self.initialize_replay_buffer()


    def get_action_values(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0)
        return self.qvalue(state)

    def pick_action(self, epsilon, state):
        with torch.no_grad():
            action_values = self.get_action_values(state)
            if random.random() <= epsilon:
                return np.random.randint(0, action_values.shape[1])
            else: return torch.argmax(action_values).item()

    def initialize_replay_buffer(self):
        for _ in range(self.min_replay_size):
            state = env.reset()
            done = False
            while not done:
                 action = env.action_space.sample()
                 next_s, reward, done, _ = self.env.step(action)
                 self.replay_buffer.push(state, action, reward, next_s, done)

    def train(self):
        episode_rewards = deque([0.], maxlen=self.eval_reward_freq)
        train_rewards = []
        for episode in range(self.episode_num):
            state = self.env.reset()
            episode_reward = 0.
            done = False
            epsilon = np.interp(episode, [0, self.episode_num], [self.epsilon_start, self.epsilon_end])            
            while not done:
                action = self.pick_action(epsilon, state)
                next_s, reward, done, _ = self.env.step(action)
                self.replay_buffer.push(state, action, reward, next_s, done)
                state = next_s
                episode_reward+=reward
            
            episode_rewards.append(episode_reward)
            
            # pick a batch randomly from the replay buffer
            transitions = self.replay_buffer.sample(self.batch_size)
            states = torch.from_numpy(np.asarray([t[0] for t in transitions], dtype=np.float32))
            actions = torch.from_numpy(np.asarray([t[1] for t in transitions], dtype=np.int64)).unsqueeze(-1)
            dones = torch.from_numpy(np.asarray([t[4] for t in transitions], dtype=np.float32)).unsqueeze(-1)
            rewards = torch.from_numpy(np.asarray([t[2] for t in transitions], dtype=np.float32)).unsqueeze(-1)
            next_states = torch.from_numpy(np.asarray([t[3] for t in transitions], dtype=np.float32))
            
            # compute target q-values
            target_q_values = self.qtarget(next_states).detach()
            max_target_q_values = target_q_values.max(dim=1, keepdim=True)[0]
            targets = rewards + self.gamma*(1-dones)*max_target_q_values

            # compute loss
            q_values = self.qvalue(states)
            action_q_values = torch.gather(input=q_values, dim=1, index=actions)
            loss = nn.functional.smooth_l1_loss(action_q_values, targets)

            # gradient decent 
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # update target network
            if episode % self.target_update_freq == 0:
                self.qtarget.load_state_dict(self.qvalue.state_dict())

            if episode % self.eval_reward_freq == 0:
                average_reward = np.mean(episode_rewards)
                train_rewards.append(average_reward)
                print("Average reward over {} episodes: {}".format(self.eval_reward_freq, average_reward))

        return train_rewards

    def evaluate(self):
        episode_rewards = deque([0.], maxlen= self.eval_reward_freq)
        for _ in range(self.eval_reward_freq):
            episode_reward = 0.
            state = self.env.reset()
            done = False
            while not done:
                action_values = self.get_action_values(state)
                best_action = torch.argmax(action_values).item()
                next_s, reward, done, _ = self.env.step(best_action)
                state = next_s
                episode_reward+=reward
            
            episode_rewards.append(episode_reward)
        print("Evaluate average reward over {} episodes: {}".format(self.eval_reward_freq, np.mean(episode_rewards)))

def visualize_episode(rewards):
    fig, ax = plt.subplots()
    ax.set_ylabel('Average rewards over a number of episodes')
    ax.set_xlabel('Number of episodes')
    ax.plot(rewards)
    plt.show()

if __name__=="__main__":
    env = gym.make("CartPole-v0") 
    env.seed(0) # Set a random seed for the environment 
    agent = DNQAgent(env, use_GPU=False, episode_num=50000)
    train_rewards = agent.train()
    agent.evaluate()
    visualize_episode(train_rewards)
            
