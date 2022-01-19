from collections import deque, namedtuple
import random
import torch

class Replay_Buffer(object):
    def __init__(self, seed, capacity=50000, device = None):
        self.memory = deque(maxlen=capacity)
        self.experience = namedtuple("Experience", field_names=["state", "action", "next_state", "reward", "done"])
        self.seed = random.seed(seed)
        if device:
            self.device = torch.device(device)
        else:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    def push(self, state, action , next_state, reward, done):
        self.memory.append(self.experience(state, action, next_state, reward, done))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)
