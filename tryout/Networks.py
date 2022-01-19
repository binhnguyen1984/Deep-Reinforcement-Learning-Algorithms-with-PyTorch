from Utils import uniform_params_initialization
import torch
import torch.nn as nn
from torch.nn.init import calculate_gain

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

class ActorNet(nn.Module):
    def __init__(self, state_size, action_size, action_bound = None, n_hiddens1 = 20, n_hiddens2=20) -> None:
        super(ActorNet, self).__init__()
        self.action_bound = action_bound
        self.net = nn.Sequential(
            nn.Linear(state_size, n_hiddens1),
            nn.ReLU(inplace=False),
            nn.LayerNorm(n_hiddens1),
            nn.Linear(n_hiddens1, n_hiddens2),
            nn.ReLU(inplace=False),
            nn.LayerNorm(n_hiddens2),
            nn.Linear(n_hiddens2, action_size),
            nn.Dropout(0.2),
            nn.Tanh()
        )

        uniform_params_initialization(self.net[0])
        uniform_params_initialization(self.net[3])
        uniform_params_initialization(self.net[6])

    def forward(self, x):
        x = self.net(x)
        if self.action_bound is not None: 
            x = x * torch.Tensor(self.action_bound)
        return x


class CriticNet(nn.Module):
    def __init__(self, state_size, action_size, n_hiddens1 = 20, n_hiddens2=20) -> None:
        super(CriticNet, self).__init__()
            
        self.critic_net = nn.Sequential(
            nn.Linear(state_size + action_size, n_hiddens1),
            nn.ReLU(inplace=False),
            nn.LayerNorm(n_hiddens1),
            nn.Linear(n_hiddens1, n_hiddens2),
            nn.ReLU(inplace=False),
            nn.LayerNorm(n_hiddens2),
            nn.Linear(n_hiddens2, 1),
            nn.Dropout(0.2)           
        )
        uniform_params_initialization(self.critic_net[0])
        uniform_params_initialization(self.critic_net[3])
        uniform_params_initialization(self.critic_net[6])


    def forward(self, state, action):
        output = torch.cat((state, action), dim=1)
        output = self.critic_net(output)
        return output

