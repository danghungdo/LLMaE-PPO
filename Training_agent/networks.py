import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class PolicyNetwork(nn.Module):  # actor network
    """
    MLP mapping states to action probabilities.
    """

    def __init__(
        self,
        state_space: gym.spaces.Box,
        action_space: gym.spaces.Discrete,
        hidden_size: int = 128,
    ):
        super().__init__()
        self.state_dim = int(np.prod(state_space.shape))
        self.actor = nn.Sequential(
            layer_init(nn.Linear(self.state_dim, hidden_size)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_size, hidden_size)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_size, action_space.n), std=0.01),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        logits = self.actor(state)
        return logits
    
class ValueNetwork(nn.Module):  # critic network
    """
    MLP mapping states to scalar value estimates.
    """

    def __init__(self, state_space: gym.spaces.Box, hidden_size: int = 64):
        super().__init__()
        self.state_dim = int(np.prod(state_space.shape))
        self.critic = nn.Sequential(
            layer_init(nn.Linear(self.state_dim, hidden_size)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_size, hidden_size)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_size, 1), std=0.01),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        value = self.critic(state)
        return value
