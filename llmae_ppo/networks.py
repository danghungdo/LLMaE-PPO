"""
Neural network architectures for PPO agent.
"""

import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical


class PolicyNetwork(nn.Module):
    """Policy network for discrete action spaces."""

    def __init__(self, input_dim, hidden_dim=256, num_actions=7):
        super(PolicyNetwork, self).__init__()

        # For MiniGrid observations (image-based)
        if len(input_dim) == 3:  # (C, H, W)
            self.conv_layers = nn.Sequential(
                nn.Conv2d(input_dim[0], 32, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d((4, 4)),
            )
            conv_output_size = 64 * 4 * 4

            self.fc_layers = nn.Sequential(
                nn.Linear(conv_output_size, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, num_actions),
            )
        else:  # Flattened input
            input_size = np.prod(input_dim)
            self.conv_layers = None
            self.fc_layers = nn.Sequential(
                nn.Linear(input_size, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, num_actions),
            )

    def forward(self, x):
        if self.conv_layers is not None:
            x = self.conv_layers(x)
            x = x.view(x.size(0), -1)

        logits = self.fc_layers(x)
        return logits

    def get_action_and_log_prob(self, state):
        """Get action and log probability for given state."""
        logits = self.forward(state)
        probs = F.softmax(logits, dim=-1)
        dist = Categorical(probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action, log_prob

    def get_log_prob(self, state, action):
        """Get log probability of action given state."""
        logits = self.forward(state)
        probs = F.softmax(logits, dim=-1)
        dist = Categorical(probs)
        return dist.log_prob(action)

    def get_entropy(self, state):
        """Get entropy of policy distribution."""
        logits = self.forward(state)
        probs = F.softmax(logits, dim=-1)
        dist = Categorical(probs)
        return dist.entropy()


class ValueNetwork(nn.Module):
    """Value network for state value estimation."""

    def __init__(self, input_dim, hidden_dim=256):
        super(ValueNetwork, self).__init__()

        # For MiniGrid observations (image-based)
        if len(input_dim) == 3:  # (C, H, W)
            self.conv_layers = nn.Sequential(
                nn.Conv2d(input_dim[0], 32, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d((4, 4)),
            )
            conv_output_size = 64 * 4 * 4

            self.fc_layers = nn.Sequential(
                nn.Linear(conv_output_size, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1),
            )
        else:  # Flattened input
            input_size = np.prod(input_dim)
            self.conv_layers = None
            self.fc_layers = nn.Sequential(
                nn.Linear(input_size, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1),
            )

    def forward(self, x):
        if self.conv_layers is not None:
            x = self.conv_layers(x)
            x = x.view(x.size(0), -1)

        value = self.fc_layers(x)
        return value.squeeze(-1)
