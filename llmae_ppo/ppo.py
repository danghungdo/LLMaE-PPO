"""
PPO Agent and Trainer implementation.
"""

import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from .networks import PolicyNetwork, ValueNetwork


class PPOAgent:
    """PPO Agent for interacting with environment."""

    def __init__(self, state_dim, action_dim, lr=3e-4, device="cpu"):
        self.device = device
        self.action_dim = action_dim

        # Networks
        self.policy_net = PolicyNetwork(state_dim, num_actions=action_dim).to(device)
        self.value_net = ValueNetwork(state_dim).to(device)

        # Optimizers
        self.policy_optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.value_optimizer = optim.Adam(self.value_net.parameters(), lr=lr)

    def get_action(self, state):
        """Get action from policy network."""
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            action, log_prob = self.policy_net.get_action_and_log_prob(state)
        return action.cpu().numpy()[0], log_prob.cpu().numpy()[0]

    def get_value(self, state):
        """Get value from value network."""
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            value = self.value_net(state)
        return value.cpu().numpy()[0]

    def save(self, filepath):
        """Save model parameters."""
        torch.save(
            {
                "policy_net": self.policy_net.state_dict(),
                "value_net": self.value_net.state_dict(),
                "policy_optimizer": self.policy_optimizer.state_dict(),
                "value_optimizer": self.value_optimizer.state_dict(),
            },
            filepath,
        )

    def load(self, filepath):
        """Load model parameters."""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.policy_net.load_state_dict(checkpoint["policy_net"])
        self.value_net.load_state_dict(checkpoint["value_net"])
        self.policy_optimizer.load_state_dict(checkpoint["policy_optimizer"])
        self.value_optimizer.load_state_dict(checkpoint["value_optimizer"])


class PPOTrainer:
    """PPO Trainer for updating the agent."""

    def __init__(
        self,
        agent,
        gamma=0.99,
        gae_lambda=0.95,
        clip_epsilon=0.2,
        value_coef=0.5,
        entropy_coef=0.01,
        max_grad_norm=0.5,
    ):
        self.agent = agent
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm

    def compute_gae(self, rewards, values, dones, next_value):
        """Compute Generalized Advantage Estimation."""
        advantages = []
        gae = 0

        for i in reversed(range(len(rewards))):
            if i == len(rewards) - 1:
                next_non_terminal = 1.0 - dones[i]
                next_value_i = next_value
            else:
                next_non_terminal = 1.0 - dones[i]
                next_value_i = values[i + 1]

            delta = (
                rewards[i] + self.gamma * next_value_i * next_non_terminal - values[i]
            )
            gae = delta + self.gamma * self.gae_lambda * next_non_terminal * gae
            advantages.insert(0, gae)

        return advantages

    def update(
        self, states, actions, old_log_probs, rewards, dones, epochs=4, batch_size=64
    ):
        """Update policy and value networks using PPO."""
        states = torch.FloatTensor(np.array(states)).to(self.agent.device)
        actions = torch.LongTensor(actions).to(self.agent.device)
        old_log_probs = torch.FloatTensor(old_log_probs).to(self.agent.device)
        rewards = torch.FloatTensor(rewards).to(self.agent.device)
        dones = torch.FloatTensor(dones).to(self.agent.device)

        # Compute values and advantages
        with torch.no_grad():
            values = self.agent.value_net(states)
            next_value = self.agent.value_net(states[-1:])

        advantages = self.compute_gae(
            rewards.cpu().numpy(),
            values.cpu().numpy(),
            dones.cpu().numpy(),
            next_value.cpu().numpy()[0],
        )
        advantages = torch.FloatTensor(advantages).to(self.agent.device)
        returns = advantages + values

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # PPO update
        dataset_size = len(states)
        indices = list(range(dataset_size))

        policy_losses = []
        value_losses = []
        entropy_losses = []

        for _ in range(epochs):
            random.shuffle(indices)

            for start in range(0, dataset_size, batch_size):
                end = min(start + batch_size, dataset_size)
                batch_indices = indices[start:end]

                batch_states = states[batch_indices]
                batch_actions = actions[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                batch_advantages = advantages[batch_indices]
                batch_returns = returns[batch_indices]

                # Policy loss
                new_log_probs = self.agent.policy_net.get_log_prob(
                    batch_states, batch_actions
                )
                ratio = torch.exp(new_log_probs - batch_old_log_probs)

                surr1 = ratio * batch_advantages
                surr2 = (
                    torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon)
                    * batch_advantages
                )
                policy_loss = -torch.min(surr1, surr2).mean()

                # Value loss
                new_values = self.agent.value_net(batch_states)
                value_loss = nn.MSELoss()(new_values, batch_returns)

                # Entropy loss
                entropy = self.agent.policy_net.get_entropy(batch_states).mean()
                entropy_loss = -self.entropy_coef * entropy

                # Update policy network
                self.agent.policy_optimizer.zero_grad()
                policy_loss.backward(retain_graph=True)
                nn.utils.clip_grad_norm_(
                    self.agent.policy_net.parameters(), self.max_grad_norm
                )
                self.agent.policy_optimizer.step()

                # Update value network
                self.agent.value_optimizer.zero_grad()
                value_loss.backward()
                nn.utils.clip_grad_norm_(
                    self.agent.value_net.parameters(), self.max_grad_norm
                )
                self.agent.value_optimizer.step()

                policy_losses.append(policy_loss.item())
                value_losses.append(value_loss.item())
                entropy_losses.append(entropy_loss.item())

        return {
            "policy_loss": np.mean(policy_losses),
            "value_loss": np.mean(value_losses),
            "entropy_loss": np.mean(entropy_losses),
        }
