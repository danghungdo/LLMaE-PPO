"""
Core PPO agent implementation.
Contains the PPOAgent class with network initialization, prediction, and update logic.
Mainly adapted from https://github.com/vwxyzjn/ppo-implementation-details
"""

from typing import Dict, Tuple

import os

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

from agent import AbstractAgent
from networks import PolicyNetwork, ValueNetwork


class PPOAgent(AbstractAgent):
    def __init__(
        self,
        envs: gym.vector.SyncVectorEnv,
        agent_name: str,
        env_id: str,
        max_episode_steps: int,
        num_envs: int,
        num_steps_env: int,
        lr_actor: float,
        lr_critic: float,
        gamma: float,
        gae_lambda: float,
        epochs: int,
        clip_eps: float,
        ent_coef: float,
        vf_coef: float,
        max_grad_norm: float,
        target_kl: float,
        batch_size: int,
        num_minibatches: int,
        hidden_size: int,
        cuda: bool,
        seed: int,
        load_initial_policy: bool = False,
        initial_policy_path: str = None,
        initial_policy_type: str = None,
    ) -> None:
        self.envs = envs
        self.agent_name = agent_name
        self.env_id = env_id
        self.max_episode_steps = max_episode_steps
        self.num_envs = num_envs
        self.num_steps_env = num_steps_env
        self.lr_actor = lr_actor
        self.lr_critic = lr_critic
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_eps = clip_eps
        self.epochs = epochs
        self.batch_size = batch_size
        self.num_minibatches = num_minibatches
        self.minibatch_size = int(self.batch_size // self.num_minibatches)
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.max_grad_norm = max_grad_norm
        self.target_kl = target_kl
        self.cuda = cuda
        self.seed = seed
        self.load_initial_policy = load_initial_policy
        self.initial_policy_path = initial_policy_path
        self.initial_policy_type = initial_policy_type
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() and self.cuda else "cpu"
        )

        # Networks
        self.actor = PolicyNetwork(
            self.envs.single_observation_space,
            self.envs.single_action_space,
            hidden_size,
        ).to(self.device)
        self.critic = ValueNetwork(self.envs.single_observation_space, hidden_size).to(
            self.device
        )

        if self.load_initial_policy:
            print(
                f"Loading {self.initial_policy_type} weights from: {self.initial_policy_path}"
            )
            try:
                if self.initial_policy_type == "ppo_actor_only":
                    # BC initialization: load only actor weights or transfer learning with actor-only
                    self.load_actor_only(self.initial_policy_path)
                elif self.initial_policy_type == "ppo_full":
                    # Transfer learning: load full PPO checkpoint without optimizer
                    self.load_checkpoint(self.initial_policy_path)
                else:
                    raise ValueError(
                        f"Unknown initial_policy_type: {self.initial_policy_type}"
                    )
            except FileNotFoundError:
                print(f"Initial policy file not found: {self.initial_policy_path}")
                raise
            except Exception as e:
                print(f"Error loading initial policy: {e}")
                raise
        else:
            print("Starting PPO training with random weights")

        # Combined optimizer
        self.optimizer = optim.Adam(
            [
                {"params": self.actor.parameters(), "lr": lr_actor, "eps": 1e-5},
                {"params": self.critic.parameters(), "lr": lr_critic, "eps": 1e-5},
            ]
        )

        # Storage setup
        self.states = torch.zeros(
            (self.num_steps_env, self.num_envs)
            + self.envs.single_observation_space.shape
        ).to(self.device)
        self.actions = torch.zeros(
            (self.num_steps_env, self.num_envs) + self.envs.single_action_space.shape
        ).to(self.device)
        self.logprobs = torch.zeros((self.num_steps_env, self.num_envs)).to(self.device)
        self.rewards = torch.zeros((self.num_steps_env, self.num_envs)).to(self.device)
        self.dones = torch.zeros((self.num_steps_env, self.num_envs)).to(self.device)
        self.values = torch.zeros((self.num_steps_env, self.num_envs)).to(self.device)

    def predict(
        self, state: torch.Tensor, action: torch.Tensor = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Predict action and value for given state.

        Args:
            state: Current state
            action: Optional action for computing log probability

        Returns:
            Tuple of (action, log_prob, entropy, value)
        """
        logits = self.actor(state)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(state)

    def compute_gae(
        self,
        rewards: torch.Tensor,
        values: torch.Tensor,
        next_values: torch.Tensor,
        next_dones: torch.Tensor,
        dones: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute Generalized Advantage Estimation (GAE).

        Args:
            rewards: Rewards tensor
            values: Value estimates
            next_values: Next value estimates
            next_dones: Next done flags
            dones: Done flags

        Returns:
            Tuple of (advantages, returns)
        """
        advantages = torch.zeros_like(rewards).to(self.device)
        lastgaelam = 0
        for t in reversed(range(self.num_steps_env)):
            if t == self.num_steps_env - 1:
                nextnonterminal = 1.0 - next_dones
                nextvalues = next_values
            else:
                nextnonterminal = 1.0 - dones[t + 1]
                nextvalues = values[t + 1]
            delta = rewards[t] + self.gamma * nextvalues * nextnonterminal - values[t]
            advantages[t] = lastgaelam = (
                delta + self.gamma * self.gae_lambda * nextnonterminal * lastgaelam
            )
        returns = advantages + values

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (
            advantages.std(unbiased=False) + 1e-8
        )
        return advantages.detach(), returns.detach()

    def update(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        logprobs: torch.Tensor,
        advantages: torch.Tensor,
        returns: torch.Tensor,
        values: torch.Tensor,
    ) -> Dict[str, float]:
        """
        Update the policy and value networks using PPO.

        Args:
            states: State tensor
            actions: Action tensor
            logprobs: Log probability tensor
            advantages: Advantage tensor
            returns: Return tensor
            values: Value tensor

        Returns:
            Dictionary of results (learning rate, value loss, policy loss, entropy, old approx KL, approx KL, clipfrac, explained variance)

        """
        # Flatten tensors
        b_states = states.reshape((-1,) + self.envs.single_observation_space.shape)
        b_actions = actions.reshape((-1,) + self.envs.single_action_space.shape)
        b_logprobs = logprobs.reshape(-1)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        inds = np.arange(self.batch_size)
        clipfracs = []

        for _ in range(self.epochs):
            np.random.shuffle(inds)
            for start in range(0, self.batch_size, self.minibatch_size):
                end = start + self.minibatch_size
                mb_inds = inds[start:end]

                _, newlogprobs, entropy, newvalues = self.predict(
                    b_states[mb_inds], b_actions.long()[mb_inds]
                )
                logratio = newlogprobs - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [
                        ((ratio - 1.0).abs() > self.clip_eps).float().mean().item()
                    ]

                mb_advantages = b_advantages[mb_inds]
                mb_returns = b_returns[mb_inds]

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(
                    ratio, 1 - self.clip_eps, 1 + self.clip_eps
                )
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                newvalues = newvalues.view(-1)
                v_loss = 0.5 * ((newvalues - mb_returns) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - self.ent_coef * entropy_loss + v_loss * self.vf_coef

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
                nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
                self.optimizer.step()
            if self.target_kl is not None:
                if approx_kl > self.target_kl:
                    break

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        results = {
            "learning_rate": self.optimizer.param_groups[0]["lr"],
            "value_loss": v_loss.item(),
            "policy_loss": pg_loss.item(),
            "entropy": entropy_loss.item(),
            "old_approx_kl": old_approx_kl.item(),
            "approx_kl": approx_kl.item(),
            "clipfrac": np.mean(clipfracs),
            "explained_variance": explained_var,
        }
        return results

    def set_train_mode(self) -> None:
        """Set networks to training mode."""
        self.actor.train()
        self.critic.train()

    def set_eval_mode(self) -> None:
        """Set networks to evaluation mode."""
        self.actor.eval()
        self.critic.eval()

    def save_checkpoint(self, path: str, step: int = 0, metadata: dict = None) -> None:
        """
        Save complete PPO checkpoint including networks and optimizer.

        Args:
            path: Path to save checkpoint
            step: Training step
            metadata: Additional metadata (success_rate, mean_return, etc.)
        """

        os.makedirs(os.path.dirname(path), exist_ok=True)

        checkpoint = {
            "step": step,
            "actor_state_dict": self.actor.state_dict(),
            "critic_state_dict": self.critic.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "hyperparameters": {
                "lr_actor": self.lr_actor,
                "lr_critic": self.lr_critic,
                "gamma": self.gamma,
                "gae_lambda": self.gae_lambda,
                "clip_eps": self.clip_eps,
                "epochs": self.epochs,
                "ent_coef": self.ent_coef,
                "vf_coef": self.vf_coef,
                "max_grad_norm": self.max_grad_norm,
            },
        }

        if metadata:
            checkpoint.update(metadata)

        torch.save(checkpoint, path)
        print(f"PPO checkpoint saved to {path}")

    def load_checkpoint(self, path: str) -> None:
        """
        Load PPO checkpoint for transfer learning with both actor and critic.

        Args:
            path: Path to checkpoint file

        Returns:
            dict: Checkpoint metadata
        """
        try:
            checkpoint = torch.load(path, map_location=self.device)

            # Load networks
            self.actor.load_state_dict(checkpoint["actor_state_dict"])
            self.critic.load_state_dict(checkpoint["critic_state_dict"])

            print(f"PPO checkpoint loaded from {path}")

        except FileNotFoundError:
            print(f"Checkpoint not found: {path}")
            raise
        except Exception as e:
            print(f"Error loading checkpoint: {e}")
            raise

        # freeze the first linear layer and its bias
        for name, param in self.actor.named_parameters():
            if "actor.0.weight" in name or "actor.0.bias" in name:
                param.requires_grad = False
        for name, param in self.critic.named_parameters():
            if "critic.0.weight" in name or "critic.0.bias" in name:
                param.requires_grad = False

    def load_actor_only(self, path: str) -> None:
        """
        Load only actor weights (for BC initialization or transfer learning).
        Supports both BC weights and PPO checkpoints.

        Args:
            path: Path to weights file
        """
        try:
            checkpoint = torch.load(path, map_location=self.device)

            # Handle different file formats
            if "actor_state_dict" in checkpoint:
                # PPO checkpoint format
                actor_state_dict = checkpoint["actor_state_dict"]
            else:
                # BC weights format - add "actor." prefix
                actor_state_dict = {f"actor.{k}": v for k, v in checkpoint.items()}
            # Load actor weights
            self.actor.load_state_dict(actor_state_dict)

        except Exception as e:
            print(f"Error loading actor weights: {e}")
            raise
