"""
Training orchestration for PPO.
Contains the PPOTrainer class with training loop and evaluation logic.
"""

from typing import List, Tuple

import gymnasium as gym
import numpy as np
import torch

from env import create_eval_env
from ppo_agent import PPOAgent


class PPOTrainer:
    def __init__(self, agent: PPOAgent):
        """
        Initialize the PPO trainer.

        Args:
            agent: PPO agent to train
        """
        self.agent = agent

    def train(
        self,
        total_steps: int,
        eval_interval: int,
        eval_episodes: int,
    ) -> Tuple[List[int], List[float], List[float]]:
        """
        Train the PPO agent.

        Args:
            total_steps: Total number of environment steps
            eval_interval: Interval between evaluations
            eval_episodes: Number of episodes for evaluation

        Returns:
            Tuple of (steps, average_returns, std_returns)
        """
        # Create evaluation environments
        eval_envs = create_eval_env(
            self.agent.env_id,
            self.agent.max_episode_steps,
            self.agent.seed,
            self.agent.num_envs,
        )

        global_step = 0
        obs, info = self.agent.envs.reset()
        next_states = torch.Tensor(obs).to(self.agent.device)
        next_dones = torch.zeros(self.agent.num_envs).to(self.agent.device)
        num_updates = total_steps // self.agent.batch_size

        # create lists of average rewards and steps for plotting
        steps: List[int] = []
        average_returns: List[float] = []
        std_returns: List[float] = []

        print(
            f"Training PPO on {self.agent.env_id} with {self.agent.num_envs} environments for {total_steps} steps..."
        )

        for update in range(1, num_updates + 1):
            # Anneal learning rate
            frac = 1.0 - (update - 1.0) / num_updates
            lrnow_actor = frac * self.agent.lr_actor
            lrnow_critic = frac * self.agent.lr_critic
            self.agent.optimizer.param_groups[0]["lr"] = lrnow_actor
            self.agent.optimizer.param_groups[1]["lr"] = lrnow_critic

            for step in range(self.agent.num_steps_env):
                global_step += 1 * self.agent.num_envs
                self.agent.states[step] = next_states
                self.agent.dones[step] = next_dones

                with torch.no_grad():
                    action, logprob, _, value = self.agent.predict(next_states)
                    self.agent.values[step] = value.flatten()
                self.agent.actions[step] = action
                self.agent.logprobs[step] = logprob

                next_states, reward, termination, truncation, info = (
                    self.agent.envs.step(action.cpu().numpy())
                )
                done = termination | truncation
                self.agent.rewards[step] = torch.tensor(reward).to(self.agent.device)
                next_states = torch.Tensor(next_states).to(self.agent.device)
                next_dones = torch.Tensor(done).to(self.agent.device)

                if global_step % eval_interval == 0:
                    mean_return, std_return = self.evaluate(eval_envs, eval_episodes)
                    steps.append(global_step)
                    average_returns.append(mean_return)
                    std_returns.append(std_return)
                    print(
                        f"[Eval ] Global Step {global_step:6d} AvgReturn {mean_return:5.1f} ± {std_return:4.1f}"
                    )

            # Compute advantages after rollout
            with torch.no_grad():
                next_value = self.agent.critic(next_states).reshape(
                    1, -1
                )  # for boostrapping, in case the last state is not terminal
                advantages, returns = self.agent.compute_gae(
                    self.agent.rewards,
                    self.agent.values,
                    next_value,
                    next_dones,
                    self.agent.dones,
                )

            # PPO update
            self.agent.update(
                self.agent.states,
                self.agent.actions,
                self.agent.logprobs,
                advantages,
                returns,
                self.agent.values,
            )

        print(f"Training complete after {global_step} steps.")

        return steps, average_returns, std_returns

    def evaluate(
        self, eval_envs: gym.vector.SyncVectorEnv, num_episodes: int
    ) -> Tuple[float, float]:
        """
        Evaluate the agent on the given environments.

        Args:
            eval_envs: Evaluation environments
            num_episodes: Number of episodes to evaluate

        Returns:
            Tuple of (mean_return, std_return)
        """
        self.agent.set_eval_mode()

        # List to store returns of completed episodes
        returns = []

        # Reset all environments and get initial states
        obs, infos = eval_envs.reset()
        next_states = torch.Tensor(obs).to(self.agent.device)
        episodes_completed = 0
        episode_returns = np.zeros(self.agent.num_envs)

        # Loop until we have collected enough completed episodes
        while episodes_completed < num_episodes:
            with torch.no_grad():
                # Get actions from the policy
                action, _, _, _ = self.agent.predict(next_states)

                # Step the environments
                next_states, rewards, terminations, truncations, infos = eval_envs.step(
                    action.cpu().numpy()
                )
                next_states = torch.Tensor(next_states).to(self.agent.device)

                # Handle terminations and truncations
                for i in range(self.agent.num_envs):
                    episode_returns[i] += rewards[i]
                    if terminations[i] or truncations[i]:
                        returns.append(episode_returns[i])
                        episode_returns[i] = 0
                        episodes_completed += 1
                        if episodes_completed >= num_episodes:
                            break

        self.agent.set_train_mode()

        return float(np.mean(returns)), float(np.std(returns))
