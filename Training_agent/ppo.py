# ppo.py
"""
On-policy Proximal Policy Optimization (PPO) with GAE, clipped surrogate objective,
value-loss coefficient, and entropy bonus, trained for a total number of environment steps.
"""

from typing import Tuple, List

import gymnasium as gym
import minigrid
from minigrid.wrappers import FlatObsWrapper
import numpy as np
from matplotlib import pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True

import os  
import random 
import time 

import hydra  
from omegaconf import DictConfig  
from agent import AbstractAgent  
from networks import PolicyNetwork, ValueNetwork

def make_env(env_name, max_episode_steps ,seed, idx, capture_video, run_name):
    def thunk():
        env = gym.make(env_name, max_episode_steps, render_mode="rgb_array")
        env = FlatObsWrapper(env)  # Flatten the observation space
        env = gym.wrappers.RecordEpisodeStatistics(env)
        if capture_video:
            if idx == 0:
                env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        env.reset(seed=seed)
        if hasattr(env.action_space, "seed"):
            env.action_space.seed(seed)
        if hasattr(env.observation_space, "seed"):
            env.observation_space.seed(seed)
        return env
    return thunk

class PPOAgent(AbstractAgent):
    def __init__(
        self,
        envs: gym.vector.SyncVectorEnv,
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
        seed: int
    ) -> None:
        self.envs = envs
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

        self.device = torch.device("cuda" if torch.cuda.is_available() and self.cuda else "cpu")

        # Networks
        self.actor = PolicyNetwork(self.envs.single_observation_space, self.envs.single_action_space, hidden_size).to(self.device)
        self.critic = ValueNetwork(self.envs.single_observation_space, hidden_size).to(self.device)

        # Combined optimizer
        self.optimizer = optim.Adam(
            [
                {"params": self.actor.parameters(), "lr": lr_actor, 'eps': 1e-5},
                {"params": self.critic.parameters(), "lr": lr_critic, 'eps': 1e-5},
            ]
        )

        # Storage setup
        self.states = torch.zeros((self.num_steps_env, self.num_envs) + self.envs.single_observation_space.shape).to(self.device)
        self.actions = torch.zeros((self.num_steps_env, self.num_envs) + self.envs.single_action_space.shape).to(self.device)
        self.logprobs = torch.zeros((self.num_steps_env, self.num_envs)).to(self.device)
        self.rewards = torch.zeros((self.num_steps_env, self.num_envs)).to(self.device)
        self.dones = torch.zeros((self.num_steps_env, self.num_envs)).to(self.device)
        self.values = torch.zeros((self.num_steps_env, self.num_envs)).to(self.device)

    def predict(
        self, 
        state: torch.Tensor, 
        action: torch.Tensor = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
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
            advantages[t] = lastgaelam = delta + self.gamma * self.gae_lambda * nextnonterminal * lastgaelam
        returns = advantages + values

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)
        return advantages.detach(), returns.detach()

    def update(
        self, 
        states: torch.Tensor,
        actions: torch.Tensor,
        logprobs: torch.Tensor,
        advantages: torch.Tensor,
        returns: torch.Tensor,
        values: torch.Tensor
    ) -> None:
        # Flatten tensors
        b_states = states.reshape((-1,) + self.envs.single_observation_space.shape)
        b_actions = actions.reshape((-1,) + self.envs.single_action_space.shape)
        b_logprobs = logprobs.reshape(-1)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        inds = np.arange(self.batch_size)

        for epoch in range(self.epochs):
            np.random.shuffle(inds)
            for start in range(0, self.batch_size, self.minibatch_size):
                end = start + self.minibatch_size
                mb_inds = inds[start:end]

                _, newlogprobs, entropy, newvalues = self.predict(b_states[mb_inds], b_actions.long()[mb_inds])
                logratio = newlogprobs - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    approx_kl = ((ratio - 1) - logratio).mean()

                mb_advantages = b_advantages[mb_inds]
                mb_returns = b_returns[mb_inds]
                mb_values = b_values[mb_inds]

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                newvalues = newvalues.view(-1)
                
                #v_loss_unclipped = (newvalues - mb_returns) ** 2
                #v_clipped = mb_values + torch.clamp(newvalues - mb_values, -self.clip_eps, self.clip_eps)
                #v_loss_clipped = (v_clipped - mb_returns) ** 2
                #v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                #v_loss = 0.5 * v_loss_max.mean()

                v_loss = 0.5 * ((newvalues - mb_returns) ** 2).mean()
                

                entropy_loss = entropy.mean()
                loss = pg_loss - self.ent_coef * entropy_loss + v_loss * self.vf_coef

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
                nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
                self.optimizer.step()

                
                #if self.target_kl is not None and approx_kl > self.target_kl:
                #    break

    def train(
        self,
        total_steps: int,
        eval_interval: int,
        eval_episodes: int,
    ) -> Tuple[List[int], List[float], List[float]]:
        
        # Create evaluation environments
        eval_envs = gym.vector.SyncVectorEnv(
            [make_env(self.env_id, self.max_episode_steps, self.seed, i, True, "eval") for i in range(self.num_envs)]
        )
        assert isinstance(eval_envs.single_action_space, gym.spaces.Discrete), "only discrete action space is supported"

        global_step = 0
        obs, info = self.envs.reset()
        next_states = torch.Tensor(obs).to(self.device)
        next_dones = torch.zeros(self.num_envs).to(self.device)
        num_updates = total_steps // self.batch_size

        # create lists of average rewards and steps for plotting
        steps: List[int] = []
        average_returns: List[float] = []
        std_returns: List[float] = []

        print(f"Training PPO on {self.env_id} with {self.num_envs} environments for {total_steps} steps...")

        for update in range(1, num_updates + 1):
            # Anneal learning rate
            frac = 1.0 - (update - 1.0) / num_updates
            lrnow_actor = frac * self.lr_actor
            lrnow_critic = frac * self.lr_critic
            self.optimizer.param_groups[0]["lr"] = lrnow_actor
            self.optimizer.param_groups[1]["lr"] = lrnow_critic

            for step in range(self.num_steps_env):
                global_step += 1*self.num_envs
                self.states[step] = next_states
                self.dones[step] = next_dones

                with torch.no_grad():
                    action, logprob, _, value = self.predict(next_states)
                    self.values[step] = value.flatten()
                self.actions[step] = action
                self.logprobs[step] = logprob

                next_states, reward, termination, truncation, info = self.envs.step(action.cpu().numpy())
                done = termination | truncation
                self.rewards[step] = torch.tensor(reward).to(self.device)
                next_states = torch.Tensor(next_states).to(self.device)
                next_dones = torch.Tensor(done).to(self.device)

                if global_step % eval_interval == 0:
                    mean_return, std_return = self.evaluate(eval_envs, eval_episodes)
                    steps.append(global_step)
                    average_returns.append(mean_return)
                    std_returns.append(std_return)
                    print(f"[Eval ] Global Step {global_step:6d} AvgReturn {mean_return:5.1f} ± {std_return:4.1f}")

            # Compute advantages after rollout
            with torch.no_grad():
                next_value = self.critic(next_states).reshape(1, -1)  # for boostrapping, in case the last state is not terminal
                advantages, returns = self.compute_gae(
                    self.rewards, self.values, next_value, next_dones, self.dones
                )

            # PPO update    
            self.update(self.states, self.actions, self.logprobs, advantages, returns, self.values)

        print(f"Training complete after {global_step} steps.")

        return steps, average_returns, std_returns

    def evaluate(
        self, 
        eval_envs: gym.vector.SyncVectorEnv, 
        num_episodes: int
    ) -> Tuple[float, float]:
        
        self.actor.eval()
        self.critic.eval()

        # List to store returns of completed episodes
        returns = []
        
        # Reset all environments and get initial states
        obs, infos = eval_envs.reset()
        next_states = torch.Tensor(obs).to(self.device)
        episodes_completed = 0
        episode_returns = np.zeros(self.num_envs)

        # Loop until we have collected enough completed episodes
        while episodes_completed < num_episodes:
            with torch.no_grad():
                # Get actions from the policy
                action, _, _, _ = self.predict(next_states)
                
                # Step the environments
                next_states, rewards, terminations, truncations, infos = eval_envs.step(action.cpu().numpy())
                next_states = torch.Tensor(next_states).to(self.device)

                # Handle terminations and truncations
                for i in range(self.num_envs):
                    episode_returns[i] += rewards[i]
                    if terminations[i] or truncations[i]:
                        returns.append(episode_returns[i])
                        episode_returns[i] = 0
                        episodes_completed += 1
                        if episodes_completed >= num_episodes:
                            break
                '''
                for info in infos["final_info"]:
                    # The info object is None if the episode is not done
                    if info is not None and "episode" in info:
                        # Append the episode return to list
                        episode_returns.append(info["episode"]["r"])
                        if len(episode_returns) >= num_episodes:
                            break
                '''
                      
        self.actor.train()
        self.critic.train()
                
        return float(np.mean(returns)), float(np.std(returns))

@hydra.main(config_path="../Training_agent/configs/agent/", config_name="ppo", version_base="1.1")
def main(cfg: DictConfig) -> None:
    run_name = f"{cfg.env.name}__{cfg.seed}__{int(time.time())}"
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    torch.cuda.manual_seed(cfg.seed)

    envs = gym.vector.SyncVectorEnv(
        [make_env(cfg.env.name, cfg.env.max_episode_steps, cfg.seed, i, cfg.capture_video, run_name) for i in range(cfg.train.num_envs)]
    )
    assert isinstance(envs.single_action_space, gym.spaces.Discrete), "only discrete action space is supported"

    agent = PPOAgent(
        envs,
        env_id=cfg.env.name,
        max_episode_steps=cfg.env.max_episode_steps,
        num_envs=cfg.train.num_envs,
        num_steps_env=cfg.train.num_steps_env,
        lr_actor=cfg.agent.lr_actor,
        lr_critic=cfg.agent.lr_critic,        
        gamma=cfg.agent.gamma,
        gae_lambda=cfg.agent.gae_lambda,
        epochs=cfg.agent.epochs,
        clip_eps=cfg.agent.clip_eps,
        ent_coef=cfg.agent.ent_coef,
        vf_coef=cfg.agent.vf_coef,
        max_grad_norm=cfg.agent.max_grad_norm,
        target_kl=cfg.agent.target_kl,
        batch_size=int(cfg.train.num_envs * cfg.train.num_steps_env),
        num_minibatches=cfg.train.num_mini_batches,
        hidden_size=cfg.agent.hidden_size,
        cuda=cfg.cuda,
        seed=cfg.seed
    )

    steps, average_returns, _ = agent.train(cfg.train.total_steps, cfg.train.eval_interval, cfg.train.eval_episodes)


    plt.figure(figsize=(10, 6))
    plt.plot(
        steps,
        average_returns,
        label=f"PPO (Seed: {cfg.seed})",
    )
    plt.xlabel("Steps")
    plt.ylabel("Average Return")
    plt.title(
        f"Average Return vs. Steps: PPO (Seed: {cfg.seed}, Env: {cfg.env.name}) "
    )
    plt.grid(True)
    plt.legend()
    # Save the plot as PNG
    plot_path = os.path.join(
        os.path.dirname(__file__), "results", f"average_return_vs_frames.png - {cfg.env.name}.png"
    )
    plt.savefig(plot_path)
    plt.close()
    print(f"Plot saved to: {plot_path}")

if __name__ == "__main__":
    main()