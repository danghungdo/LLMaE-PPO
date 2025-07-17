"""
Main training entry point for PPO.
Handles configuration, initialization, and results plotting.
"""

import time

import hydra
import torch
from omegaconf import DictConfig
from torch.utils.tensorboard import SummaryWriter

from env import create_vector_env
from ppo_agent import PPOAgent
from trainer import PPOTrainer
from utils import plot_and_save_results, set_seed

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True


@hydra.main(
    config_path="configs",
    config_name="ppo",
    version_base="1.1",
)
def main(cfg: DictConfig) -> None:
    """
    Main training function.

    Args:
        cfg: Hydra configuration
    """
    run_name = f"{cfg.agent.name}__{cfg.env.name}__{cfg.seed}__{int(time.time())}"
    writer = SummaryWriter(f"runs/{run_name}")

    # Set random seeds
    set_seed(cfg.seed)

    # Create training environments
    envs = create_vector_env(
        cfg.env.name,
        cfg.env.max_episode_steps,
        cfg.seed,
        cfg.train.num_envs,
        cfg.capture_video,
        run_name,
    )

    # Initialize PPO agent
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
        seed=cfg.seed,
    )

    # Initialize trainer
    trainer = PPOTrainer(agent, writer)

    # Train the agent
    steps, average_returns, _ = trainer.train(
        cfg.train.total_steps, cfg.train.eval_interval, cfg.train.eval_episodes
    )

    # Plot and save results
    plot_and_save_results(steps, average_returns, cfg.seed, cfg.env.name)


if __name__ == "__main__":
    main()
