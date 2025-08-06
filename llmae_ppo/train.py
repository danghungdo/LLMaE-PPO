"""
Main training entry point for PPO.
Handles configuration, initialization, and results plotting.
"""

import json
import os
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

# Important! Specify the last seed you have or it won't delete the choice file, making it default to your previous settings. (Alternative: Delete multirun_choice.json in main Folder)
final_seed = 1234567


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

    # Check for existing multirun choice
    choice_file = os.path.join(hydra.utils.get_original_cwd(), "multirun_choice.json")
    choice = None
    weights_path = os.path.join(hydra.utils.get_original_cwd(), "pretrained_weights")

    if os.path.exists(choice_file):
        with open(choice_file, "r") as f:
            data = json.load(f)
            choice = data.get("choice")
            weights_path = data.get("weights_path", weights_path)
    else:
        # Prompt user for save/load option
        print("Choose an option for training:")
        print("1. Train from scratch")
        print("2. Load pre-trained weights and fine-tune")
        print("3. Train and save weights")
        choice = input("Enter 1, 2, or 3: ").strip()
        if choice == "2":
            weights_path = input(
                "Enter the path to the pre-trained weights directory (use double backslashes or forward slashes): "
            ).strip()
            weights_path = os.path.normpath(weights_path)
        # Save choice for multirun
        with open(choice_file, "w") as f:
            json.dump({"choice": choice, "weights_path": weights_path}, f)

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
    lr_actor = cfg.agent.lr_actor
    lr_critic = cfg.agent.lr_critic

    if choice == "2":
        # Lower learning rates for fine-tuning
        lr_actor *= 0.1  # e.g., 0.0003 -> 0.00003
        lr_critic *= 0.1

    agent = PPOAgent(
        envs,
        env_id=cfg.env.name,
        max_episode_steps=cfg.env.max_episode_steps,
        num_envs=cfg.train.num_envs,
        num_steps_env=cfg.train.num_steps_env,
        lr_actor=lr_actor,
        lr_critic=lr_critic,
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

    # Load pre-trained weights if selected
    if choice == "2":
        actor_path = os.path.join(weights_path, "actor.pth")
        critic_path = os.path.join(weights_path, "critic.pth")
        if os.path.exists(actor_path) and os.path.exists(critic_path):
            print(f"Loading pre-trained weights from {weights_path}")
            agent.load(weights_path)
        else:
            print(
                f"Error: Pre-trained weights not found at {actor_path} or {critic_path}. Training from scratch."
            )
            choice = "1"  # Fallback to training from scratch

    # Initialize trainer
    trainer = PPOTrainer(agent, writer)

    # Train the agent
    steps, average_returns, _ = trainer.train(
        cfg.train.total_steps, cfg.train.eval_interval, cfg.train.eval_episodes
    )

    # Save weights if selected
    if choice == "3":
        os.makedirs(weights_path, exist_ok=True)
        print(f"Saving weights to {weights_path}")
        agent.save(weights_path)

    # Plot and save results
    plot_and_save_results(steps, average_returns, cfg.seed, cfg.env.name)

    # Clean up choice file after the last run (optional, to avoid reuse in future runs)
    if os.path.exists(choice_file) and cfg.seed == final_seed:
        os.remove(choice_file)


if __name__ == "__main__":
    main()
