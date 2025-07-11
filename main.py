"""
Simplified PPO training script for MiniGrid environments.
"""

import argparse
import os
import time

from llmae_ppo import MiniGridWrapper, PPOAgent, PPOTrainer
from llmae_ppo.utils import set_seed


def collect_trajectories(env, agent, n_steps):
    """Collect trajectories from environment."""
    states = []
    actions = []
    log_probs = []
    rewards = []
    dones = []

    state, _ = env.reset()

    for _ in range(n_steps):
        action, log_prob = agent.get_action(state)
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        states.append(state)
        actions.append(action)
        log_probs.append(log_prob)
        rewards.append(reward)
        dones.append(done)

        if done:
            state, _ = env.reset()
        else:
            state = next_state

    return states, actions, log_probs, rewards, dones


def train_ppo(env_name="MiniGrid-Empty-8x8-v0", total_timesteps=100000, device="cpu"):
    """Simplified training function."""
    # Set seed for reproducibility
    set_seed(42)

    # Create environment
    env = MiniGridWrapper(
        env_name, flatten_obs=True, normalize_obs=True, max_steps=1000
    )

    # Get environment dimensions
    state_dim = env.observation_space.shape
    action_dim = env.action_space.n

    print(f"Environment: {env_name}")
    print(f"State dimension: {state_dim}")
    print(f"Action dimension: {action_dim}")

    # Initialize agent and trainer
    agent = PPOAgent(state_dim, action_dim, lr=3e-4, device=device)
    trainer = PPOTrainer(agent)

    # Create save directory
    os.makedirs("checkpoints", exist_ok=True)

    # Training parameters
    n_steps = 2048
    n_epochs = 4
    batch_size = 64

    # Training loop
    total_steps = 0
    episode_count = 0
    start_time = time.time()

    print(f"Starting training for {total_timesteps} steps...")

    while total_steps < total_timesteps:
        # Collect trajectories
        states, actions, log_probs, rewards, dones = collect_trajectories(
            env, agent, n_steps
        )

        # Update agent
        training_metrics = trainer.update(
            states,
            actions,
            log_probs,
            rewards,
            dones,
            epochs=n_epochs,
            batch_size=batch_size,
        )

        total_steps += n_steps
        episode_count += sum(dones)

        # Log progress every 10 updates
        if total_steps % (n_steps * 10) == 0:
            elapsed_time = time.time() - start_time
            steps_per_sec = total_steps / elapsed_time

            print(
                f"Steps: {total_steps}/{total_timesteps} "
                f"({100 * total_steps / total_timesteps:.1f}%) | "
                f"Episodes: {episode_count} | "
                f"Time: {elapsed_time:.0f}s | "
                f"Steps/sec: {steps_per_sec:.1f}"
            )
            print(f"  Policy loss: {training_metrics['policy_loss']:.4f}")
            print(f"  Value loss: {training_metrics['value_loss']:.4f}")
            print(f"  Entropy loss: {training_metrics['entropy_loss']:.4f}")

        # Save checkpoint every 50 updates
        if total_steps % (n_steps * 50) == 0:
            save_path = f"checkpoints/ppo_checkpoint_{total_steps}.pt"
            agent.save(save_path)
            print(f"Checkpoint saved: {save_path}")

    # Save final model
    final_save_path = "checkpoints/ppo_final.pt"
    agent.save(final_save_path)
    print(f"Final model saved: {final_save_path}")

    env.close()
    print("Training completed!")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Train PPO on MiniGrid")
    parser.add_argument(
        "--env-name",
        type=str,
        default="MiniGrid-Empty-8x8-v0",
        help="MiniGrid environment name",
    )
    parser.add_argument(
        "--total-timesteps", type=int, default=100000, help="Total training timesteps"
    )
    parser.add_argument(
        "--device", type=str, default="cpu", help="Device to use (cpu, cuda)"
    )

    args = parser.parse_args()

    # Start training
    train_ppo(args.env_name, args.total_timesteps, args.device)


if __name__ == "__main__":
    main()
