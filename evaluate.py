"""
Simple evaluation script that generates GIFs of trained PPO agents.
"""

import argparse

import numpy as np
from PIL import Image

from llmae_ppo import MiniGridWrapper, PPOAgent
from llmae_ppo.utils import set_seed


def create_gif_from_episode(env_name, agent, gif_path, max_steps=200):
    """
    Create a GIF from a single episode using a fresh environment.

    Args:
        env_name: Name of the MiniGrid environment
        agent: Trained PPO agent
        gif_path: Path to save the GIF
        max_steps: Maximum steps per episode

    Returns:
        Episode reward and length
    """
    import gymnasium as gym

    frames = []
    # Create a fresh environment for rendering
    raw_env = gym.make(env_name, render_mode="rgb_array")

    # Create our wrapper for the agent
    wrapped_env = MiniGridWrapper(env_name, flatten_obs=True, normalize_obs=True)

    # Reset both environments
    raw_obs, _ = raw_env.reset()
    wrapped_state, _ = wrapped_env.reset()

    episode_reward = 0
    episode_length = 0
    done = False

    # Capture initial frame
    try:
        img = raw_env.render()
        if img is not None:
            frames.append(Image.fromarray(img))
    except Exception as e:
        print(f"Warning: Could not capture initial frame: {e}")

    while not done and episode_length < max_steps:
        # Get action from agent using wrapped state
        action, _ = agent.get_action(wrapped_state)

        # Step both environments
        raw_obs, reward, terminated, truncated, info = raw_env.step(action)
        wrapped_state, _, _, _, _ = wrapped_env.step(action)

        done = terminated or truncated
        episode_reward += reward
        episode_length += 1

        # Capture frame from raw environment
        try:
            img = raw_env.render()
            if img is not None:
                frames.append(Image.fromarray(img))
        except Exception as e:
            print(f"Warning: Could not capture frame at step {episode_length}: {e}")

    # Close environments
    raw_env.close()
    wrapped_env.close()

    # Save as GIF
    if frames:
        try:
            frames[0].save(
                gif_path,
                save_all=True,
                append_images=frames[1:],
                duration=300,  # 300ms per frame for better visibility
                loop=0,
            )
            print(f"GIF saved: {gif_path} ({len(frames)} frames)")
        except Exception as e:
            print(f"Error saving GIF: {e}")
    else:
        print("No frames captured - GIF not created")

    return episode_reward, episode_length


def evaluate_agent(env, agent, n_episodes=10):
    """
    Evaluate agent performance over multiple episodes.

    Args:
        env: Environment to evaluate on
        agent: Trained PPO agent
        n_episodes: Number of episodes to evaluate

    Returns:
        Dictionary with evaluation metrics
    """
    episode_rewards = []
    episode_lengths = []
    success_rate = 0

    for episode in range(n_episodes):
        state, _ = env.reset()
        episode_reward = 0
        episode_length = 0
        done = False

        while not done:
            action, _ = agent.get_action(state)
            state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            episode_reward += reward
            episode_length += 1

        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)

        # Check if episode was successful (reached goal)
        if episode_reward > 0:
            success_rate += 1

        if (episode + 1) % 5 == 0:
            print(f"Episode {episode + 1}/{n_episodes} completed")

    success_rate = success_rate / n_episodes

    return {
        "mean_reward": np.mean(episode_rewards),
        "std_reward": np.std(episode_rewards),
        "mean_length": np.mean(episode_lengths),
        "std_length": np.std(episode_lengths),
        "success_rate": success_rate,
    }


def main():
    """Main evaluation function."""
    parser = argparse.ArgumentParser(
        description="Evaluate trained PPO agent and create GIF"
    )
    parser.add_argument(
        "--model-path", type=str, required=True, help="Path to trained model checkpoint"
    )
    parser.add_argument(
        "--env-name",
        type=str,
        default="MiniGrid-Empty-8x8-v0",
        help="MiniGrid environment name",
    )
    parser.add_argument(
        "--n-episodes", type=int, default=10, help="Number of episodes to evaluate"
    )
    parser.add_argument(
        "--gif-path",
        type=str,
        default="agent_trajectory.gif",
        help="Path to save trajectory GIF",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for evaluation"
    )

    args = parser.parse_args()

    # Set seed
    set_seed(args.seed)

    print(f"Evaluating agent: {args.model_path}")
    print(f"Environment: {args.env_name}")
    print(f"Episodes: {args.n_episodes}")

    # Create environment
    env = MiniGridWrapper(args.env_name, flatten_obs=True, normalize_obs=True)
    state_dim = env.observation_space.shape
    action_dim = env.action_space.n

    # Load agent
    agent = PPOAgent(state_dim, action_dim, device="cpu")
    agent.load(args.model_path)
    print("Agent loaded successfully!")

    # Create GIF of one episode
    print(f"\nCreating GIF: {args.gif_path}")
    episode_reward, episode_length = create_gif_from_episode(
        args.env_name, agent, args.gif_path
    )
    print(f"GIF episode - Reward: {episode_reward:.3f}, Length: {episode_length}")

    # Evaluate performance over multiple episodes
    print(f"\nEvaluating over {args.n_episodes} episodes...")
    results = evaluate_agent(env, agent, args.n_episodes)

    # Print results
    print("\n" + "=" * 50)
    print("EVALUATION RESULTS")
    print("=" * 50)
    print(f"Mean reward: {results['mean_reward']:.3f} ± {results['std_reward']:.3f}")
    print(f"Mean length: {results['mean_length']:.1f} ± {results['std_length']:.1f}")
    print(f"Success rate: {results['success_rate']:.2%}")
    print("=" * 50)

    env.close()


if __name__ == "__main__":
    main()
