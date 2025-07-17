"""
Environment management for PPO training.
Handles environment creation, wrappers, and configuration.
"""

import gymnasium as gym
from minigrid.wrappers import FlatObsWrapper


def make_env(env_name, max_episode_steps, seed, idx, capture_video, run_name):
    """
    Create a single environment with proper wrappers and configuration.

    Args:
        env_name: Name of the environment
        max_episode_steps: Maximum steps per episode
        seed: Random seed
        idx: Environment index
        capture_video: Whether to capture video
        run_name: Name for the run (used in video recording)

    Returns:
        Function that creates the environment
    """

    def thunk():
        env = gym.make(env_name, max_episode_steps, render_mode="rgb_array")
        env.unwrapped.max_steps = max_episode_steps
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


def create_vector_env(
    env_name, max_episode_steps, seed, num_envs, capture_video, run_name
):
    """
    Create a vectorized environment for training.

    Args:
        env_name: Name of the environment
        max_episode_steps: Maximum steps per episode
        seed: Random seed
        num_envs: Number of parallel environments
        capture_video: Whether to capture video
        run_name: Name for the run

    Returns:
        Vectorized environment
    """
    envs = gym.vector.SyncVectorEnv(
        [
            make_env(env_name, max_episode_steps, seed, i, capture_video, run_name)
            for i in range(num_envs)
        ]
    )

    assert isinstance(envs.single_action_space, gym.spaces.Discrete), (
        "only discrete action space is supported"
    )

    return envs


def create_eval_env(env_name, max_episode_steps, seed, num_envs):
    """
    Create a vectorized environment for evaluation.

    Args:
        env_name: Name of the environment
        max_episode_steps: Maximum steps per episode
        seed: Random seed
        num_envs: Number of parallel environments

    Returns:
        Vectorized environment for evaluation
    """
    eval_envs = gym.vector.SyncVectorEnv(
        [
            make_env(env_name, max_episode_steps, seed, i, True, "eval")
            for i in range(num_envs)
        ]
    )

    assert isinstance(eval_envs.single_action_space, gym.spaces.Discrete), (
        "only discrete action space is supported"
    )

    return eval_envs
