"""
Environment wrapper for MiniGrid environments.
"""

import gymnasium as gym
import minigrid  # noqa: F401
import numpy as np
from gymnasium import spaces


class MiniGridWrapper(gym.Wrapper):
    """
    Wrapper for MiniGrid environments to make them compatible with PPO.
    """

    def __init__(
        self,
        env_name="MiniGrid-Empty-8x8-v0",
        flatten_obs=True,
        normalize_obs=True,
        max_steps=None,
    ):
        """
        Initialize MiniGrid wrapper.

        Args:
            env_name: Name of the MiniGrid environment
            flatten_obs: Whether to flatten the observation
            normalize_obs: Whether to normalize observations to [0, 1]
            max_steps: Maximum number of steps per episode
        """
        env = gym.make(env_name)
        super().__init__(env)

        self.flatten_obs = flatten_obs
        self.normalize_obs = normalize_obs
        self.max_steps = max_steps or getattr(env, "max_steps", 1000)
        self.step_count = 0

        # Get original observation space
        original_obs_space = env.observation_space["image"]

        if flatten_obs:
            # Flatten the observation space
            flat_dim = np.prod(original_obs_space.shape)
            self.observation_space = spaces.Box(
                low=0.0 if normalize_obs else original_obs_space.low.min(),
                high=1.0 if normalize_obs else original_obs_space.high.max(),
                shape=(flat_dim,),
                dtype=np.float32,
            )
        else:
            # Keep original shape but potentially normalize
            if normalize_obs:
                self.observation_space = spaces.Box(
                    low=0.0, high=1.0, shape=original_obs_space.shape, dtype=np.float32
                )
            else:
                self.observation_space = original_obs_space

    def reset(self, **kwargs):
        """Reset the environment."""
        obs, info = self.env.reset(**kwargs)
        self.step_count = 0
        return self._process_observation(obs), info

    def step(self, action):
        """Take a step in the environment."""
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.step_count += 1

        # Check if max steps reached
        if self.max_steps and self.step_count >= self.max_steps:
            truncated = True

        return self._process_observation(obs), reward, terminated, truncated, info

    def render(self):
        """Render the environment and return RGB array."""
        try:
            # Get the current observation which contains the image
            obs = self.env.get_wrapper_attr("get_obs")()
            if "image" in obs:
                # Convert from MiniGrid's encoding to RGB
                img = obs["image"]
                # MiniGrid images need to be converted to proper RGB
                if img.max() <= 10:  # MiniGrid uses values 0-10
                    img = (img * 25.5).astype("uint8")  # Convert to 0-255 range
                return img
            return None
        except Exception as e:
            print(f"Render error: {e}")
            return None

    def _process_observation(self, obs):
        """Process the observation according to wrapper settings."""
        # Extract image from observation dict
        image = obs["image"]

        if self.normalize_obs:
            # Normalize to [0, 1]
            image = image.astype(np.float32) / 10.0  # MiniGrid uses values 0-10

        if self.flatten_obs:
            # Flatten the observation
            image = image.flatten()

        return image


class MultiEnvWrapper:
    """
    Wrapper for running multiple MiniGrid environments in parallel.
    """

    def __init__(self, env_name="MiniGrid-Empty-8x8-v0", num_envs=4, **wrapper_kwargs):
        """
        Initialize multiple environments.

        Args:
            env_name: Name of the MiniGrid environment
            num_envs: Number of parallel environments
            **wrapper_kwargs: Arguments for MiniGridWrapper
        """
        self.num_envs = num_envs
        self.envs = [
            MiniGridWrapper(env_name, **wrapper_kwargs) for _ in range(num_envs)
        ]
        self.observation_space = self.envs[0].observation_space
        self.action_space = self.envs[0].action_space

    def reset(self):
        """Reset all environments."""
        observations = []
        infos = []
        for env in self.envs:
            obs, info = env.reset()
            observations.append(obs)
            infos.append(info)
        return np.array(observations), infos

    def step(self, actions):
        """Take steps in all environments."""
        observations = []
        rewards = []
        terminateds = []
        truncateds = []
        infos = []

        for i, (env, action) in enumerate(zip(self.envs, actions)):
            obs, reward, terminated, truncated, info = env.step(action)

            # Auto-reset if episode is done
            if terminated or truncated:
                obs, reset_info = env.reset()
                info["final_observation"] = obs
                info.update(reset_info)

            observations.append(obs)
            rewards.append(reward)
            terminateds.append(terminated)
            truncateds.append(truncated)
            infos.append(info)

        return (
            np.array(observations),
            np.array(rewards),
            np.array(terminateds),
            np.array(truncateds),
            infos,
        )

    def close(self):
        """Close all environments."""
        for env in self.envs:
            env.close()


def make_env(env_name="MiniGrid-Empty-8x8-v0", **kwargs):
    """
    Factory function to create MiniGrid environment.

    Args:
        env_name: Name of the MiniGrid environment
        **kwargs: Additional arguments for MiniGridWrapper

    Returns:
        Wrapped MiniGrid environment
    """
    return MiniGridWrapper(env_name, **kwargs)


def get_available_envs():
    """Get list of available MiniGrid environments."""
    available_envs = []

    # Common MiniGrid environments
    common_envs = [
        "MiniGrid-Empty-5x5-v0",
        "MiniGrid-Empty-8x8-v0",
        "MiniGrid-Empty-16x16-v0",
        "MiniGrid-FourRooms-v0",
        "MiniGrid-DoorKey-5x5-v0",
        "MiniGrid-DoorKey-6x6-v0",
        "MiniGrid-DoorKey-8x8-v0",
        "MiniGrid-MultiRoom-N2-S4-v0",
        "MiniGrid-MultiRoom-N4-S5-v0",
        "MiniGrid-MultiRoom-N6-v0",
        "MiniGrid-Fetch-5x5-N2-v0",
        "MiniGrid-Fetch-6x6-N2-v0",
        "MiniGrid-Fetch-8x8-N3-v0",
        "MiniGrid-GoToDoor-5x5-v0",
        "MiniGrid-GoToDoor-6x6-v0",
        "MiniGrid-GoToDoor-8x8-v0",
        "MiniGrid-PutNear-6x6-N2-v0",
        "MiniGrid-PutNear-8x8-N3-v0",
        "MiniGrid-RedBlueDoors-6x6-v0",
        "MiniGrid-RedBlueDoors-8x8-v0",
        "MiniGrid-MemoryS17Random-v0",
        "MiniGrid-MemoryS13Random-v0",
        "MiniGrid-MemoryS13-v0",
        "MiniGrid-MemoryS11-v0",
        "MiniGrid-MemoryS9-v0",
        "MiniGrid-MemoryS7-v0",
        "MiniGrid-LockedRoom-v0",
        "MiniGrid-KeyCorridor-S3R1-v0",
        "MiniGrid-KeyCorridor-S3R2-v0",
        "MiniGrid-KeyCorridor-S3R3-v0",
        "MiniGrid-KeyCorridor-S4R3-v0",
        "MiniGrid-KeyCorridor-S5R3-v0",
        "MiniGrid-KeyCorridor-S6R3-v0",
        "MiniGrid-Unlock-v0",
        "MiniGrid-UnlockPickup-v0",
        "MiniGrid-BlockedUnlockPickup-v0",
        "MiniGrid-ObstructedMaze-1Dl-v0",
        "MiniGrid-ObstructedMaze-1Dlh-v0",
        "MiniGrid-ObstructedMaze-1Dlhb-v0",
        "MiniGrid-ObstructedMaze-2Dl-v0",
        "MiniGrid-ObstructedMaze-2Dlh-v0",
        "MiniGrid-ObstructedMaze-2Dlhb-v0",
        "MiniGrid-ObstructedMaze-Full-v0",
        "MiniGrid-DistShift1-v0",
        "MiniGrid-DistShift2-v0",
        "MiniGrid-LavaGapS5-v0",
        "MiniGrid-LavaGapS6-v0",
        "MiniGrid-LavaGapS7-v0",
        "MiniGrid-LavaCrossingS9N1-v0",
        "MiniGrid-LavaCrossingS9N2-v0",
        "MiniGrid-LavaCrossingS9N3-v0",
        "MiniGrid-LavaCrossingS11N5-v0",
        "MiniGrid-SimpleCrossingS9N1-v0",
        "MiniGrid-SimpleCrossingS9N2-v0",
        "MiniGrid-SimpleCrossingS9N3-v0",
        "MiniGrid-SimpleCrossingS11N5-v0",
        "MiniGrid-Dynamic-Obstacles-5x5-v0",
        "MiniGrid-Dynamic-Obstacles-Random-5x5-v0",
        "MiniGrid-Dynamic-Obstacles-6x6-v0",
        "MiniGrid-Dynamic-Obstacles-Random-6x6-v0",
        "MiniGrid-Dynamic-Obstacles-8x8-v0",
        "MiniGrid-Dynamic-Obstacles-16x16-v0",
    ]

    # Check which environments are actually available
    for env_name in common_envs:
        try:
            env = gym.make(env_name)
            env.close()
            available_envs.append(env_name)
        except Exception:
            pass

    return available_envs
