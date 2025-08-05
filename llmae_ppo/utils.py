"""
Utility functions for PPO training.
"""

from typing import List

import os
import random

import numpy as np
import torch
from matplotlib import pyplot as plt


def set_seed(seed: int) -> None:
    """
    Set random seed for reproducibility across all libraries.

    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def plot_and_save_results(
    steps: List[int],
    average_returns: List[float],
    success_rates: List[float],
    seed: int,
    env_name: str,
    save_dir: str = "results",
) -> None:
    """
    Plot training results and save to file.

    Args:
        steps: List of training steps
        average_returns: List of average returns
        success_rates: List of success rates
        seed: Random seed used for training
        env_name: Environment name
        save_dir: Directory to save plot
    """
    fig, axs = plt.subplots(2, figsize=(10, 6))

    axs[0].plot(
        steps,
        average_returns,
        label=f"PPO (Seed: {seed})",
    )
    axs[0].set_xlabel("Steps")
    axs[0].set_ylabel("Average Return")
    axs[0].set_title(f"Average Return vs. Steps: PPO (Seed: {seed}, Env: {env_name})")
    axs[0].grid(True)
    axs[0].legend()

    axs[1].plot(
        steps,
        success_rates,
        label=f"PPO (Seed: {seed})",
    )
    axs[1].set_xlabel("Steps")
    axs[1].set_ylabel("Success Rate")
    axs[1].set_title(f"Success Rate vs. Steps: PPO (Seed: {seed}, Env: {env_name})")
    axs[1].grid(True)
    axs[1].legend()

    # Create results directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)

    # Save the plot as PNG
    plot_path = os.path.join(
        save_dir,
        f"training_results_{env_name}.png",
    )
    plt.savefig(plot_path)
    plt.close()
    print(f"Plot saved to: {plot_path}")
