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
    seed: int,
    env_name: str,
    save_dir: str = "results",
) -> None:
    """
    Plot training results and save to file.

    Args:
        steps: List of training steps
        average_returns: List of average returns
        seed: Random seed used for training
        env_name: Environment name
        save_dir: Directory to save plot
    """
    plt.figure(figsize=(10, 6))
    plt.plot(
        steps,
        average_returns,
        label=f"PPO (Seed: {seed})",
    )
    plt.xlabel("Steps")
    plt.ylabel("Average Return")
    plt.title(f"Average Return vs. Steps: PPO (Seed: {seed}, Env: {env_name})")
    plt.grid(True)
    plt.legend()

    # Create results directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)

    # Save the plot as PNG
    plot_path = os.path.join(
        save_dir,
        f"average_return_vs_frames_{env_name}.png",
    )
    plt.savefig(plot_path)
    plt.close()
    print(f"Plot saved to: {plot_path}")
