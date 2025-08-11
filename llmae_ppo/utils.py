"""
Utility functions for PPO training and visualization
"""

from typing import List

import os
import random
import re

import numpy as np
import torch
import yaml
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
    axs[0].set_ylabel("Mean Average Return")
    axs[0].set_title(
        f"Mean Average Return vs. Steps: PPO (Seed: {seed}, Env: {env_name})"
    )
    axs[0].grid(True)
    axs[0].legend()

    # add space between the two plots
    fig.subplots_adjust(hspace=0.5)

    axs[1].plot(
        steps,
        success_rates,
        label=f"PPO (Seed: {seed})",
    )
    axs[1].set_xlabel("Steps")
    axs[1].set_ylabel("Mean Success Rate")
    axs[1].set_title(
        f"Mean Success Rate vs. Steps: PPO (Seed: {seed}, Env: {env_name})"
    )
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


# Visualization helper
def extract_step(path: str) -> int:
    m = re.search(r"_step(\d+)\.npz$", os.path.basename(path))
    return int(m.group(1)) if m else -1


def run_dir_from_npz(path: str) -> str:
    parts = os.path.normpath(path).split(os.sep)
    if "eval_logs" in parts:
        return os.sep.join(parts[: parts.index("eval_logs")])
    return os.path.dirname(path)


def read_seed(run_dir: str) -> int:
    cfg = os.path.join(run_dir, ".hydra", "config.yaml")
    with open(cfg, "r") as f:
        data = yaml.safe_load(f)
    if "seed" in data:
        return int(data["seed"])
    if "train" in data and "seed" in data["train"]:
        return int(data["train"]["seed"])
    raise KeyError(f"Seed not found in {cfg}")
