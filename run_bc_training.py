#!/usr/bin/env python3
"""
Example script showing how to run behavioral cloning training with different configurations.
"""

import subprocess
import sys

def run_bc_training():
    """Run behavioral cloning training with default configuration."""
    cmd = [sys.executable, "llmae_ppo_pre-train.py"]
    subprocess.run(cmd)

def run_bc_training_custom():
    """Run behavioral cloning training with custom configuration overrides."""
    cmd = [
        sys.executable, "llmae_ppo_pre-train.py",
        "env.name=MiniGrid-DoorKey-8x8-v0",   # Change environment
        "train.num_epochs=100",               # More epochs
        "model.hidden_size=512",              # Larger network
        "train.batch_size=64",                # Larger batch size
        "model.lr=0.0005",                    # Different learning rate
        "seed=123"                            # Different seed
    ]
    subprocess.run(cmd)


if __name__ == "__main__":
    print("Running behavioral cloning training examples...")
    
    print("\n1. Default configuration:")
    run_bc_training()
    
    # print("\n2. Custom configuration:")
    # run_bc_training_custom()
