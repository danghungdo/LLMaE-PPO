# LLMaE-PPO

**LLM as Expert PPO** is a research project that explores how large language models (LLMs) can be used to initialize policies in reinforcement learning. Specifically, we use an LLM to generate expert-like trajectories from natural language descriptions of a task. These trajectories are used to pretrain a PPO (Proximal Policy Optimization) agent via imitation learning.

By starting PPO training from an LLM-informed prior, we aim to improve convergence speed and sample efficiency—especially in environments where exploration is difficult or reward signals are sparse. This method avoids the need for in-loop LLM calls, making it lightweight and scalable.

## Key Features
- One-time LLM prompt to generate demonstrations
- Behavior cloning for policy pretraining
- PPO training and evaluation in MiniGrid environment
- Baselines: Random initialization and transfer learning
- Evaluation via sample efficiency, return curves, and visitation heatmaps

## Project Scope
This project was developed as part of a university reinforcement learning course. It includes a complete experimental pipeline, including data generation, training, evaluation, and visualization.

## Quick Start

1. **Install dependencies:**
   ```bash
   make install
   ```
   Or just sync dependencies:
   ```bash
   make sync
   ```

2. **Train PPO agent with config in `llmae_ppo/configs/ppo.yaml`:**
   ```bash
   python llmae_ppo/train.py
   ```

### Configuration

The project uses Hydra for configuration management. The default configuration is in `llmae_ppo/configs/ppo.yaml`. You can override any parameter using command line arguments:

```bash
# Override multiple parameters
python llmae_ppo/train.py env_name=MiniGrid-Empty-8x8-v0 total_timesteps=100000 learning_rate=0.0003
```

## Files

- `llmae_ppo/` - Main package directory
  - `train.py` - Training script with Hydra configuration
  - `trainer.py` - PPO trainer implementation
  - `ppo_agent.py` - PPO agent implementation
  - `networks.py` - Policy and value networks
  - `env.py` - Environment wrapper for MiniGrid
  - `utils.py` - Utility functions
  - `agent/` - Agent-related modules
    - `abstract_agent.py` - Abstract agent interface
    - `buffer.py` - Experience buffer implementation
  - `configs/` - Configuration files
    - `ppo.yaml` - Default PPO configuration

## Output

- **Training**: By default, results are saved to `outputs/YYYY-MM-DD/HH-MM-SS/`
  - Training logs
  - Performance plots (e.g., `average_return_vs_frames_*.png`)
  - Video recordings of agent episodes
- **Evaluation**: By default, evaluation videos are saved to `outputs/.../videos/eval/`
- **TensorBoard**: Logs are saved to `outputs/.../runs/` directory

## TensorBoard

Start TensorBoard for monitoring training:
```bash
tensorboard --logdir=runs
```

For multirun comparison:
```bash
python llmae_ppo/train.py seed=0,1,2 -m
tensorboard --logdir=runs
```

Collect logging data for plotting:
```bash
python plot_sample_efficiency.py
```

## Available Make Commands

```bash
# Show all available commands
make help

# Install dependencies and pre-commit hooks
make install

# Sync dependencies only
make sync

# Format code with ruff and isort
make format

# Check code for issues (dry run)
make check

# Run pre-commit hooks
make pre-commit
```

## Development

### Code Quality

This project uses automated formatting and linting with Ruff:

```bash
# Format and lint code
make format

# Check code for issues (without fixing)
make check
```

### Pre-commit Hooks

Pre-commit hooks automatically format code on commit:

```bash
# Install pre-commit hooks if not already installed
pre-commit install

# Run hooks manually
make pre-commit
```

The hooks will automatically:
- Format code with Ruff
- Sort imports with isort
