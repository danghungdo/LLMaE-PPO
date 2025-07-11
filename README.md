# LLMaE-PPO

**LLM as Expert PPO** is a research project that explores how large language models (LLMs) can be used to initialize policies in reinforcement learning. Specifically, we use an LLM to generate expert-like trajectories from natural language descriptions of a task. These trajectories are used to pretrain a PPO (Proximal Policy Optimization) agent via imitation learning.

By starting PPO training from an LLM-informed prior, we aim to improve convergence speed and sample efficiency—especially in environments where exploration is difficult or reward signals are sparse. This method avoids the need for in-loop LLM calls, making it lightweight and scalable.

## Key Features
- One-time LLM prompt to generate demonstrations
- Behavior cloning for policy pretraining
- PPO training and evaluation in MiniGrid and MinAtar environments
- Baselines: Random initialization and transfer learning
- Evaluation via sample efficiency, return curves, and visitation heatmaps

## Project Scope
This project was developed as part of a university reinforcement learning course. It includes a complete experimental pipeline, including data generation, training, evaluation, and visualization.

## Quick Start

1. **Install dependencies:**
   ```bash
   uv sync
   ```

2. **Train PPO agent:**
   ```bash
   python main.py --env-name MiniGrid-Empty-8x8-v0 --total-timesteps 100000
   ```

3. **Evaluate and create GIF:**
   ```bash
   python evaluate.py --model-path checkpoints/ppo_final.pt --gif-path agent_trajectory.gif
   ```

## Usage

### Training
```bash
# Basic training
python main.py

# Custom environment and timesteps
python main.py --env-name MiniGrid-DoorKey-6x6-v0 --total-timesteps 200000

# GPU training
python main.py --device cuda
```

### Evaluation with GIF
```bash
# Create GIF and evaluate performance
python evaluate.py --model-path checkpoints/ppo_final.pt --gif-path my_agent.gif

# Different environment
python evaluate.py --model-path checkpoints/ppo_final.pt --env-name MiniGrid-DoorKey-6x6-v0 --gif-path doorkey_agent.gif
```

## Files

- `main.py` - Training script
- `evaluate.py` - Evaluation script with GIF generation
- `llmae_ppo/` - PPO implementation
  - `ppo.py` - PPO agent and trainer
  - `networks.py` - Policy and value networks
  - `env_wrapper.py` - MiniGrid environment wrapper
  - `utils.py` - Utility functions

## Output

- **Training**: Saves checkpoints to `checkpoints/`
- **Evaluation**: Creates GIF showing agent's trajectory and prints performance metrics

## Development

### Code Quality

This project uses automated formatting and linting:

```bash
# Format and lint code
python format.py

# Or manually:
black llmae_ppo/ main.py evaluate.py
isort llmae_ppo/ main.py evaluate.py
flake8 llmae_ppo/ main.py evaluate.py
```

### Pre-commit Hooks

Pre-commit hooks automatically format code on commit:

```bash
# Install hooks (already done if you ran uv sync)
pre-commit install

# Run hooks manually
pre-commit run --all-files
```

The hooks will automatically:
- Format code with Black
- Sort imports with isort
- Lint with flake8
- Fix trailing whitespace
- Check YAML files
