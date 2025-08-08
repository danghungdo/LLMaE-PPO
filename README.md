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
- **Weights**: If option 3 (Train and save weights) is selected during training, weights of actor and critic will be saved to `pretrained_weights/actor.pth` and `pretrained_weights/critic.pth` once the training is finished.

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

## Reproducing the Results

### Normal PPO

For our regular PPO results, we run:

```bash
python llmae_ppo/train.py env_name=MiniGrid-Doorkey-8x8-v0 total_timesteps=1000000 seed=0,7,42,69,73,666,888,9001,314159,1234567 -m
```
When prompted we choose option 1 (Train from scratch). This will train and save the results for each seed.
## Transfer Learning PPO

For our Transfer Learning results, we first run:

```bash
python llmae_ppo/train.py env_name=MiniGrid-Unlock-v0 total_timesteps=250000 seed=0
```
When prompted we choose option 3 (Train and save weights). This will pre-train the model on the *Minigrid-Unlock* enviroment and save the weights at the end of the run.

Afterwards we follow it up with the same command as the **Normal PPO**. This time when prompted we select option 2 (Load pre-trained weights and fine-tune). This triggers a followup propt asking for the path. We input the path to the weights we previously generated and let it train.

!!!Important Step: Inside on **train.py** there is a line *final_seed = 1234567*. This will need to be edited in case of different seed usage. Otherwise the multirun_choice won't get deleted once the run ends.
