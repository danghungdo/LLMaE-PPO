import pickle
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import gymnasium as gym
from minigrid.wrappers import FlatObsWrapper
from sklearn.model_selection import train_test_split
import hydra
from omegaconf import DictConfig


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    """Initialize layer weights using orthogonal initialization."""
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

# Custom Dataset for state-action pairs
class MiniGridDataset(Dataset):
    def __init__(self, state_action_pairs):
        """
        Initialize dataset with state-action pairs.
        
        Args:
            state_action_pairs: List of dictionaries with 'state' and 'action' keys
        """
        if not state_action_pairs:
            raise ValueError("state_action_pairs cannot be empty")
        
        # Validate data format
        for i, pair in enumerate(state_action_pairs):
            if not isinstance(pair, dict) or 'state' not in pair or 'action' not in pair:
                raise ValueError(f"Invalid data format at index {i}. Expected dict with 'state' and 'action' keys")
        
        # Convert to tensors for better performance
        try:
            states = [pair['state'] for pair in state_action_pairs]
            actions = [pair['action'] for pair in state_action_pairs]
            
            self.states = torch.tensor(np.array(states, dtype=np.float32))
            self.actions = torch.tensor(np.array(actions, dtype=np.int64))
            
        except Exception as e:
            raise ValueError(f"Error converting data to tensors: {e}")
        
        # Validate shapes
        if len(self.states) != len(self.actions):
            raise ValueError("Mismatch between number of states and actions")

    def __len__(self):
        return len(self.states)

    def __getitem__(self, idx):
        if idx >= len(self.states):
            raise IndexError(f"Index {idx} out of range for dataset of size {len(self.states)}")
        return self.states[idx], self.actions[idx]
    
    def get_state_shape(self):
        """Return the shape of a single state."""
        return self.states[0].shape if len(self.states) > 0 else None
    
    def get_num_actions(self):
        """Return the number of actions in the dataset."""
        return len(torch.unique(self.actions))

# Neural Network Model - Same architecture as PolicyNetwork
class BehavioralCloningModel(nn.Module):
    """
    Behavioral Cloning Model for MiniGrid environments.
    Uses the same architecture as PolicyNetwork from PPO for consistency.
    """

    def __init__(self, input_size, output_size, hidden_size=64):
        """
        Initialize Behavioral Cloning Model.

        Args:
            input_size (int): Size of the input state vector
            output_size (int): Number of possible actions
            hidden_size (int): Size of hidden layers (default: 64, same as PPO)
        """
        super(BehavioralCloningModel, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size

        # Same architecture as PolicyNetwork
        self.actor = nn.Sequential(
            layer_init(nn.Linear(input_size, hidden_size)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_size, hidden_size)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_size, output_size), std=0.01),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network."""
        logits = self.actor(state)
        return logits

def load_pkl_files(pkl_dir):
    """
    Load state-action pairs from all PKL files in directory
    
    Args:
        pkl_dir (str): Directory containing PKL files with state-action pairs
        
    Returns:
        list: Combined list of state-action pairs from all files
    """
    all_pairs = []
    for pkl_file in os.listdir(pkl_dir):
        if pkl_file.endswith('.pkl'):
            file_path = os.path.join(pkl_dir, pkl_file)
            try:
                with open(file_path, 'rb') as f:
                    state_action_pairs = pickle.load(f)
                    all_pairs.extend(state_action_pairs)
                print(f"Loaded {len(state_action_pairs)} pairs from {pkl_file}")
            except Exception as e:
                print(f"Error loading {pkl_file}: {e}")
                continue
    
    print(f"Total state-action pairs loaded: {len(all_pairs)}")
    if not all_pairs:
        raise ValueError("No data loaded from PKL files")
        
    return all_pairs

def get_model_dimensions(data_pairs, env_name, max_steps):
    """
    Dynamically determine input and output dimensions for the model.

    Args:
        data_pairs: List of state-action pairs
        env_name: Name of the MiniGrid environment

    Returns:
        tuple: (input_size, output_size)
    """
    if not data_pairs:
        raise ValueError("No data pairs provided")

    # Get input size from state shape
    sample_state = data_pairs[0]['state']
    if isinstance(sample_state, np.ndarray):
        input_size = sample_state.shape[0] if len(sample_state.shape) == 1 else int(np.prod(sample_state.shape))
    else:
        input_size = len(sample_state)

    # Get output size from environment action space
    env = gym.make(env_name, max_steps=max_steps)
    env = FlatObsWrapper(env)
    output_size = env.action_space.n
    env.close()

    return input_size, output_size

# Enhanced evaluation function for multiple episodes
def evaluate_bc_model(model, env_name, max_steps, num_episodes, start_seed=0):
    """
    Evaluate BC model over multiple episodes and return comprehensive metrics.

    Args:
        model: BC model to evaluate
        env_name: Environment name
        max_steps: Maximum steps per episode
        num_episodes: Number of episodes to evaluate
        start_seed: Starting seed for episodes

    Returns:
        dict: Contains mean_return, std_return, success_rate, mean_episode_length, std_episode_length
    """
    model.eval()
    returns = []
    episode_lengths = []
    successes = 0

    for episode in range(num_episodes):
        env = gym.make(env_name, max_steps=max_steps)
        env = FlatObsWrapper(env)
        obs, _ = env.reset(seed=0)

        episode_return = 0
        steps = 0
        done = False

        while not done and steps < max_steps:
            state = obs.astype(np.float32)
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)

            with torch.no_grad():
                action_probs = model(state_tensor)
                action_probs_softmax = torch.softmax(action_probs, dim=1)
                action = torch.multinomial(action_probs_softmax, 1).item()

            obs, reward, terminated, truncated, _ = env.step(action)
            episode_return += reward
            steps += 1
            done = terminated or truncated

            if terminated and reward > 0:
                successes += 1

        returns.append(episode_return)
        episode_lengths.append(steps)
        env.close()

    return {
        'mean_return': np.mean(returns),
        'std_return': np.std(returns),
        'success_rate': successes / num_episodes,
        'mean_episode_length': np.mean(episode_lengths),
        'std_episode_length': np.std(episode_lengths)
    }

# Enhanced training function with periodic evaluation
def train_bc_enhanced(model, train_loader, test_loader, cfg, device='cpu'):
    """
    Enhanced BC training with train-test validation, then training on full dataset.

    Args:
        model: BC model to train
        train_loader: Training data loader
        test_loader: Test data loader for validation
        cfg: Configuration object
        device: Training device

    Returns:
        tuple: (trained_model, training_metrics)
    """
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=cfg.model.lr)

    training_metrics = []
    global_step = 0

    print("Phase 1: Training with train-test split for validation")
    print("="*60)

    for epoch in range(cfg.train.num_epochs):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        # Training phase
        for states, actions in train_loader:
            states, actions = states.to(device), actions.to(device)
            optimizer.zero_grad()
            outputs = model(states)
            loss = criterion(outputs, actions)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * states.size(0)
            _, predicted = torch.max(outputs, 1)
            train_total += actions.size(0)
            train_correct += (predicted == actions).sum().item()

            global_step += states.size(0)

            # Periodic evaluation during training
            if hasattr(cfg.train, 'eval_interval') and global_step % cfg.train.eval_interval == 0:
                eval_metrics = evaluate_bc_model(
                    model,
                    cfg.env.name,
                    cfg.train.eval_max_steps,
                    cfg.train.eval_episodes if hasattr(cfg.train, 'eval_episodes') else 5
                )

                print(f"Step {global_step}: "
                      f"Mean Return: {eval_metrics['mean_return']:.3f} ± {eval_metrics['std_return']:.3f}, "
                      f"Success Rate: {eval_metrics['success_rate']:.3f}, "
                      f"Avg Length: {eval_metrics['mean_episode_length']:.1f}")

                training_metrics.append({
                    'step': global_step,
                    'epoch': epoch,
                    **eval_metrics
                })

                model.train()  # Switch back to training mode

        # Calculate average training metrics
        train_loss /= train_total
        train_accuracy = train_correct / train_total

        # Evaluate on test set
        model.eval()
        test_loss = 0.0
        test_correct = 0
        test_total = 0

        with torch.no_grad():
            for states, actions in test_loader:
                states, actions = states.to(device), actions.to(device)
                outputs = model(states)
                loss = criterion(outputs, actions)

                test_loss += loss.item() * states.size(0)
                _, predicted = torch.max(outputs, 1)
                test_total += actions.size(0)
                test_correct += (predicted == actions).sum().item()

        test_loss /= test_total
        test_accuracy = test_correct / test_total

        if cfg.logging.verbose and (epoch + 1) % cfg.logging.print_freq == 0:
            print(f'Epoch {epoch+1}/{cfg.train.num_epochs}, Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.4f}, Test Loss: {test_loss:.4f}, Test Acc: {test_accuracy:.4f}')

    return model, training_metrics

# Function to train on full dataset
def train_bc_full_dataset(model, full_loader, cfg, device='cpu'):
    """
    Train BC model on the full dataset after initial train-test validation.

    Args:
        model: BC model to train
        full_loader: Full dataset loader
        cfg: Configuration object
        device: Training device

    Returns:
        trained_model
    """
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=cfg.model.lr)

    print("\nPhase 2: Training on full dataset")
    print("="*60)

    for epoch in range(cfg.train.num_epochs):
        model.train()
        epoch_loss = 0.0
        epoch_correct = 0
        epoch_total = 0

        for states, actions in full_loader:
            states, actions = states.to(device), actions.to(device)
            optimizer.zero_grad()
            outputs = model(states)
            loss = criterion(outputs, actions)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * states.size(0)
            _, predicted = torch.max(outputs, 1)
            epoch_total += actions.size(0)
            epoch_correct += (predicted == actions).sum().item()

        # Calculate average metrics
        avg_loss = epoch_loss / epoch_total
        accuracy = epoch_correct / epoch_total

        if cfg.logging.verbose and (epoch + 1) % cfg.logging.print_freq == 0:
            print(f'Full Dataset Epoch {epoch+1}/{cfg.train.num_epochs}, '
                  f'Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}')

    return model

@hydra.main(
    config_path="llmae_ppo/configs",
    config_name="bc",
    version_base="1.1",
)
def main(cfg: DictConfig):
    """
    Main behavioral cloning training function.

    Args:
        cfg: Hydra configuration
    """
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() and cfg.cuda else 'cpu')
    print(f"Using device: {device}")

    # Set random seed
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    # Load data
    all_pairs = load_pkl_files(cfg.data.training_data_dir)

    # Train-test split
    train_pairs, test_pairs = train_test_split(
        all_pairs,
        test_size=cfg.data.test_size,
        random_state=cfg.data.random_state
    )

    # Create datasets and dataloaders
    train_dataset = MiniGridDataset(train_pairs)
    test_dataset = MiniGridDataset(test_pairs)
    train_loader = DataLoader(train_dataset, batch_size=cfg.train.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=cfg.train.batch_size, shuffle=False)

    # Get input and output sizes dynamically
    input_size, output_size = get_model_dimensions(all_pairs, cfg.env.name, cfg.env.max_episode_steps)
    print(f"Model configuration: input_size={input_size}, output_size={output_size}")

    # Initialize model with dynamic sizes
    model = BehavioralCloningModel(
        input_size=input_size,
        output_size=output_size,
        hidden_size=cfg.model.hidden_size
    ).to(device)

    # Phase 1: Train with train-test split for validation
    model, training_metrics = train_bc_enhanced(
        model,
        train_loader,
        test_loader,
        cfg,
        device=device
    )

    # Phase 2: Train on full dataset
    full_dataset = MiniGridDataset(all_pairs)
    full_loader = DataLoader(full_dataset, batch_size=cfg.train.batch_size, shuffle=True)

    model = train_bc_full_dataset(
        model,
        full_loader,
        cfg,
        device=device
    )

    # Save model
    torch.save(model.actor.state_dict(), cfg.train.save_path)
    print(f"Model saved to {cfg.train.save_path}")

    # Final comprehensive evaluation
    final_eval = evaluate_bc_model(
        model,
        cfg.env.name,
        cfg.train.eval_max_steps,
        cfg.train.eval_episodes * 2 if hasattr(cfg.train, 'eval_episodes') else 5  # More episodes for final eval
    )

    print("\n🎯 Final Evaluation Results:")
    print(f"Mean Return: {final_eval['mean_return']:.3f} ± {final_eval['std_return']:.3f}")
    print(f"Success Rate: {final_eval['success_rate']:.3f}")
    print(f"Mean Episode Length: {final_eval['mean_episode_length']:.1f} ± {final_eval['std_episode_length']:.1f}")

    # Print training metrics summary if available
    if training_metrics:
        print("\n📊 Training Metrics Summary:")
        print(f"Total evaluation points: {len(training_metrics)}")
        best_eval = max(training_metrics, key=lambda x: x['success_rate'])
        print(f"Best success rate during training: {best_eval['success_rate']:.3f} at step {best_eval['step']}")

if __name__ == '__main__':
    main()