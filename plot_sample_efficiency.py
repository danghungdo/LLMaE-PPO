#!/usr/bin/env python3

import os
import glob
import pandas as pd
import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from rliable import library as rly
from rliable import plot_utils
import matplotlib.pyplot as plt

def read_all_logs(multirun_dir):
    """
    Read all TensorBoard logs from multirun directory.
    
    Args:
        multirun_dir: Path to multirun directory containing seed subdirectories
        
    Returns:
        Dictionary with structure: {seed: {metric_name: DataFrame}}
    """
    all_data = {}
    
    # Find all seed directories
    seed_dirs = glob.glob(os.path.join(multirun_dir, "*"))
    seed_dirs = [d for d in seed_dirs if os.path.isdir(d) and d.split(os.sep)[-1].isdigit()]
    
    print(f"Found {len(seed_dirs)} seed directories")
    
    for seed_dir in seed_dirs:
        seed = int(os.path.basename(seed_dir))
        print(f"Reading logs for seed {seed}")
        
        # Look for TensorBoard event files
        runs_dir = os.path.join(seed_dir, "runs")
        if not os.path.exists(runs_dir):
            continue
            
        event_files = glob.glob(os.path.join(runs_dir, "*", "events.out.tfevents.*"))
        
        if not event_files:
            continue
            
        try:
            # Read TensorBoard events
            ea = EventAccumulator(event_files[0])
            ea.Reload()
            
            seed_data = {}
            for tag in ea.Tags()['scalars']:
                scalar_events = ea.Scalars(tag)
                steps = [event.step for event in scalar_events]
                values = [event.value for event in scalar_events]
                seed_data[tag] = pd.DataFrame({'step': steps, 'value': values})
            
            all_data[seed] = seed_data
            print(f"  Loaded {len(seed_data)} metrics")
            
        except Exception as e:
            print(f"  Error reading logs for seed {seed}: {e}")
    
    return all_data

def plot_sample_efficiency(all_data, metric_name, save_path=None):
    """
    Plot sample efficiency curve using rliable.
    
    Args:
        all_data: Dictionary from read_all_logs()
        metric_name: Name of metric to plot (e.g., "charts/average_return")
        save_path: Optional path to save plot
    """
    # Prepare data for rliable
    all_steps = set()
    for seed_data in all_data.values():
        if metric_name in seed_data:
            all_steps.update(seed_data[metric_name]['step'].values)
    
    if not all_steps:
        raise ValueError(f"No data found for metric '{metric_name}'")
    
    sorted_steps = sorted(all_steps)
    seeds = sorted(all_data.keys())
    
    # Create score matrix: (n_seeds, n_steps)
    scores = np.zeros((len(seeds), len(sorted_steps)))
    
    for i, seed in enumerate(seeds):
        if metric_name in all_data[seed]:
            df = all_data[seed][metric_name]
            # Interpolate to common steps
            interpolated_values = np.interp(sorted_steps, df['step'], df['value'])
            scores[i, :] = interpolated_values
    
    # Create score dictionary for rliable
    score_dict = {"Algorithm": scores}
    
    # Get bootstrap confidence intervals
    point_estimates, interval_estimates = rly.get_interval_estimates(
        score_dict, 
        lambda x: np.mean(x, axis=0),  # Mean across seeds
        reps=10000
    )
    
    # Create plot
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    # Plot using rliable
    plot_utils.plot_sample_efficiency_curve(
        frames=np.array(sorted_steps),
        point_estimates=point_estimates,
        interval_estimates=interval_estimates,
        algorithms=["Algorithm"],
        xlabel='Training Steps',
        ylabel=metric_name,
        ax=ax
    )
    
    plt.title(f'Sample Efficiency Curve: {metric_name}')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    
    plt.show()
    
    # Print final performance statistics
    final_scores = scores[:, -1]
    print(f"\nFinal performance statistics for {metric_name}:")
    print(f"  Mean: {np.mean(final_scores):.4f}")
    print(f"  Std: {np.std(final_scores):.4f}")
    print(f"  Min: {np.min(final_scores):.4f}")
    print(f"  Max: {np.max(final_scores):.4f}")
    
    return scores, sorted_steps

# Example usage
if __name__ == "__main__":
    # Read all logs
    multirun_dir = "multirun/2025-08-06/12-57-49"
    all_data = read_all_logs(multirun_dir)
    
    # Show available metrics
    if all_data:
        sample_seed = list(all_data.keys())[0]
        print(f"\nAvailable metrics:")
        for metric in all_data[sample_seed].keys():
            print(f"  - {metric}")
    
    # Plot sample efficiency for average return
    if all_data:
        os.makedirs("results", exist_ok=True)
        plot_sample_efficiency(
            all_data, 
            "charts/average_return",
            save_path="results/sample_efficiency_average_return.png"
        )