#!/usr/bin/env python3

import os
import glob
import pandas as pd
import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from rliable import library as rly
from rliable import plot_utils, metrics
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

def plot_multi_algorithm_comparison(algorithm_data_dict, metric_name, save_path=None):
    """
    Plot sample efficiency curves for multiple algorithms using rliable.
    
    Args:
        algorithm_data_dict: Dict with structure {algorithm_name: all_data_dict}
        metric_name: Name of metric to plot (e.g., "charts/average_return")
        save_path: Optional path to save plot
    """
    # Collect all steps across all algorithms
    all_steps = set()
    for alg_name, all_data in algorithm_data_dict.items():
        for seed_data in all_data.values():
            if metric_name in seed_data:
                all_steps.update(seed_data[metric_name]['step'].values)
    
    if not all_steps:
        raise ValueError(f"No data found for metric '{metric_name}'")
    
    sorted_steps = sorted(all_steps)
    
    # Prepare score dictionary for rliable
    score_dict = {}
    
    for alg_name, all_data in algorithm_data_dict.items():
        seeds = sorted(all_data.keys())
        scores = np.zeros((len(seeds), len(sorted_steps)))
        
        for i, seed in enumerate(seeds):
            if metric_name in all_data[seed]:
                df = all_data[seed][metric_name]
                # Interpolate to common steps
                interpolated_values = np.interp(sorted_steps, df['step'], df['value'])
                scores[i, :] = interpolated_values
        
        score_dict[alg_name] = scores
    
    # Get bootstrap confidence intervals
    iqm = lambda scores: np.array([metrics.aggregate_iqm(scores[..., t])
                               for t in range(scores.shape[-1])])
    
    point_estimates, interval_estimates = rly.get_interval_estimates(
        score_dict, 
        iqm,
        reps=50000
    )
    
    # Create plot
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    
    # Plot using rliable
    plot_utils.plot_sample_efficiency_curve(
        frames=np.array(sorted_steps),
        point_estimates=point_estimates,
        interval_estimates=interval_estimates,
        algorithms=list(algorithm_data_dict.keys()),
        xlabel='Training Steps',
        ylabel=metric_name,
        ax=ax
    )
    
    plt.title(f'Sample Efficiency Comparison: {metric_name.replace("charts/", "").replace("_", " ").title()}')
    plt.legend(loc='best')  # Add legend
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Comparison plot saved to {save_path}")
    
    plt.show()
    
    # Print final performance statistics for each algorithm
    print(f"\nFinal performance statistics for {metric_name}:")
    for alg_name, scores in score_dict.items():
        final_scores = scores[:, -1]
        print(f"  {alg_name}:")
        print(f"    Mean: {np.mean(final_scores):.4f} ± {np.std(final_scores):.4f}")
        print(f"    Min: {np.min(final_scores):.4f}")
        print(f"    Max: {np.max(final_scores):.4f}")
    
    return score_dict, sorted_steps


# Example usage
if __name__ == "__main__":
    # Read data from all 3 directories
    LLMaE_PPO_multirun_dir = "multirun/2025-08-09/00-19-29"
    Transfer_PPO_multirun_dir = "multirun/2025-08-10/21-00-55"
    PPO_multirun_dir = "multirun/NormalPPO/19-26-01"
    
    
    #print("Reading LLMaE-PPO data...")
    #LLMaE_PPO_data = read_all_logs(LLMaE_PPO_multirun_dir)
    
    print("Reading Transfer PPO data...")
    Transfer_PPO_data = read_all_logs(Transfer_PPO_multirun_dir)
    
    #print("Reading Normal PPO data...")
    #PPO_data = read_all_logs(PPO_multirun_dir)

    # Combine all data
    algorithm_data = {
        #"LLMaE-PPO": LLMaE_PPO_data,
        "Transfer PPO": Transfer_PPO_data,
        #"Normal PPO": PPO_data
    }

    # Show available metrics (from first algorithm that has data)
    for alg_name, data in algorithm_data.items():
        if data:
            sample_seed = list(data.keys())[0]
            print(f"\nAvailable metrics in {alg_name}:")
            for metric in data[sample_seed].keys():
                print(f"  - {metric}")
            break

    # Create comparison plots
    os.makedirs("results", exist_ok=True)

    # Plot average return comparison
    plot_multi_algorithm_comparison(
        algorithm_data,
        "charts/average_return",
        save_path="results/average_return_TF.png"
    )

    # Plot success rate comparison
    plot_multi_algorithm_comparison(
        algorithm_data,
        "charts/success_rate",
        save_path="results/success_rate_TF.png"
    )
    
    # Show available metrics (from first algorithm that has data)
    for alg_name, data in algorithm_data.items():
        if data:
            sample_seed = list(data.keys())[0]
            print(f"\nAvailable metrics in {alg_name}:")
            for metric in data[sample_seed].keys():
                print(f"  - {metric}")
            break