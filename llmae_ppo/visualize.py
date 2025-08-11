"""
Visualize results from multiple runs
"""

import glob
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from rliable import library as rly
from rliable import metrics, plot_utils

from utils import extract_step, read_seed, run_dir_from_npz

AGENTS = [
    {
        "name": "LLMaE-PPO",
        "multirun_root": "multirun/2025-08-11/llmae_ppo",
        "npz_prefix": "LLMaE_PPO",
    },
    {
        "name": "Transfer PPO",
        "multirun_root": "multirun/2025-08-11/transfer_ppo",
        "npz_prefix": "Transfer_PPO",
    },
    {
        "name": "PPO from scratch",
        "multirun_root": "multirun/2025-08-11/ppo",
        "npz_prefix": "PPO",
    },
]

METRICS_TO_PLOT = [
    ("Average Return", "returns"),  # returns + mask
    ("Success Rate", "success"),  # success + success_mask
]

THRESHOLD = 0.8  # threshold for TTT (Time To Threshold)
BOOT_REPS = 50000  # bootstrap repetitions for confidence intervals
SAVE_DIR = "results"
os.makedirs(SAVE_DIR, exist_ok=True)


def reduce_mean_per_env(arr: np.ndarray, mask: np.ndarray) -> float:
    """
    Compute the mean of an array over evaluation environments, applying a mask.
    """
    vals = []
    for i in range(arr.shape[0]):
        m = mask[i].astype(bool)
        if np.any(m):
            vals.append(np.nanmean(arr[i][m]))
    return float(np.nanmean(vals)) if vals else np.nan


def load_npz_files(multirun_root: str, prefix: str, which: str):
    """
    Return {seed: DataFrame(step, value)}, each seed = 1 run
    """
    pattern = os.path.join(multirun_root, "**", f"{prefix}_step*.npz")
    files = sorted(glob.glob(pattern, recursive=True), key=extract_step)
    if not files:
        print(f"No npz found for {prefix}")
        return {}

    grouped = {}
    for p in files:
        try:
            run_dir = run_dir_from_npz(p)
            seed = read_seed(run_dir)
            grouped.setdefault(seed, []).append(p)
        except Exception as e:
            print(f"Skip {p}: {e}")
            continue

    out = {}
    for seed, paths in grouped.items():
        steps, vals = [], []
        for p in sorted(paths, key=extract_step):
            z = np.load(p, allow_pickle=True)
            step = int(z["step"])
            if which == "returns":
                if "returns" not in z or "mask" not in z:
                    continue
                arr, m = z["returns"], z["mask"].astype(bool)
            else:
                if "success" not in z or "success_mask" not in z:
                    continue
                arr, m = z["success"], z["success_mask"].astype(bool)
            val = reduce_mean_per_env(arr, m)
            steps.append(step)
            vals.append(val)
        if steps:
            out[seed] = (
                pd.DataFrame({"step": steps, "value": vals})
                .dropna()
                .sort_values("step")
            )
    return out


def build_score_dict(agent_series_dict):
    """
    Build a score dictionary from agent series data.
    agent_series_dict: {agent_name: {seed: DF(step,value)}}
    -> score_dict {agent_name: np.ndarray(num_runs, num_ckpts)}, steps
    """
    all_steps = set()
    for series in agent_series_dict.values():
        for df in series.values():
            all_steps.update(df["step"].tolist())
    steps = sorted(all_steps)
    score_dict = {}
    for agent_name, series in agent_series_dict.items():
        runs = sorted(series.keys())
        mat = np.full((len(runs), len(steps)), np.nan, dtype=np.float32)
        for i, seed in enumerate(runs):
            df = series[seed]
            x, y = df["step"].values, df["value"].values
            if len(x) == 0:
                continue
            # interpolate to match all steps
            mat[i, :] = np.interp(steps, x, y, left=y[0], right=y[-1])
        score_dict[agent_name] = mat
    return score_dict, np.array(steps, dtype=np.int64)


def plot_sample_efficiency(agent_data_dict, label, save_path=None):
    score_dict, steps = build_score_dict(agent_data_dict)
    # IQM over time
    iqm_time = lambda scores: np.array(  # noqa: E731
        [metrics.aggregate_iqm(scores[..., t]) for t in range(scores.shape[-1])]
    )
    point_estimates, interval_estimates = rly.get_interval_estimates(
        score_dict, iqm_time, reps=BOOT_REPS
    )
    fig, ax = plt.subplots(figsize=(12, 8))

    plot_utils.plot_sample_efficiency_curve(
        frames=steps,
        point_estimates=point_estimates,
        interval_estimates=interval_estimates,
        algorithms=list(score_dict.keys()),
        xlabel="Training Steps",
        ylabel=label,
        ax=ax,
    )
    ax.set_title(f"Sample Efficiency — {label}")
    ax.legend(loc="best")
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved: {save_path}")
    plt.show()
    return score_dict, steps


def bootstrap_ci(values, reps=50000, agg=np.nanmean, random_state=0):
    """Simple bootstrap CI on 1D vector (ignores NaNs).
    This is for IQM+CI final, Median+CI final, Mean final, TTT.
    """
    rng = np.random.default_rng(random_state)
    vals = np.asarray(values, dtype=float)
    vals = vals[~np.isnan(vals)]
    if vals.size == 0:
        return (np.nan, np.nan)
    idx = rng.integers(0, vals.size, size=(reps, vals.size))
    boot = agg(vals[idx], axis=1)
    return (float(np.nanpercentile(boot, 2.5)), float(np.nanpercentile(boot, 97.5)))


def summarize_final_metrics(
    score_dict, steps, threshold, save_csv_path=None, title="Metrics Summary"
):
    """
    score_dict: {agent: (num_runs, num_ckpts)}  ; steps: (num_ckpts,)
    """
    # prepare for rliable CI (IQM & Median) at the last step
    last_idx = -1
    final_scores = {agent: arr[:, [last_idx]] for agent, arr in score_dict.items()}

    # IQM + CI (rliable)
    iqm_point, iqm_ci = rly.get_interval_estimates(
        final_scores, metrics.aggregate_iqm, reps=BOOT_REPS
    )
    # Median + CI (rliable)
    med_point, med_ci = rly.get_interval_estimates(
        final_scores, metrics.aggregate_median, reps=BOOT_REPS
    )

    rows = []
    for agent, scores in final_scores.items():
        v = scores  # shape (num_runs,)
        # Mean (point) + bootstrap CI
        mean_val = float(np.nanmean(v))
        mean_ci = bootstrap_ci(v, reps=BOOT_REPS, agg=np.nanmean)

        # TTT: first time step where the score exceeds the threshold
        ttt = []

        runs_over_time = score_dict[agent]  # (num_runs, num_ckpts)
        for run in runs_over_time:
            idx = np.where(run >= threshold)[0]
            if idx.size > 0:
                ttt.append(float(steps[idx[0]]))
            else:
                ttt.append(np.inf)

        ttt = np.array(ttt, dtype=float)
        finite = ttt[np.isfinite(ttt)]
        ttt_mean = float(np.mean(finite)) if finite.size else np.inf
        ttt_median = float(np.median(finite)) if finite.size else np.inf
        # bootstrap CI for TTT
        ttt_ci = (
            bootstrap_ci(finite, reps=BOOT_REPS, agg=np.nanmean)
            if finite.size
            else (np.inf, np.inf)
        )
        hit_rate = float(np.mean(np.isfinite(ttt)))  # % run đạt threshold

        rows.append(
            {
                "Agent": agent,
                "Runs": v.shape[0],
                "Final IQM": f"{iqm_point[agent].item():.3f}",
                "Final IQM 95% CI": f"[{iqm_ci[agent][0].item():.3f}, {iqm_ci[agent][1].item():.3f}]",
                "Final Median": f"{med_point[agent].item():.3f}",
                "Final Median 95% CI": f"[{med_ci[agent][0].item():.3f}, {med_ci[agent][1].item():.3f}]",
                "Final Mean": f"{mean_val:.3f}",
                "Final Mean 95% CI": f"[{mean_ci[0]:.3f}, {mean_ci[1]:.3f}]",
                f"TTT@{threshold} mean": "∞"
                if not np.isfinite(ttt_mean)
                else f"{ttt_mean:.0f}",
                f"TTT@{threshold} median": "∞"
                if not np.isfinite(ttt_median)
                else f"{ttt_median:.0f}",
                f"TTT@{threshold} 95% CI": "∞"
                if not np.isfinite(ttt_mean)
                else f"[{ttt_ci[0]:.0f}, {ttt_ci[1]:.0f}]",
                f"% runs ≥ {threshold}": f"{hit_rate * 100:.1f}%",
            }
        )

    df = pd.DataFrame(rows)
    print("\n" + title)
    print(df.to_string(index=False))
    if save_csv_path:
        df.to_csv(save_csv_path, index=False)
        print(f"Saved CSV: {save_csv_path}")
    return df


if __name__ == "__main__":
    for label, which in METRICS_TO_PLOT:
        agent_series = {}
        for agent in AGENTS:
            data = load_npz_files(agent["multirun_root"], agent["npz_prefix"], which)
            if data:
                agent_series[agent["name"]] = data

        if not agent_series:
            print(f"No data for {label}")
            continue

        # plot sample efficiency
        print(f"Plotting {label}...")
        out_path = os.path.join(SAVE_DIR, f"{which}.png")
        score_dict, steps = plot_sample_efficiency(
            agent_series, label, save_path=out_path
        )

        # summarize final metrics
        print(f"Summarizing final metrics for {label}...")
        csv_path = os.path.join(SAVE_DIR, f"summary_{which}.csv")
        summarize_final_metrics(
            score_dict,
            steps,
            THRESHOLD,
            save_csv_path=csv_path,
            title=f"Summary — {label}",
        )
