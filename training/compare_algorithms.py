from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.config import PLOTS_DIR, RESULTS_DIR
from utils.io import save_json
from utils.plotting import plot_algorithm_comparison, plot_summary_bars, plot_training_stability


def main() -> None:
    algorithms = ["dqn", "reinforce", "ppo", "a2c"]
    best_runs: dict[str, dict[str, object]] = {}

    for algorithm in algorithms:
        summary_path = RESULTS_DIR / algorithm / "sweep_summary.csv"
        if not summary_path.exists():
            continue
        summary = pd.read_csv(summary_path)
        if summary.empty:
            continue
        best_row = summary.sort_values("mean_reward", ascending=False).iloc[0].to_dict()
        best_runs[algorithm] = {
            "run_name": best_row["run_name"],
            "mean_reward": float(best_row["mean_reward"]),
            "success_rate": float(best_row["success_rate"]),
            "convergence_episode": None if pd.isna(best_row["convergence_episode"]) else int(best_row["convergence_episode"]),
            "metrics_path": str(best_row["metrics_path"]),
            "progress_path": str(best_row["progress_path"]),
            "model_path": str(best_row["model_path"]),
            "generalization_success_rate": float(best_row["success_rate"]),
        }

    if not best_runs:
        raise RuntimeError("No sweep summaries were found. Train the algorithms first.")

    overview_dir = PLOTS_DIR / "overview"
    plot_algorithm_comparison(best_runs, overview_dir / "cumulative_rewards.png")
    plot_training_stability(best_runs, overview_dir / "training_stability.png")
    plot_summary_bars(best_runs, overview_dir / "summary_bars.png")
    save_json(RESULTS_DIR / "best_models.json", best_runs)
    print(f"Saved comparison plots to {overview_dir}")


if __name__ == "__main__":
    main()
