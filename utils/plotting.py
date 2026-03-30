from __future__ import annotations

import os
from pathlib import Path
from typing import Any

os.environ.setdefault("XDG_CACHE_HOME", str(Path.cwd() / ".cache"))
os.environ.setdefault("MPLCONFIGDIR", str(Path.cwd() / ".cache" / "matplotlib"))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from utils.io import ensure_dir


def rolling_mean(values: pd.Series, window: int = 10) -> pd.Series:
    return values.rolling(window=window, min_periods=1).mean()


def plot_episode_metrics(metrics_path: Path, title: str, output_path: Path) -> None:
    data = pd.read_csv(metrics_path)
    if data.empty:
        return
    ensure_dir(output_path.parent)
    figure, axis = plt.subplots(figsize=(10, 5))
    axis.plot(data["episode"], data["reward"], alpha=0.35, label="Episode Reward", color="#7aa6c2")
    axis.plot(data["episode"], rolling_mean(data["reward"]), label="Rolling Mean (10)", color="#1f4f70", linewidth=2)
    axis.set_title(title)
    axis.set_xlabel("Episode")
    axis.set_ylabel("Reward")
    axis.legend()
    axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(output_path, dpi=200)
    plt.close(figure)


def plot_contact_sheet(frames: list[np.ndarray], output_path: Path, title: str = "Random Policy Contact Sheet") -> None:
    if not frames:
        return
    ensure_dir(output_path.parent)
    columns = 3
    rows = int(np.ceil(len(frames) / columns))
    figure, axes = plt.subplots(rows, columns, figsize=(12, 4 * rows))
    axes_array = np.atleast_1d(axes).reshape(rows, columns)
    for axis in axes_array.flat:
        axis.axis("off")
    for index, frame in enumerate(frames):
        axis = axes_array.flat[index]
        axis.imshow(frame)
        axis.set_title(f"Frame {index + 1}")
        axis.axis("off")
    figure.suptitle(title, fontsize=16)
    figure.tight_layout()
    figure.savefig(output_path, dpi=200)
    plt.close(figure)


def plot_algorithm_comparison(best_runs: dict[str, dict[str, Any]], output_path: Path) -> None:
    ensure_dir(output_path.parent)
    figure, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes_list = axes.flatten()

    for axis, (algorithm, record) in zip(axes_list, sorted(best_runs.items())):
        metrics_path = Path(record["metrics_path"])
        data = pd.read_csv(metrics_path)
        axis.plot(data["episode"], data["reward"], alpha=0.25, color="#94b4c1")
        axis.plot(data["episode"], rolling_mean(data["reward"]), color="#204969", linewidth=2)
        axis.set_title(f"{algorithm.upper()} Reward Curve")
        axis.set_xlabel("Episode")
        axis.set_ylabel("Reward")
        axis.grid(alpha=0.25)

    figure.tight_layout()
    figure.savefig(output_path, dpi=200)
    plt.close(figure)


def plot_training_stability(best_runs: dict[str, dict[str, Any]], output_path: Path) -> None:
    ensure_dir(output_path.parent)
    figure, axes = plt.subplots(2, 2, figsize=(14, 10))
    axis_lookup = {
        "dqn": ("train/loss", "DQN Objective Loss"),
        "ppo": ("train/entropy_loss", "PPO Entropy"),
        "a2c": ("train/entropy_loss", "A2C Entropy"),
        "reinforce": ("entropy", "REINFORCE Entropy"),
    }

    for axis, algorithm in zip(axes.flatten(), ["dqn", "ppo", "a2c", "reinforce"]):
        record = best_runs.get(algorithm)
        if record is None:
            axis.axis("off")
            continue
        progress_path = Path(record["progress_path"])
        data = pd.read_csv(progress_path)
        column, title = axis_lookup[algorithm]
        if column not in data.columns:
            axis.text(0.5, 0.5, f"{column} not logged", ha="center", va="center")
            axis.axis("off")
            continue
        filtered = data[[column]].dropna().reset_index(drop=True)
        axis.plot(filtered.index, filtered[column], color="#8a3d3d")
        axis.set_title(title)
        axis.set_xlabel("Training Updates")
        axis.set_ylabel(column.split("/")[-1].replace("_", " ").title())
        axis.grid(alpha=0.25)

    figure.tight_layout()
    figure.savefig(output_path, dpi=200)
    plt.close(figure)


def plot_summary_bars(best_runs: dict[str, dict[str, Any]], output_path: Path) -> None:
    ensure_dir(output_path.parent)
    algorithms = list(sorted(best_runs))
    convergence = [best_runs[algo].get("convergence_episode") or 0 for algo in algorithms]
    success_rates = [best_runs[algo]["generalization_success_rate"] for algo in algorithms]

    figure, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].bar(algorithms, convergence, color="#457b9d")
    axes[0].set_title("Episodes to Converge")
    axes[0].set_ylabel("Episode")
    axes[0].grid(axis="y", alpha=0.2)

    axes[1].bar(algorithms, success_rates, color="#2a9d8f")
    axes[1].set_title("Generalization Success Rate")
    axes[1].set_ylabel("Success Rate")
    axes[1].set_ylim(0.0, 1.0)
    axes[1].grid(axis="y", alpha=0.2)

    figure.tight_layout()
    figure.savefig(output_path, dpi=200)
    plt.close(figure)
