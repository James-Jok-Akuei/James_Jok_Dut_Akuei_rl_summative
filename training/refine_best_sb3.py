from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from stable_baselines3 import A2C, DQN, PPO
from stable_baselines3.common.logger import configure

from training.common import EpisodeStatsCallback, make_env, save_run_summary
from utils.config import MODELS_DIR, PLOTS_DIR, RESULTS_DIR, EnvironmentConfig
from utils.evaluation import evaluate_sb3_model, held_out_seeds
from utils.io import ensure_dir, save_json
from utils.plotting import plot_episode_metrics


ALGORITHM_CLASSES = {
    "dqn": DQN,
    "ppo": PPO,
    "a2c": A2C,
}


def load_registry(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    ensure_dir(path.parent)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def refine_algorithm(
    algorithm: str,
    extra_timesteps: int,
    device: str,
    eval_episodes: int,
    registry_path: Path,
) -> dict[str, Any]:
    registry = load_registry(registry_path)
    record = registry[algorithm]
    env_config = EnvironmentConfig()

    model_cls = ALGORITHM_CLASSES[algorithm]
    source_model_path = Path(record["model_path"])
    source_run_name = str(record["run_name"])
    refined_run_name = f"{source_run_name}_refined_{extra_timesteps}"

    result_dir = ensure_dir(RESULTS_DIR / "refinement" / algorithm / refined_run_name)
    plot_dir = ensure_dir(PLOTS_DIR / "refinement" / algorithm / refined_run_name)
    model_dir = ensure_dir(MODELS_DIR / "refinement" / algorithm / refined_run_name)

    train_env = make_env(
        env_config=env_config,
        seed=11000,
        render_mode=None,
        monitor_path=result_dir / "monitor.csv",
    )

    model = model_cls.load(str(source_model_path), env=train_env, device=device)
    model.set_logger(configure(str(result_dir), ["stdout", "csv"]))

    episode_callback = EpisodeStatsCallback(result_dir / "episode_metrics.csv")
    model.learn(total_timesteps=extra_timesteps, callback=episode_callback, reset_num_timesteps=False, progress_bar=False)

    refined_model_path = model_dir / "refined_model"
    model.save(str(refined_model_path))
    train_env.close()

    evaluation = evaluate_sb3_model(
        model,
        env_config,
        seeds=held_out_seeds(count=eval_episodes, start=7000),
        n_episodes=eval_episodes,
        deterministic=True,
    )

    metrics_path = result_dir / "episode_metrics.csv"
    progress_path = result_dir / "progress.csv"
    plot_episode_metrics(metrics_path, f"{algorithm.upper()} Refinement Reward Curve", plot_dir / "training_curve.png")

    summary = {
        "algorithm": algorithm,
        "source_run_name": source_run_name,
        "refined_run_name": refined_run_name,
        "source_model_path": str(source_model_path),
        "refined_model_path": str(refined_model_path.with_suffix(".zip")),
        "extra_timesteps": extra_timesteps,
        "metrics_path": str(metrics_path),
        "progress_path": str(progress_path),
        "plot_path": str(plot_dir / "training_curve.png"),
        "evaluation": evaluation,
    }
    save_run_summary(result_dir / "summary.json", summary)

    return {
        "algorithm": algorithm,
        "source_run_name": source_run_name,
        "refined_run_name": refined_run_name,
        "extra_timesteps": extra_timesteps,
        "mean_reward": evaluation["mean_reward"],
        "std_reward": evaluation["std_reward"],
        "success_rate": evaluation["success_rate"],
        "mean_steps": evaluation["mean_steps"],
        "model_path": str(refined_model_path.with_suffix(".zip")),
        "summary_path": str(result_dir / "summary.json"),
        "metrics_path": str(metrics_path),
        "progress_path": str(progress_path),
        "plot_path": str(plot_dir / "training_curve.png"),
    }


def select_final_demo_candidate(
    refinement_records: list[dict[str, Any]],
    base_registry_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    base_registry = load_registry(base_registry_path)
    candidates: list[dict[str, Any]] = []

    for algorithm, record in base_registry.items():
        candidates.append(
            {
                "algorithm": algorithm,
                "source": "sweep_best",
                "run_name": record["run_name"],
                "mean_reward": record["mean_reward"],
                "success_rate": record["success_rate"],
                "model_path": record["model_path"],
            }
        )

    for record in refinement_records:
        candidates.append(
            {
                "algorithm": record["algorithm"],
                "source": "refined",
                "run_name": record["refined_run_name"],
                "mean_reward": record["mean_reward"],
                "success_rate": record["success_rate"],
                "model_path": record["model_path"],
            }
        )

    best = max(candidates, key=lambda item: (item["success_rate"], item["mean_reward"]))
    save_json(output_path, best)
    return best


def main() -> None:
    parser = argparse.ArgumentParser(description="Continue training the strongest SB3 models for a better final demo agent.")
    parser.add_argument(
        "--algorithms",
        nargs="+",
        choices=["dqn", "ppo", "a2c"],
        default=["a2c", "ppo"],
        help="Algorithms to refine. Defaults to the strongest on-policy candidates.",
    )
    parser.add_argument(
        "--extra-timesteps",
        type=int,
        default=80000,
        help="Additional timesteps to train each selected model.",
    )
    parser.add_argument("--device", default="cpu", help="Torch device to use.")
    parser.add_argument("--eval-episodes", type=int, default=20, help="Held-out episodes for post-refinement evaluation.")
    parser.add_argument(
        "--registry-path",
        default=str(RESULTS_DIR / "best_models.json"),
        help="Path to the base best-model registry.",
    )
    args = parser.parse_args()

    registry_path = Path(args.registry_path)
    refinement_records: list[dict[str, Any]] = []
    for algorithm in args.algorithms:
        refinement_records.append(
            refine_algorithm(
                algorithm=algorithm,
                extra_timesteps=args.extra_timesteps,
                device=args.device,
                eval_episodes=args.eval_episodes,
                registry_path=registry_path,
            )
        )

    write_csv(RESULTS_DIR / "refinement" / "refined_summary.csv", refinement_records)
    best = select_final_demo_candidate(
        refinement_records=refinement_records,
        base_registry_path=registry_path,
        output_path=RESULTS_DIR / "final_demo_model.json",
    )
    print(f"Saved refinement summary to {RESULTS_DIR / 'refinement' / 'refined_summary.csv'}")
    print(f"Selected final demo agent: {best['algorithm'].upper()} from {best['source']} ({best['run_name']})")


if __name__ == "__main__":
    main()
