from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.logger import configure

from training.common import DEFAULT_POLICY_KWARGS, EpisodeStatsCallback, make_env, prepare_run_dirs, save_run_summary, set_global_seeds, write_summary_csv
from utils.config import EnvironmentConfig
from utils.evaluation import compute_convergence_episode, evaluate_sb3_model, held_out_seeds
from utils.plotting import plot_episode_metrics


def run_sb3_experiments(
    algorithm_name: str,
    algorithm_cls: Any,
    sweep_configs: list[dict[str, Any]],
    device: str = "auto",
    base_seed: int = 42,
    limit_runs: int | None = None,
    timesteps_scale: float = 1.0,
) -> list[dict[str, Any]]:
    env_config = EnvironmentConfig()
    summary_records: list[dict[str, Any]] = []

    selected_configs = sweep_configs[:limit_runs] if limit_runs is not None else sweep_configs

    for index, config in enumerate(selected_configs):
        run_name = config["run_name"]
        seed = base_seed + index
        set_global_seeds(seed)
        run_dirs = prepare_run_dirs(algorithm_name, run_name)

        train_env = make_env(
            env_config=env_config,
            seed=seed,
            render_mode=None,
            monitor_path=run_dirs["result_dir"] / "monitor.csv",
        )
        eval_env = make_env(
            env_config=env_config,
            seed=seed + 1000,
            render_mode=None,
            monitor_path=run_dirs["result_dir"] / "eval_monitor.csv",
        )

        episode_callback = EpisodeStatsCallback(run_dirs["result_dir"] / "episode_metrics.csv")
        eval_callback = EvalCallback(
            eval_env,
            best_model_save_path=str(run_dirs["model_dir"]),
            log_path=str(run_dirs["result_dir"] / "eval_logs"),
            eval_freq=max(250, int(config["total_timesteps"] * timesteps_scale) // 10),
            n_eval_episodes=5,
            deterministic=True,
            render=False,
        )

        model_kwargs = {key: value for key, value in config.items() if key not in {"run_name", "total_timesteps"}}
        if algorithm_name == "dqn":
            scaled_total_timesteps = max(200, int(config["total_timesteps"] * timesteps_scale))
            model_kwargs.setdefault("learning_starts", min(1000, max(100, scaled_total_timesteps // 5)))

        model = algorithm_cls(
            "MlpPolicy",
            train_env,
            verbose=1,
            seed=seed,
            device=device,
            policy_kwargs=DEFAULT_POLICY_KWARGS,
            tensorboard_log=str(run_dirs["result_dir"] / "tb"),
            **model_kwargs,
        )
        model.set_logger(configure(str(run_dirs["result_dir"]), ["stdout", "csv"]))
        model.learn(
            total_timesteps=max(200, int(config["total_timesteps"] * timesteps_scale)),
            callback=[episode_callback, eval_callback],
            progress_bar=False,
        )
        model.save(str(run_dirs["model_dir"] / "final_model"))

        best_model_path = run_dirs["model_dir"] / "best_model.zip"
        if not best_model_path.exists():
            best_model_path = run_dirs["model_dir"] / "final_model.zip"
        best_model = algorithm_cls.load(str(best_model_path), device=device)

        generalization = evaluate_sb3_model(best_model, env_config, seeds=held_out_seeds())
        metrics_path = run_dirs["result_dir"] / "episode_metrics.csv"
        plot_episode_metrics(metrics_path, f"{algorithm_name.upper()} Episode Reward", run_dirs["plot_dir"] / "training_curve.png")

        metrics = pd.read_csv(metrics_path) if metrics_path.exists() else pd.DataFrame()
        episode_rewards = metrics["reward"].tolist() if "reward" in metrics else []
        convergence_episode = compute_convergence_episode(episode_rewards)

        summary_payload = {
            "algorithm": algorithm_name,
            "run_name": run_name,
            "seed": seed,
            "best_model_path": str(best_model_path),
            "final_model_path": str(run_dirs["model_dir"] / "final_model.zip"),
            "metrics_path": str(metrics_path),
            "progress_path": str(run_dirs["result_dir"] / "progress.csv"),
            "plot_path": str(run_dirs["plot_dir"] / "training_curve.png"),
            "convergence_episode": convergence_episode,
            "generalization": generalization,
            "hyperparameters": model_kwargs,
        }
        save_run_summary(run_dirs["result_dir"] / "summary.json", summary_payload)

        summary_record = {
            "algorithm": algorithm_name,
            "run_name": run_name,
            "seed": seed,
            **model_kwargs,
            "mean_reward": generalization["mean_reward"],
            "std_reward": generalization["std_reward"],
            "success_rate": generalization["success_rate"],
            "mean_steps": generalization["mean_steps"],
            "convergence_episode": convergence_episode,
            "model_path": str(best_model_path),
            "metrics_path": str(metrics_path),
            "progress_path": str(run_dirs["result_dir"] / "progress.csv"),
            "plot_path": str(run_dirs["plot_dir"] / "training_curve.png"),
            "summary_path": str(run_dirs["result_dir"] / "summary.json"),
        }
        summary_records.append(summary_record)

        train_env.close()
        eval_env.close()

    write_summary_csv(Path("results") / algorithm_name / "sweep_summary.csv", summary_records)
    return summary_records
