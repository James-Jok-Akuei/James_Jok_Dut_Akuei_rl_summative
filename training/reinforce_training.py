from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
from torch.distributions import Categorical

from environment.custom_env import EmergencyMedicalDroneEnv
from training.common import prepare_run_dirs, save_run_summary, set_global_seeds, write_summary_csv
from training.experiment_configs import REINFORCE_SWEEP
from training.reinforce_core import ReinforcePolicy, load_reinforce_checkpoint, save_reinforce_checkpoint
from utils.config import EnvironmentConfig
from utils.evaluation import compute_convergence_episode, evaluate_reinforce_policy, held_out_seeds
from utils.io import ensure_dir
from utils.plotting import plot_episode_metrics


def discounted_returns(rewards: list[float], gamma: float, device: str) -> torch.Tensor:
    returns: list[float] = []
    running_return = 0.0
    for reward in reversed(rewards):
        running_return = reward + gamma * running_return
        returns.append(running_return)
    returns.reverse()
    return torch.tensor(returns, dtype=torch.float32, device=device)


def write_csv(path: Path, rows: list[dict[str, float | int | str]]) -> None:
    if not rows:
        return
    ensure_dir(path.parent)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def run_experiments(
    device: str = "cpu",
    base_seed: int = 342,
    limit_runs: int | None = None,
    episode_scale: float = 1.0,
) -> list[dict[str, object]]:
    env_config = EnvironmentConfig()
    summary_records: list[dict[str, object]] = []

    selected_configs = REINFORCE_SWEEP[:limit_runs] if limit_runs is not None else REINFORCE_SWEEP

    for index, config in enumerate(selected_configs):
        run_name = config["run_name"]
        seed = base_seed + index
        set_global_seeds(seed)
        run_dirs = prepare_run_dirs("reinforce", run_name)

        env = EmergencyMedicalDroneEnv(config=env_config, render_mode=None, randomize_mission=True)
        observation_dim = env.observation_space.shape[0]
        action_dim = env.action_space.n
        policy = ReinforcePolicy(observation_dim, action_dim, hidden_dim=int(config["hidden_dim"])).to(device)
        optimizer = torch.optim.Adam(policy.parameters(), lr=float(config["learning_rate"]))

        episode_rows: list[dict[str, float | int | str]] = []
        progress_rows: list[dict[str, float | int | str]] = []
        best_eval_reward = float("-inf")
        best_model_path = run_dirs["model_dir"] / "best_model.pt"
        timesteps_seen = 0
        latest_eval_reward = float("nan")

        scaled_episodes = max(20, int(config["episodes"] * episode_scale))

        for episode in range(1, scaled_episodes + 1):
            observation, _ = env.reset(seed=seed + episode)
            log_probs: list[torch.Tensor] = []
            entropies: list[torch.Tensor] = []
            rewards: list[float] = []
            total_reward = 0.0
            step_count = 0
            done = False
            truncated = False
            info: dict[str, object] = {}

            while not (done or truncated):
                observation_tensor = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)
                logits = policy(observation_tensor)
                distribution = Categorical(logits=logits)
                action = distribution.sample()
                observation, reward, done, truncated, info = env.step(int(action.item()))
                log_probs.append(distribution.log_prob(action))
                entropies.append(distribution.entropy())
                rewards.append(reward)
                total_reward += reward
                step_count += 1

            returns = discounted_returns(rewards, gamma=float(config["gamma"]), device=device)
            if bool(config["normalize_returns"]) and len(returns) > 1:
                returns = (returns - returns.mean()) / (returns.std(unbiased=False) + 1e-8)

            log_probs_tensor = torch.stack(log_probs)
            entropy_tensor = torch.stack(entropies)
            policy_loss = -(log_probs_tensor * returns.detach()).mean() - float(config["entropy_coef"]) * entropy_tensor.mean()

            optimizer.zero_grad()
            policy_loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=1.0)
            optimizer.step()

            timesteps_seen += step_count
            episode_row = {
                "episode": episode,
                "reward": total_reward,
                "length": step_count,
                "timesteps": timesteps_seen,
                "success": int(bool(info.get("is_success", False))),
                "mission_status": str(info.get("mission_status", "unknown")),
                "policy_loss": float(policy_loss.item()),
                "entropy": float(entropy_tensor.mean().item()),
            }
            episode_rows.append(episode_row)

            if episode % 25 == 0 or episode == scaled_episodes:
                evaluation = evaluate_reinforce_policy(policy, env_config, seeds=held_out_seeds(), device=device)
                latest_eval_reward = evaluation["mean_reward"]
                if latest_eval_reward > best_eval_reward:
                    best_eval_reward = latest_eval_reward
                    save_reinforce_checkpoint(
                        best_model_path,
                        policy,
                        observation_dim=observation_dim,
                        action_dim=action_dim,
                        hidden_dim=int(config["hidden_dim"]),
                        metadata={"algorithm": "reinforce", "run_name": run_name, "seed": seed},
                    )

            progress_rows.append(
                {
                    "episode": episode,
                    "policy_loss": float(policy_loss.item()),
                    "entropy": float(entropy_tensor.mean().item()),
                    "mean_reward": total_reward,
                    "eval_mean_reward": latest_eval_reward,
                }
            )

        final_model_path = run_dirs["model_dir"] / "final_model.pt"
        save_reinforce_checkpoint(
            final_model_path,
            policy,
            observation_dim=observation_dim,
            action_dim=action_dim,
            hidden_dim=int(config["hidden_dim"]),
            metadata={"algorithm": "reinforce", "run_name": run_name, "seed": seed},
        )
        if not best_model_path.exists():
            best_model_path = final_model_path

        best_policy, _ = load_reinforce_checkpoint(best_model_path, device=device)
        generalization = evaluate_reinforce_policy(best_policy, env_config, seeds=held_out_seeds(), device=device)

        metrics_path = run_dirs["result_dir"] / "episode_metrics.csv"
        progress_path = run_dirs["result_dir"] / "progress.csv"
        write_csv(metrics_path, episode_rows)
        write_csv(progress_path, progress_rows)
        plot_episode_metrics(metrics_path, "REINFORCE Episode Reward", run_dirs["plot_dir"] / "training_curve.png")

        convergence_episode = compute_convergence_episode([float(row["reward"]) for row in episode_rows])
        summary_payload = {
            "algorithm": "reinforce",
            "run_name": run_name,
            "seed": seed,
            "best_model_path": str(best_model_path),
            "final_model_path": str(final_model_path),
            "metrics_path": str(metrics_path),
            "progress_path": str(progress_path),
            "plot_path": str(run_dirs["plot_dir"] / "training_curve.png"),
            "convergence_episode": convergence_episode,
            "generalization": generalization,
            "hyperparameters": {key: value for key, value in config.items() if key != "run_name"},
        }
        save_run_summary(run_dirs["result_dir"] / "summary.json", summary_payload)

        summary_records.append(
            {
                "algorithm": "reinforce",
                "run_name": run_name,
                "seed": seed,
                **{key: value for key, value in config.items() if key != "run_name"},
                "mean_reward": generalization["mean_reward"],
                "std_reward": generalization["std_reward"],
                "success_rate": generalization["success_rate"],
                "mean_steps": generalization["mean_steps"],
                "convergence_episode": convergence_episode,
                "model_path": str(best_model_path),
                "metrics_path": str(metrics_path),
                "progress_path": str(progress_path),
                "plot_path": str(run_dirs["plot_dir"] / "training_curve.png"),
                "summary_path": str(run_dirs["result_dir"] / "summary.json"),
            }
        )

        env.close()

    write_summary_csv(Path("results") / "reinforce" / "sweep_summary.csv", summary_records)
    return summary_records


def main() -> None:
    parser = argparse.ArgumentParser(description="Train vanilla REINFORCE on the custom drone environment.")
    parser.add_argument("--device", default="cpu", help="Torch device to use, for example cpu or cuda.")
    parser.add_argument("--base-seed", type=int, default=342, help="Base random seed for the sweep.")
    parser.add_argument("--limit-runs", type=int, help="Optionally limit the number of sweep runs for quick tests.")
    parser.add_argument("--episode-scale", type=float, default=1.0, help="Scale factor for episode counts.")
    args = parser.parse_args()
    run_experiments(
        device=args.device,
        base_seed=args.base_seed,
        limit_runs=args.limit_runs,
        episode_scale=args.episode_scale,
    )


if __name__ == "__main__":
    main()
