from __future__ import annotations

from typing import Any, Iterable

import numpy as np

from environment.custom_env import EmergencyMedicalDroneEnv
from utils.config import EnvironmentConfig


def held_out_seeds(count: int = 10, start: int = 5000) -> list[int]:
    return [start + offset for offset in range(count)]


def compute_convergence_episode(episode_rewards: list[float], window: int = 10, threshold: float = 0.9) -> int | None:
    if len(episode_rewards) < window:
        return None
    rewards = np.asarray(episode_rewards, dtype=np.float32)
    rolling = np.convolve(rewards, np.ones(window) / window, mode="valid")
    target = float(np.max(rolling) * threshold)
    for index, value in enumerate(rolling, start=window):
        if value >= target:
            return index
    return None


def evaluate_sb3_model(
    model: Any,
    env_config: EnvironmentConfig,
    seeds: Iterable[int] | None = None,
    n_episodes: int = 10,
    deterministic: bool = True,
) -> dict[str, Any]:
    evaluation_seeds = list(seeds) if seeds is not None else held_out_seeds(count=n_episodes)
    episode_records: list[dict[str, Any]] = []

    for seed in evaluation_seeds[:n_episodes]:
        env = EmergencyMedicalDroneEnv(config=env_config, render_mode=None, randomize_mission=True)
        observation, _ = env.reset(seed=seed)
        done = False
        truncated = False
        total_reward = 0.0
        step_count = 0
        info: dict[str, Any] = {}
        while not (done or truncated):
            action, _ = model.predict(observation, deterministic=deterministic)
            observation, reward, done, truncated, info = env.step(int(action))
            total_reward += reward
            step_count += 1
        episode_records.append(
            {
                "seed": seed,
                "reward": total_reward,
                "steps": step_count,
                "success": bool(info.get("is_success", False)),
                "mission_status": info.get("mission_status", "unknown"),
                "battery_remaining": float(info.get("battery", 0.0)),
            }
        )
        env.close()

    return summarise_evaluation(episode_records)


def evaluate_reinforce_policy(
    policy: Any,
    env_config: EnvironmentConfig,
    seeds: Iterable[int] | None = None,
    n_episodes: int = 10,
    device: str = "cpu",
) -> dict[str, Any]:
    import torch

    evaluation_seeds = list(seeds) if seeds is not None else held_out_seeds(count=n_episodes)
    episode_records: list[dict[str, Any]] = []
    policy.eval()

    with torch.no_grad():
        for seed in evaluation_seeds[:n_episodes]:
            env = EmergencyMedicalDroneEnv(config=env_config, render_mode=None, randomize_mission=True)
            observation, _ = env.reset(seed=seed)
            done = False
            truncated = False
            total_reward = 0.0
            step_count = 0
            info: dict[str, Any] = {}
            while not (done or truncated):
                observation_tensor = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)
                logits = policy(observation_tensor)
                action = int(torch.argmax(logits, dim=-1).item())
                observation, reward, done, truncated, info = env.step(action)
                total_reward += reward
                step_count += 1
            episode_records.append(
                {
                    "seed": seed,
                    "reward": total_reward,
                    "steps": step_count,
                    "success": bool(info.get("is_success", False)),
                    "mission_status": info.get("mission_status", "unknown"),
                    "battery_remaining": float(info.get("battery", 0.0)),
                }
            )
            env.close()

    return summarise_evaluation(episode_records)


def summarise_evaluation(records: list[dict[str, Any]]) -> dict[str, Any]:
    rewards = [record["reward"] for record in records]
    steps = [record["steps"] for record in records]
    success_rate = float(np.mean([record["success"] for record in records])) if records else 0.0
    return {
        "episodes": records,
        "mean_reward": float(np.mean(rewards)) if rewards else 0.0,
        "std_reward": float(np.std(rewards)) if rewards else 0.0,
        "mean_steps": float(np.mean(steps)) if steps else 0.0,
        "success_rate": success_rate,
    }
