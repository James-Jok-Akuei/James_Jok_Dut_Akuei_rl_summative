from __future__ import annotations

import csv
import random
from pathlib import Path
from typing import Any

import numpy as np

from environment.custom_env import EmergencyMedicalDroneEnv
from utils.config import MODELS_DIR, PLOTS_DIR, RESULTS_DIR, EnvironmentConfig
from utils.io import ensure_dir, save_json

try:
    import torch
except ImportError:  # pragma: no cover - depends on local setup.
    torch = None

from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor


DEFAULT_POLICY_KWARGS = {"net_arch": [256, 256]}


class EpisodeStatsCallback(BaseCallback):
    def __init__(self, output_path: Path, verbose: int = 0) -> None:
        super().__init__(verbose=verbose)
        self.output_path = output_path
        self.records: list[dict[str, Any]] = []

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        dones = self.locals.get("dones", [])
        if not isinstance(dones, (list, tuple, np.ndarray)):
            dones = [dones]
        for done, info in zip(dones, infos):
            if done and "episode" in info:
                record = {
                    "episode": len(self.records) + 1,
                    "reward": info["episode"]["r"],
                    "length": info["episode"]["l"],
                    "timesteps": self.num_timesteps,
                    "success": int(bool(info.get("is_success", False))),
                    "mission_status": info.get("mission_status", "unknown"),
                }
                self.records.append(record)
        return True

    def _on_training_end(self) -> None:
        ensure_dir(self.output_path.parent)
        if not self.records:
            return
        fieldnames = list(self.records[0].keys())
        with self.output_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self.records)


def set_global_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    if torch is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)


def make_env(
    env_config: EnvironmentConfig,
    seed: int | None = None,
    render_mode: str | None = None,
    randomize_mission: bool = True,
    monitor_path: Path | None = None,
):
    env = EmergencyMedicalDroneEnv(config=env_config, render_mode=render_mode, randomize_mission=randomize_mission)
    env.reset(seed=seed)
    if monitor_path is not None:
        ensure_dir(monitor_path.parent)
        return Monitor(env, filename=str(monitor_path), info_keywords=("mission_status", "is_success"))
    return env


def prepare_run_dirs(algorithm: str, run_name: str) -> dict[str, Path]:
    result_dir = ensure_dir(RESULTS_DIR / algorithm / run_name)
    model_dir = ensure_dir(MODELS_DIR / algorithm / run_name)
    plot_dir = ensure_dir(PLOTS_DIR / algorithm / run_name)
    return {"result_dir": result_dir, "model_dir": model_dir, "plot_dir": plot_dir}


def write_summary_csv(path: Path, records: list[dict[str, Any]]) -> None:
    if not records:
        return
    ensure_dir(path.parent)
    fieldnames = list(records[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)


def save_run_summary(path: Path, payload: dict[str, Any]) -> None:
    save_json(path, payload)
