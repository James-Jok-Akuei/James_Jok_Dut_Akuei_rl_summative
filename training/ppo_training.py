from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from stable_baselines3 import PPO

from training.sb3_runner import run_sb3_experiments


def run(device: str = "auto", base_seed: int = 142, limit_runs: int | None = None, timesteps_scale: float = 1.0):
    from training.experiment_configs import PPO_SWEEP

    return run_sb3_experiments(
        "ppo",
        PPO,
        PPO_SWEEP,
        device=device,
        base_seed=base_seed,
        limit_runs=limit_runs,
        timesteps_scale=timesteps_scale,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Train PPO on the Emergency Medical Drone Delivery environment.")
    parser.add_argument("--device", default="auto", help="Torch device to use, for example cpu, cuda, or auto.")
    parser.add_argument("--base-seed", type=int, default=142, help="Base random seed for the sweep.")
    parser.add_argument("--limit-runs", type=int, help="Optionally limit the number of sweep runs for quick tests.")
    parser.add_argument("--timesteps-scale", type=float, default=1.0, help="Scale factor for total timesteps.")
    args = parser.parse_args()
    run(device=args.device, base_seed=args.base_seed, limit_runs=args.limit_runs, timesteps_scale=args.timesteps_scale)


if __name__ == "__main__":
    main()
