from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from training.a2c_training import run as run_a2c
from training.ppo_training import run as run_ppo
from training.reinforce_training import run_experiments as run_reinforce


def main() -> None:
    parser = argparse.ArgumentParser(description="Run policy-gradient-based training scripts.")
    parser.add_argument(
        "--algorithm",
        choices=["ppo", "a2c", "reinforce", "all"],
        default="all",
        help="Which policy method to train.",
    )
    parser.add_argument("--device", default="cpu", help="Torch device to use.")
    parser.add_argument("--limit-runs", type=int, help="Optionally limit the number of sweep runs for quick tests.")
    parser.add_argument("--timesteps-scale", type=float, default=1.0, help="Scale factor for PPO and A2C timesteps.")
    parser.add_argument("--episode-scale", type=float, default=1.0, help="Scale factor for REINFORCE episode counts.")
    args = parser.parse_args()

    if args.algorithm in {"ppo", "all"}:
        run_ppo(device=args.device, limit_runs=args.limit_runs, timesteps_scale=args.timesteps_scale)
    if args.algorithm in {"a2c", "all"}:
        run_a2c(device=args.device, limit_runs=args.limit_runs, timesteps_scale=args.timesteps_scale)
    if args.algorithm in {"reinforce", "all"}:
        run_reinforce(device=args.device, limit_runs=args.limit_runs, episode_scale=args.episode_scale)


if __name__ == "__main__":
    main()
