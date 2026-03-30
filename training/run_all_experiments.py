from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from training.a2c_training import run as run_a2c
from training.compare_algorithms import main as compare_main
from training.dqn_training import run as run_dqn
from training.ppo_training import run as run_ppo
from training.reinforce_training import run_experiments as run_reinforce


def main() -> None:
    parser = argparse.ArgumentParser(description="Run all algorithm sweeps and rebuild the comparison outputs.")
    parser.add_argument("--device", default="cpu", help="Torch device to use.")
    parser.add_argument("--limit-runs", type=int, help="Optionally limit the number of runs per algorithm.")
    parser.add_argument("--timesteps-scale", type=float, default=1.0, help="Scale factor for DQN, PPO, and A2C timesteps.")
    parser.add_argument("--episode-scale", type=float, default=1.0, help="Scale factor for REINFORCE episodes.")
    args = parser.parse_args()

    run_dqn(device=args.device, limit_runs=args.limit_runs, timesteps_scale=args.timesteps_scale)
    run_ppo(device=args.device, limit_runs=args.limit_runs, timesteps_scale=args.timesteps_scale)
    run_a2c(device=args.device, limit_runs=args.limit_runs, timesteps_scale=args.timesteps_scale)
    run_reinforce(device=args.device, limit_runs=args.limit_runs, episode_scale=args.episode_scale)
    compare_main()


if __name__ == "__main__":
    main()
