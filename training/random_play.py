from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from environment.custom_env import EmergencyMedicalDroneEnv
from utils.config import RESULTS_DIR, EnvironmentConfig
from utils.io import save_json
from utils.plotting import plot_contact_sheet


def maybe_save_gif(frames: list, output_path: Path) -> None:
    try:
        import imageio.v2 as imageio
    except ImportError:
        return
    imageio.mimsave(output_path, frames, fps=4)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a random-action visualization for the custom environment.")
    parser.add_argument("--seed", type=int, default=7, help="Seed for the random policy demonstration.")
    parser.add_argument("--episodes", type=int, default=1, help="Number of episodes to simulate.")
    parser.add_argument("--frames", type=int, default=12, help="How many frames to include in the contact sheet.")
    parser.add_argument("--save-gif", action="store_true", help="Also save an animated GIF when imageio is installed.")
    args = parser.parse_args()

    env = EmergencyMedicalDroneEnv(config=EnvironmentConfig(), render_mode="rgb_array", randomize_mission=True)
    collected_frames = []
    last_state = {}

    for episode in range(args.episodes):
        observation, info = env.reset(seed=args.seed + episode)
        last_state = info["mission_state"]
        done = False
        truncated = False
        while not (done or truncated):
            action = env.action_space.sample()
            observation, reward, done, truncated, info = env.step(action)
            frame = env.render()
            if frame is not None:
                collected_frames.append(frame)
            last_state = info["mission_state"]

    env.close()

    if not collected_frames:
        raise RuntimeError("No frames were rendered. Ensure pygame is installed and rendering is available.")

    sample_indices = list(
        sorted(set(int(index) for index in np.linspace(0, len(collected_frames) - 1, num=min(args.frames, len(collected_frames)))))
    )
    sampled_frames = [collected_frames[index] for index in sample_indices]

    contact_sheet_path = RESULTS_DIR / "random_policy_contact_sheet.png"
    plot_contact_sheet(sampled_frames, contact_sheet_path)
    save_json(RESULTS_DIR / "random_policy_state.json", last_state)

    if args.save_gif:
        maybe_save_gif(sampled_frames, RESULTS_DIR / "random_policy_demo.gif")

    print(f"Saved contact sheet to {contact_sheet_path}")


if __name__ == "__main__":
    main()
