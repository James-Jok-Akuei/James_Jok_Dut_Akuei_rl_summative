from __future__ import annotations

import argparse
import time
from pathlib import Path

from environment.custom_env import EmergencyMedicalDroneEnv
from training.reinforce_core import load_reinforce_checkpoint
from utils.config import EnvironmentConfig, RESULTS_DIR
from utils.io import load_json, save_json


def load_model(algorithm: str, model_path: Path, device: str):
    if algorithm == "dqn":
        from stable_baselines3 import DQN

        return DQN.load(str(model_path), device=device)
    if algorithm == "ppo":
        from stable_baselines3 import PPO

        return PPO.load(str(model_path), device=device)
    if algorithm == "a2c":
        from stable_baselines3 import A2C

        return A2C.load(str(model_path), device=device)
    if algorithm == "reinforce":
        policy, _ = load_reinforce_checkpoint(model_path, device=device)
        return policy
    raise ValueError(f"Unsupported algorithm: {algorithm}")


def choose_action(model, algorithm: str, observation, device: str) -> int:
    if algorithm == "reinforce":
        import torch

        with torch.no_grad():
            logits = model(torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0))
            return int(torch.argmax(logits, dim=-1).item())
    action, _ = model.predict(observation, deterministic=True)
    return int(action)


def resolve_model(algorithm: str | None, model_path: str | None, registry_path: Path) -> tuple[str, Path]:
    if algorithm and model_path:
        return algorithm, Path(model_path)
    registry = load_json(registry_path)
    if "algorithm" in registry and "model_path" in registry:
        return str(registry["algorithm"]), Path(registry["model_path"])
    if algorithm:
        record = registry[algorithm]
        return algorithm, Path(record["model_path"])
    best_algorithm = max(registry, key=lambda name: registry[name]["mean_reward"])
    return best_algorithm, Path(registry[best_algorithm]["model_path"])


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the best-performing model inside the drone delivery environment.")
    parser.add_argument("--algorithm", choices=["dqn", "ppo", "a2c", "reinforce"], help="Algorithm to run.")
    parser.add_argument("--model-path", help="Path to a trained model. If omitted, uses results/best_models.json.")
    parser.add_argument("--device", default="cpu", help="Torch device for model loading.")
    parser.add_argument("--episodes", type=int, default=1, help="Number of demo episodes to run.")
    parser.add_argument("--seed", type=int, default=9000, help="Starting seed for replay episodes.")
    parser.add_argument("--sleep", type=float, default=0.08, help="Delay between environment steps for human rendering.")
    parser.add_argument("--export-trace", action="store_true", help="Save episode state traces to JSON.")
    parser.add_argument("--render-mode", choices=["human", "rgb_array", "none"], default="human", help="Rendering mode.")
    parser.add_argument(
        "--registry-path",
        default=str(RESULTS_DIR / "best_models.json"),
        help="Registry JSON to use when --model-path is omitted.",
    )
    args = parser.parse_args()

    algorithm, model_path = resolve_model(args.algorithm, args.model_path, Path(args.registry_path))
    model = load_model(algorithm, model_path, device=args.device)
    render_mode = None if args.render_mode == "none" else args.render_mode
    env = EmergencyMedicalDroneEnv(config=EnvironmentConfig(), render_mode=render_mode, randomize_mission=True)

    for episode in range(args.episodes):
        observation, info = env.reset(seed=args.seed + episode)
        trace = [info["mission_state"]]
        done = False
        truncated = False
        total_reward = 0.0
        print(f"Episode {episode + 1} | Algorithm: {algorithm.upper()} | Model: {model_path}")

        while not (done or truncated):
            action = choose_action(model, algorithm, observation, device=args.device)
            observation, reward, done, truncated, info = env.step(action)
            if render_mode is not None:
                env.render()
            total_reward += reward
            trace.append(info["mission_state"])
            print(
                f"step={info['episode_step']:03d} action={info['mission_state']['last_action']:<18} "
                f"reward={reward:7.2f} battery={info['battery']:6.2f} status={info['mission_status']}"
            )
            if render_mode == "human":
                time.sleep(args.sleep)

        print(f"Episode reward: {total_reward:.2f} | Success: {info['is_success']} | Final status: {info['mission_status']}")
        if args.export_trace:
            save_json(RESULTS_DIR / f"demo_trace_{algorithm}_episode_{episode + 1:02d}.json", {"trace": trace})

    env.close()


if __name__ == "__main__":
    main()
