from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT_DIR / "models"
RESULTS_DIR = ROOT_DIR / "results"
PLOTS_DIR = ROOT_DIR / "plots"
REPORT_ASSETS_DIR = ROOT_DIR / "report_assets"


@dataclass(frozen=True)
class EnvironmentConfig:
    world_size: float = 100.0
    move_distance: float = 6.0
    initial_battery: float = 100.0
    max_steps: int = 250
    movement_battery_cost: float = 1.7
    hover_battery_cost: float = 0.35
    diagonal_energy_multiplier: float = 1.2
    pickup_radius: float = 6.0
    delivery_radius: float = 6.0
    recharge_radius: float = 7.0
    recharge_amount: float = 24.0
    pickup_bonus: float = 30.0
    delivery_bonus: float = 140.0
    recharge_bonus: float = 8.0
    recharge_efficiency_reward: float = 0.18
    step_penalty: float = -0.15
    progress_reward_scale: float = 3.0
    invalid_action_penalty: float = -4.5
    collision_penalty: float = -65.0
    no_fly_penalty: float = -75.0
    battery_depleted_penalty: float = -55.0
    timeout_penalty: float = -30.0
    hover_bonus: float = 0.05
    max_wind_strength: float = 1.6
    start_position_jitter: float = 4.0
    render_fps: int = 12
