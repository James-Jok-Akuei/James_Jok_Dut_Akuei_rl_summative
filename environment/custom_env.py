from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from utils.config import EnvironmentConfig


@dataclass(frozen=True)
class RectangleObstacle:
    x: float
    y: float
    width: float
    height: float
    label: str = "building"

    def contains(self, point: np.ndarray) -> bool:
        return self.x <= point[0] <= self.x + self.width and self.y <= point[1] <= self.y + self.height

    def distance_to(self, point: np.ndarray) -> float:
        dx = max(self.x - point[0], 0.0, point[0] - (self.x + self.width))
        dy = max(self.y - point[1], 0.0, point[1] - (self.y + self.height))
        return float(np.hypot(dx, dy))

    def center(self) -> np.ndarray:
        return np.array([self.x + self.width / 2.0, self.y + self.height / 2.0], dtype=np.float32)


@dataclass(frozen=True)
class CircleZone:
    x: float
    y: float
    radius: float
    label: str = "no_fly_zone"

    def contains(self, point: np.ndarray) -> bool:
        return float(np.linalg.norm(point - np.array([self.x, self.y], dtype=np.float32))) <= self.radius

    def distance_to(self, point: np.ndarray) -> float:
        distance = float(np.linalg.norm(point - np.array([self.x, self.y], dtype=np.float32))) - self.radius
        return max(0.0, distance)


class EmergencyMedicalDroneEnv(gym.Env[np.ndarray, int]):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 12}

    ACTION_LABELS = {
        0: "hover",
        1: "move_north",
        2: "move_south",
        3: "move_east",
        4: "move_west",
        5: "move_northeast",
        6: "move_northwest",
        7: "move_southeast",
        8: "move_southwest",
        9: "pickup_supplies",
        10: "deliver_supplies",
        11: "recharge_battery",
    }

    MOVEMENT_VECTORS = {
        1: np.array([0.0, 1.0], dtype=np.float32),
        2: np.array([0.0, -1.0], dtype=np.float32),
        3: np.array([1.0, 0.0], dtype=np.float32),
        4: np.array([-1.0, 0.0], dtype=np.float32),
        5: np.array([1.0, 1.0], dtype=np.float32) / np.sqrt(2.0),
        6: np.array([-1.0, 1.0], dtype=np.float32) / np.sqrt(2.0),
        7: np.array([1.0, -1.0], dtype=np.float32) / np.sqrt(2.0),
        8: np.array([-1.0, -1.0], dtype=np.float32) / np.sqrt(2.0),
    }

    def __init__(
        self,
        config: EnvironmentConfig | None = None,
        render_mode: str | None = None,
        randomize_mission: bool = True,
    ) -> None:
        super().__init__()
        self.config = config or EnvironmentConfig()
        self.render_mode = render_mode
        self.randomize_mission = randomize_mission

        self.action_space = spaces.Discrete(len(self.ACTION_LABELS))
        self.observation_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(22,),
            dtype=np.float32,
        )

        self.renderer = None
        self.agent_pos = np.zeros(2, dtype=np.float32)
        self.base_pos = np.array([14.0, 14.0], dtype=np.float32)
        self.patient_pos = np.array([85.0, 50.0], dtype=np.float32)
        self.charge_pos = np.array([18.0, 84.0], dtype=np.float32)
        self.wind_vector = np.zeros(2, dtype=np.float32)
        self.obstacles: list[RectangleObstacle] = []
        self.no_fly_zones: list[CircleZone] = []
        self.battery = self.config.initial_battery
        self.carrying_supplies = False
        self.delivery_complete = False
        self.last_action = "hover"
        self.last_reward = 0.0
        self.episode_step = 0
        self.mission_status = "idle"
        self._distance_reference = 0.0

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        self.episode_step = 0
        self.battery = self.config.initial_battery
        self.carrying_supplies = False
        self.delivery_complete = False
        self.last_action = "hover"
        self.last_reward = 0.0
        self.mission_status = "mission_active"
        self._generate_scenario(options or {})
        self._distance_reference = self._distance_to_goal()
        observation = self._get_observation()
        info = self._build_info()
        return observation, info

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        reward = self.config.step_penalty
        terminated = False
        truncated = False
        status = "mission_active"
        previous_goal_distance = self._distance_to_goal()
        movement_energy_cost = self.config.hover_battery_cost

        action = int(action)
        self.last_action = self.ACTION_LABELS[action]

        if action in self.MOVEMENT_VECTORS:
            direction = self.MOVEMENT_VECTORS[action]
            movement_energy_cost = self.config.movement_battery_cost
            if action >= 5:
                movement_energy_cost *= self.config.diagonal_energy_multiplier
            candidate_pos = self.agent_pos + direction * self.config.move_distance + self.wind_vector
            self.agent_pos = np.clip(candidate_pos, 0.0, self.config.world_size)
            reward += self._progress_reward(previous_goal_distance)
        elif action == 9:
            reward += self._handle_pickup()
        elif action == 10:
            delivery_reward, terminated, status = self._handle_delivery()
            reward += delivery_reward
        elif action == 11:
            reward += self._handle_recharge()
        else:
            reward += self.config.hover_bonus if self._safe_to_hover() else 0.0

        self.battery -= movement_energy_cost
        self.episode_step += 1

        collision = any(obstacle.contains(self.agent_pos) for obstacle in self.obstacles)
        no_fly_violation = any(zone.contains(self.agent_pos) for zone in self.no_fly_zones)

        if collision:
            reward += self.config.collision_penalty
            terminated = True
            status = "collision"
        elif no_fly_violation:
            reward += self.config.no_fly_penalty
            terminated = True
            status = "no_fly_violation"
        elif self.battery <= 0.0:
            reward += self.config.battery_depleted_penalty
            terminated = True
            status = "battery_depleted"
        elif self.episode_step >= self.config.max_steps:
            reward += self.config.timeout_penalty
            truncated = True
            status = "timeout"

        self.mission_status = status
        self.last_reward = float(reward)
        info = self._build_info()
        observation = self._get_observation()
        return observation, float(reward), terminated, truncated, info

    def render(self) -> np.ndarray | None:
        if self.render_mode is None:
            return None
        if self.renderer is None:
            from environment.rendering import DroneDeliveryRenderer

            self.renderer = DroneDeliveryRenderer(self.config, render_mode=self.render_mode)
        return self.renderer.render(self.serialize_state())

    def close(self) -> None:
        if self.renderer is not None:
            self.renderer.close()
            self.renderer = None

    def serialize_state(self) -> dict[str, Any]:
        return {
            "world_size": self.config.world_size,
            "agent_position": self.agent_pos.tolist(),
            "base_position": self.base_pos.tolist(),
            "patient_position": self.patient_pos.tolist(),
            "charge_position": self.charge_pos.tolist(),
            "battery": float(self.battery),
            "carrying_supplies": self.carrying_supplies,
            "delivery_complete": self.delivery_complete,
            "episode_step": self.episode_step,
            "max_steps": self.config.max_steps,
            "wind_vector": self.wind_vector.tolist(),
            "last_action": self.last_action,
            "last_reward": float(self.last_reward),
            "mission_status": self.mission_status,
            "action_labels": self.ACTION_LABELS,
            "obstacles": [asdict(obstacle) for obstacle in self.obstacles],
            "no_fly_zones": [asdict(zone) for zone in self.no_fly_zones],
        }

    def _generate_scenario(self, options: dict[str, Any]) -> None:
        rng = self.np_random
        world = self.config.world_size

        start_noise = rng.uniform(-self.config.start_position_jitter, self.config.start_position_jitter, size=2)
        self.agent_pos = np.clip(self.base_pos + start_noise, 2.0, world - 2.0).astype(np.float32)

        patient_x = float(rng.uniform(68.0, 92.0))
        patient_y = float(rng.uniform(14.0, 86.0))
        self.patient_pos = np.array([patient_x, patient_y], dtype=np.float32)

        self.obstacles = []
        base_obstacles = [
            RectangleObstacle(28.0, 18.0, 12.0, 20.0, label="city_hospital"),
            RectangleObstacle(46.0, 52.0, 16.0, 16.0, label="office_block"),
            RectangleObstacle(63.0, 22.0, 12.0, 22.0, label="apartment_block"),
        ]
        for obstacle in base_obstacles:
            if self.randomize_mission:
                jitter = rng.uniform(-4.0, 4.0, size=2)
                width_jitter = float(rng.uniform(-2.0, 2.0))
                height_jitter = float(rng.uniform(-2.0, 2.0))
                candidate = RectangleObstacle(
                    x=float(np.clip(obstacle.x + jitter[0], 20.0, 72.0)),
                    y=float(np.clip(obstacle.y + jitter[1], 10.0, 72.0)),
                    width=max(8.0, obstacle.width + width_jitter),
                    height=max(8.0, obstacle.height + height_jitter),
                    label=obstacle.label,
                )
            else:
                candidate = obstacle
            self.obstacles.append(candidate)

        self.no_fly_zones = []
        base_zones = [
            CircleZone(52.0, 31.0, 9.0),
            CircleZone(76.0, 68.0, 10.0),
        ]
        for zone in base_zones:
            if self.randomize_mission:
                jitter = rng.uniform(-3.5, 3.5, size=2)
                radius_jitter = float(rng.uniform(-1.0, 1.5))
                candidate = CircleZone(
                    x=float(np.clip(zone.x + jitter[0], 25.0, 88.0)),
                    y=float(np.clip(zone.y + jitter[1], 12.0, 88.0)),
                    radius=max(6.0, zone.radius + radius_jitter),
                    label=zone.label,
                )
            else:
                candidate = zone
            self.no_fly_zones.append(candidate)

        wind_angle = float(rng.uniform(0.0, 2.0 * np.pi))
        wind_strength = float(rng.uniform(0.0, self.config.max_wind_strength))
        self.wind_vector = np.array(
            [np.cos(wind_angle) * wind_strength, np.sin(wind_angle) * wind_strength],
            dtype=np.float32,
        )

    def _handle_pickup(self) -> float:
        if self.carrying_supplies:
            return self.config.invalid_action_penalty
        if self._distance(self.agent_pos, self.base_pos) > self.config.pickup_radius:
            return self.config.invalid_action_penalty
        self.carrying_supplies = True
        self._distance_reference = self._distance_to_goal()
        return self.config.pickup_bonus

    def _handle_delivery(self) -> tuple[float, bool, str]:
        if not self.carrying_supplies:
            return self.config.invalid_action_penalty, False, "mission_active"
        if self._distance(self.agent_pos, self.patient_pos) > self.config.delivery_radius:
            return self.config.invalid_action_penalty, False, "mission_active"
        self.carrying_supplies = False
        self.delivery_complete = True
        return self.config.delivery_bonus, True, "delivered"

    def _handle_recharge(self) -> float:
        recharge_source_nearby = (
            self._distance(self.agent_pos, self.base_pos) <= self.config.recharge_radius
            or self._distance(self.agent_pos, self.charge_pos) <= self.config.recharge_radius
        )
        if not recharge_source_nearby:
            return self.config.invalid_action_penalty
        if self.battery >= self.config.initial_battery - 1.0:
            return self.config.invalid_action_penalty / 2.0
        previous_battery = self.battery
        self.battery = min(self.config.initial_battery, self.battery + self.config.recharge_amount)
        recovered = self.battery - previous_battery
        return self.config.recharge_bonus + recovered * self.config.recharge_efficiency_reward

    def _progress_reward(self, previous_goal_distance: float) -> float:
        current_goal_distance = self._distance_to_goal()
        improvement = previous_goal_distance - current_goal_distance
        return improvement * self.config.progress_reward_scale

    def _distance_to_goal(self) -> float:
        goal = self.patient_pos if self.carrying_supplies else self.base_pos
        return self._distance(self.agent_pos, goal)

    def _safe_to_hover(self) -> bool:
        return not any(zone.contains(self.agent_pos) for zone in self.no_fly_zones)

    def _get_observation(self) -> np.ndarray:
        world = self.config.world_size
        goal = self.patient_pos if self.carrying_supplies else self.base_pos
        goal_delta = (goal - self.agent_pos) / world
        min_obstacle_distance = min((obstacle.distance_to(self.agent_pos) for obstacle in self.obstacles), default=world)
        min_zone_distance = min((zone.distance_to(self.agent_pos) for zone in self.no_fly_zones), default=world)
        observation = np.array(
            [
                self.agent_pos[0] / world,
                self.agent_pos[1] / world,
                self.base_pos[0] / world,
                self.base_pos[1] / world,
                self.patient_pos[0] / world,
                self.patient_pos[1] / world,
                self.charge_pos[0] / world,
                self.charge_pos[1] / world,
                self.battery / self.config.initial_battery,
                (self.config.max_steps - self.episode_step) / self.config.max_steps,
                float(self.carrying_supplies),
                float(self.delivery_complete),
                goal_delta[0],
                goal_delta[1],
                min_obstacle_distance / world,
                min_zone_distance / world,
                self.wind_vector[0] / max(1.0, self.config.max_wind_strength),
                self.wind_vector[1] / max(1.0, self.config.max_wind_strength),
                float(self._distance(self.agent_pos, self.base_pos) <= self.config.pickup_radius),
                float(self._distance(self.agent_pos, self.patient_pos) <= self.config.delivery_radius),
                float(self._distance(self.agent_pos, self.charge_pos) <= self.config.recharge_radius),
                float(any(zone.contains(self.agent_pos) for zone in self.no_fly_zones)),
            ],
            dtype=np.float32,
        )
        return np.clip(observation, -1.0, 1.0)

    def _build_info(self) -> dict[str, Any]:
        return {
            "battery": float(self.battery),
            "episode_step": self.episode_step,
            "mission_status": self.mission_status,
            "delivery_complete": self.delivery_complete,
            "carrying_supplies": self.carrying_supplies,
            "distance_to_patient": self._distance(self.agent_pos, self.patient_pos),
            "distance_to_base": self._distance(self.agent_pos, self.base_pos),
            "distance_to_charge": self._distance(self.agent_pos, self.charge_pos),
            "is_success": self.delivery_complete,
            "mission_state": self.serialize_state(),
        }

    @staticmethod
    def _distance(point_a: np.ndarray, point_b: np.ndarray) -> float:
        return float(np.linalg.norm(point_a - point_b))
