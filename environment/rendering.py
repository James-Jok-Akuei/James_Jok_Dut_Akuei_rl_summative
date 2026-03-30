from __future__ import annotations

from typing import Any

import numpy as np

from utils.config import EnvironmentConfig

try:
    import pygame
except ImportError:  # pragma: no cover - handled at runtime.
    pygame = None


class DroneDeliveryRenderer:
    def __init__(self, config: EnvironmentConfig, render_mode: str = "human") -> None:
        if pygame is None:
            raise ImportError("pygame is required for rendering. Install project dependencies from requirements.txt.")
        self.config = config
        self.render_mode = render_mode
        self.canvas_size = 900
        self.map_size = 700
        self.sidebar_width = self.canvas_size - self.map_size
        self.window = None
        self.clock = None
        self.surface = None

        pygame.init()
        if render_mode == "human":
            pygame.display.init()
            self.window = pygame.display.set_mode((self.canvas_size, self.canvas_size))
            pygame.display.set_caption("Emergency Medical Drone Delivery")
        self.clock = pygame.time.Clock()
        self.surface = pygame.Surface((self.canvas_size, self.canvas_size), pygame.SRCALPHA)
        self.font = pygame.font.SysFont("arial", 20)
        self.small_font = pygame.font.SysFont("arial", 16)

    def render(self, state: dict[str, Any]) -> np.ndarray | None:
        assert self.surface is not None
        self.surface.fill((238, 242, 232))

        self._draw_map_background()
        self._draw_grid()
        self._draw_obstacles(state)
        self._draw_no_fly_zones(state)
        self._draw_landmarks(state)
        self._draw_drone(state)
        self._draw_sidebar(state)

        if self.render_mode == "human":
            assert self.window is not None
            self.window.blit(self.surface, (0, 0))
            pygame.event.pump()
            pygame.display.update()
            if self.clock is not None:
                self.clock.tick(self.config.render_fps)
            return None

        frame = pygame.surfarray.array3d(self.surface)
        return np.transpose(frame, (1, 0, 2))

    def close(self) -> None:
        if pygame is None:
            return
        if self.window is not None:
            pygame.display.quit()
        pygame.quit()

    def _scale(self, point: list[float] | tuple[float, float]) -> tuple[int, int]:
        scale = self.map_size / self.config.world_size
        x = int(point[0] * scale)
        y = int(self.map_size - point[1] * scale)
        return x, y

    def _draw_map_background(self) -> None:
        assert self.surface is not None
        pygame.draw.rect(self.surface, (220, 232, 223), pygame.Rect(0, 0, self.map_size, self.map_size))
        pygame.draw.rect(
            self.surface,
            (59, 76, 92),
            pygame.Rect(0, 0, self.map_size, self.map_size),
            width=4,
            border_radius=8,
        )
        pygame.draw.rect(
            self.surface,
            (248, 245, 239),
            pygame.Rect(self.map_size, 0, self.sidebar_width, self.canvas_size),
        )

    def _draw_grid(self) -> None:
        assert self.surface is not None
        step = self.map_size // 10
        for offset in range(step, self.map_size, step):
            pygame.draw.line(self.surface, (198, 209, 199), (offset, 0), (offset, self.map_size), 1)
            pygame.draw.line(self.surface, (198, 209, 199), (0, offset), (self.map_size, offset), 1)

    def _draw_obstacles(self, state: dict[str, Any]) -> None:
        assert self.surface is not None
        scale = self.map_size / self.config.world_size
        for obstacle in state["obstacles"]:
            rect = pygame.Rect(
                int(obstacle["x"] * scale),
                int(self.map_size - (obstacle["y"] + obstacle["height"]) * scale),
                int(obstacle["width"] * scale),
                int(obstacle["height"] * scale),
            )
            pygame.draw.rect(self.surface, (97, 112, 133), rect, border_radius=4)
            pygame.draw.rect(self.surface, (65, 78, 96), rect, width=2, border_radius=4)

    def _draw_no_fly_zones(self, state: dict[str, Any]) -> None:
        assert self.surface is not None
        for zone in state["no_fly_zones"]:
            center = self._scale((zone["x"], zone["y"]))
            radius = int(zone["radius"] * self.map_size / self.config.world_size)
            overlay = pygame.Surface((radius * 2 + 4, radius * 2 + 4), pygame.SRCALPHA)
            pygame.draw.circle(overlay, (212, 79, 79, 80), (radius + 2, radius + 2), radius)
            pygame.draw.circle(overlay, (179, 34, 34), (radius + 2, radius + 2), radius, width=3)
            self.surface.blit(overlay, (center[0] - radius - 2, center[1] - radius - 2))

    def _draw_landmarks(self, state: dict[str, Any]) -> None:
        assert self.surface is not None
        self._draw_base(state["base_position"])
        self._draw_patient(state["patient_position"])
        self._draw_charge_station(state["charge_position"])

    def _draw_base(self, position: list[float]) -> None:
        assert self.surface is not None
        x, y = self._scale(position)
        rect = pygame.Rect(x - 18, y - 18, 36, 36)
        pygame.draw.rect(self.surface, (54, 109, 179), rect, border_radius=6)
        pygame.draw.rect(self.surface, (35, 74, 126), rect, width=2, border_radius=6)
        text = self.small_font.render("BASE", True, (255, 255, 255))
        self.surface.blit(text, text.get_rect(center=(x, y)))

    def _draw_patient(self, position: list[float]) -> None:
        assert self.surface is not None
        x, y = self._scale(position)
        pygame.draw.circle(self.surface, (73, 168, 95), (x, y), 18)
        pygame.draw.circle(self.surface, (40, 115, 57), (x, y), 18, width=2)
        pygame.draw.line(self.surface, (255, 255, 255), (x - 8, y), (x + 8, y), 4)
        pygame.draw.line(self.surface, (255, 255, 255), (x, y - 8), (x, y + 8), 4)

    def _draw_charge_station(self, position: list[float]) -> None:
        assert self.surface is not None
        x, y = self._scale(position)
        points = [(x, y - 20), (x + 16, y - 8), (x + 16, y + 8), (x, y + 20), (x - 16, y + 8), (x - 16, y - 8)]
        pygame.draw.polygon(self.surface, (234, 184, 67), points)
        pygame.draw.polygon(self.surface, (160, 116, 17), points, width=2)
        bolt = [(x - 4, y - 10), (x + 2, y - 10), (x - 2, y + 1), (x + 5, y + 1), (x - 5, y + 12), (x - 1, y + 3)]
        pygame.draw.polygon(self.surface, (255, 255, 255), bolt)

    def _draw_drone(self, state: dict[str, Any]) -> None:
        assert self.surface is not None
        x, y = self._scale(state["agent_position"])
        pygame.draw.circle(self.surface, (48, 63, 86), (x, y), 14)
        for dx, dy in [(-20, -20), (20, -20), (-20, 20), (20, 20)]:
            pygame.draw.line(self.surface, (48, 63, 86), (x, y), (x + dx, y + dy), 3)
            pygame.draw.circle(self.surface, (127, 158, 195), (x + dx, y + dy), 8)
        if state["carrying_supplies"]:
            pygame.draw.rect(self.surface, (201, 101, 52), pygame.Rect(x - 10, y + 16, 20, 14), border_radius=3)

    def _draw_sidebar(self, state: dict[str, Any]) -> None:
        assert self.surface is not None
        origin_x = self.map_size + 20
        lines = [
            ("Mission", "Emergency Medical Drone Delivery"),
            ("Status", state["mission_status"].replace("_", " ").title()),
            ("Battery", f"{state['battery']:.1f}%"),
            ("Step", f"{state['episode_step']} / {state['max_steps']}"),
            ("Action", state["last_action"]),
            ("Reward", f"{state['last_reward']:.2f}"),
            ("Payload", "Onboard" if state["carrying_supplies"] else "At Base"),
            ("Wind", f"({state['wind_vector'][0]:.2f}, {state['wind_vector'][1]:.2f})"),
        ]

        title = self.font.render("Mission Telemetry", True, (34, 45, 58))
        self.surface.blit(title, (origin_x, 24))

        y = 70
        for label, value in lines:
            label_surface = self.small_font.render(label.upper(), True, (118, 128, 138))
            value_surface = self.small_font.render(value, True, (42, 53, 67))
            self.surface.blit(label_surface, (origin_x, y))
            self.surface.blit(value_surface, (origin_x, y + 20))
            y += 62

        legend_title = self.small_font.render("Legend", True, (34, 45, 58))
        self.surface.blit(legend_title, (origin_x, y + 10))
        legend_items = [
            ((54, 109, 179), "Dispatch base"),
            ((73, 168, 95), "Patient location"),
            ((234, 184, 67), "Charge station"),
            ((97, 112, 133), "Urban obstacle"),
            ((212, 79, 79), "No-fly zone"),
        ]
        y += 42
        for color, text in legend_items:
            pygame.draw.circle(self.surface, color, (origin_x + 10, y + 8), 8)
            text_surface = self.small_font.render(text, True, (42, 53, 67))
            self.surface.blit(text_surface, (origin_x + 28, y))
            y += 30
