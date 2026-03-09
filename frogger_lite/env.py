from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces


@dataclass
class Lane:
    row: int
    direction: int  # -1 (left), +1 (right)
    speed: float
    cars: list[float]  # float x positions for smooth movement


class FroggerLiteEnv(gym.Env[np.ndarray, int]):
    metadata = {
        "render_modes": ["ansi", "rgb_array", "human"],
        "render_fps": 15,
    }

    def __init__(
        self,
        grid_width: int = 10,
        grid_height: int = 12,
        n_lanes: int = 8,
        car_density: float = 0.28,
        max_steps: int = 200,
        render_mode: str | None = None,
        render_fps: int | None = None,
    ) -> None:
        super().__init__()
        if render_mode is not None and render_mode not in self.metadata["render_modes"]:
            raise ValueError(
                f"Unsupported render_mode={render_mode}. "
                f"Expected one of {self.metadata['render_modes']}"
            )

        if grid_width < 6:
            raise ValueError("grid_width must be >= 6")
        if grid_height < 8:
            raise ValueError("grid_height must be >= 8")
        if not (0.05 <= car_density <= 0.6):
            raise ValueError("car_density should be in [0.05, 0.6]")

        self.grid_width = grid_width
        self.grid_height = grid_height
        self.goal_row = 0
        self.start_row = grid_height - 1
        self.max_steps = max_steps
        self.render_mode = render_mode
        self.render_fps = int(render_fps) if render_fps is not None else int(
            self.metadata["render_fps"]
        )
        if self.render_fps <= 0:
            raise ValueError("render_fps must be a positive integer")
        self.car_density = car_density

        # Keep one safe row near goal and one safe row near spawn.
        self.traffic_row_start = 2
        self.traffic_row_end = grid_height - 3
        self.traffic_rows = list(range(self.traffic_row_start, self.traffic_row_end + 1))

        if n_lanes > len(self.traffic_rows):
            raise ValueError(
                f"n_lanes={n_lanes} is too high for grid_height={grid_height}; "
                f"max is {len(self.traffic_rows)}"
            )

        self.n_lanes = n_lanes
        self.max_lane_speed = 0.6
        self.collision_threshold = 0.45

        # 0=stay, 1=up, 2=down, 3=left, 4=right
        self.action_space = spaces.Discrete(5)

        # Observation shape (16,):
        # [row_norm, col_norm, dist_goal_norm, step_frac]
        # + 3 x lane_features(row): [danger, lane_dir, lane_speed_norm, occupied]
        self.observation_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(16,),
            dtype=np.float32,
        )

        self.np_random = np.random.default_rng()
        self.agent_row = self.start_row
        self.agent_col = self.grid_width // 2
        self.best_row_reached = self.start_row
        self.step_count = 0
        self.lanes: list[Lane] = []
        self._lane_by_row: dict[int, Lane] = {}
        self._window: Any | None = None
        self._clock: Any | None = None
        self._font_small: Any | None = None
        self._font_large: Any | None = None
        self._human_closed = False
        self._human_scale = 2
        self._cell_px = 18
        self._panel_width_px = 300

        # runtime stats for HUD
        self.episode_index = 0
        self.current_episode_reward = 0.0
        self.last_episode_reward = 0.0
        self.last_outcome = "running"
        self.total_goals = 0
        self.total_collisions = 0
        self.completed_episodes = 0

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        if self.step_count > 0 or self.current_episode_reward != 0:
            self.last_episode_reward = self.current_episode_reward
        self.episode_index += 1

        self.agent_row = self.start_row
        self.agent_col = self.grid_width // 2
        self.best_row_reached = self.start_row
        self.step_count = 0
        self.current_episode_reward = 0.0
        self.last_outcome = "running"

        selected_rows = self.traffic_rows[: self.n_lanes]
        self.lanes = []

        for idx, row in enumerate(selected_rows):
            direction = 1 if idx % 2 == 0 else -1
            speed = float(self.np_random.uniform(0.25, self.max_lane_speed))
            car_count = max(1, int(round(self.grid_width * self.car_density)))
            cars = [float(self.np_random.uniform(0, self.grid_width - 1)) for _ in range(car_count)]
            self.lanes.append(Lane(row=row, direction=direction, speed=speed, cars=cars))

        self._lane_by_row = {lane.row: lane for lane in self.lanes}

        return self._get_obs(), {
            "episode_index": self.episode_index,
            "last_outcome": self.last_outcome,
        }

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        if not self.action_space.contains(action):
            raise ValueError(f"Invalid action: {action}")

        prev_dist = self.agent_row - self.goal_row

        if action == 1:  # up
            self.agent_row -= 1
        elif action == 2:  # down
            self.agent_row += 1
        elif action == 3:  # left
            self.agent_col -= 1
        elif action == 4:  # right
            self.agent_col += 1

        self.agent_row = int(np.clip(self.agent_row, self.goal_row, self.start_row))
        self.agent_col = int(np.clip(self.agent_col, 0, self.grid_width - 1))

        self._move_cars()
        self.step_count += 1

        collision = self._is_collision()
        reached_goal = self.agent_row == self.goal_row

        terminated = collision or reached_goal
        truncated = self.step_count >= self.max_steps and not terminated

        reward = -0.01  # small time penalty

        # Encourage upward progress toward goal.
        new_dist = self.agent_row - self.goal_row
        reward += 0.10 * float(prev_dist - new_dist)

        # Reward first-time best progress.
        if self.agent_row < self.best_row_reached:
            self.best_row_reached = self.agent_row
            reward += 0.20

        if collision:
            reward -= 1.0
        if reached_goal:
            reward += 2.0
        if truncated:
            reward -= 0.2

        self.current_episode_reward += float(reward)
        if collision:
            self.total_collisions += 1
        if reached_goal:
            self.total_goals += 1
        if terminated or truncated:
            self.completed_episodes += 1
            self.last_episode_reward = self.current_episode_reward
            if reached_goal:
                self.last_outcome = "goal"
            elif collision:
                self.last_outcome = "collision"
            else:
                self.last_outcome = "timeout"

        info = {
            "collision": collision,
            "reached_goal": reached_goal,
            "best_row": self.best_row_reached,
            "steps": self.step_count,
            "episode_index": self.episode_index,
            "episode_reward": self.current_episode_reward,
            "last_outcome": self.last_outcome,
        }

        return self._get_obs(), float(reward), terminated, truncated, info

    def render(self) -> np.ndarray | str | None:
        if self.render_mode is None:
            return None
        if self.render_mode == "rgb_array":
            return self._render_rgb_array()
        if self.render_mode == "ansi":
            return self._render_ansi()
        if self.render_mode == "human":
            if self._human_closed:
                return None
            self._render_human()
            return None
        raise ValueError(f"Unsupported render mode: {self.render_mode}")

    def close(self) -> None:
        if self._window is not None:
            try:
                import pygame

                pygame.display.quit()
                pygame.quit()
            except Exception:
                pass
        self._window = None
        self._clock = None
        self._font_small = None
        self._font_large = None
        self._human_closed = True
        return

    def _move_cars(self) -> None:
        for lane in self.lanes:
            moved = []
            for x in lane.cars:
                x_next = (x + lane.direction * lane.speed) % self.grid_width
                moved.append(float(x_next))
            lane.cars = moved

    def _is_collision(self) -> bool:
        lane = self._lane_by_row.get(self.agent_row)
        if lane is None:
            return False

        for x in lane.cars:
            if self._circular_dist(self.agent_col, x) <= self.collision_threshold:
                return True
        return False

    def _circular_dist(self, a: float, b: float) -> float:
        d = abs(a - b)
        return min(d, self.grid_width - d)

    def _lane_features(self, row: int) -> list[float]:
        lane = self._lane_by_row.get(row)
        if lane is None:
            return [0.0, 0.0, 0.0, 0.0]

        nearest = min(self._circular_dist(self.agent_col, x) for x in lane.cars)
        danger = 1.0 - min(nearest / (self.grid_width / 2.0), 1.0)
        occupied = 1.0 if nearest <= self.collision_threshold else 0.0
        speed_norm = min(lane.speed / self.max_lane_speed, 1.0)

        return [float(danger), float(lane.direction), float(speed_norm), float(occupied)]

    def _get_obs(self) -> np.ndarray:
        row_norm = self.agent_row / max(self.start_row, 1)
        col_norm = self.agent_col / max(self.grid_width - 1, 1)
        dist_goal_norm = (self.agent_row - self.goal_row) / max(self.start_row - self.goal_row, 1)
        step_frac = self.step_count / max(self.max_steps, 1)

        features: list[float] = [
            float(row_norm),
            float(col_norm),
            float(dist_goal_norm),
            float(step_frac),
        ]

        # Current row + two rows ahead (toward goal).
        scan_rows = [self.agent_row, self.agent_row - 1, self.agent_row - 2]
        for row in scan_rows:
            features.extend(self._lane_features(row))

        obs = np.asarray(features, dtype=np.float32)
        return np.clip(obs, -1.0, 1.0)

    def _render_ansi(self) -> str:
        grid = [[" " for _ in range(self.grid_width)] for _ in range(self.grid_height)]

        for c in range(self.grid_width):
            grid[self.goal_row][c] = "G"

        for lane in self.lanes:
            for c in range(self.grid_width):
                if grid[lane.row][c] == " ":
                    grid[lane.row][c] = "."

            for x in lane.cars:
                c = int(round(x)) % self.grid_width
                grid[lane.row][c] = "C"

        grid[self.agent_row][self.agent_col] = "A"

        lines = ["+" + "-" * self.grid_width + "+"]
        for row in grid:
            lines.append("|" + "".join(row) + "|")
        lines.append("+" + "-" * self.grid_width + "+")

        return "\n".join(lines)

    def _render_rgb_array(self) -> np.ndarray:
        cell = 18
        h = self.grid_height * cell
        w = self.grid_width * cell

        frame = np.full((h, w, 3), 245, dtype=np.uint8)

        def paint_cell(r: int, c: int, color: tuple[int, int, int]) -> None:
            y0 = r * cell
            y1 = y0 + cell
            x0 = c * cell
            x1 = x0 + cell
            frame[y0:y1, x0:x1] = color

        # Goal row (green).
        for c in range(self.grid_width):
            paint_cell(self.goal_row, c, (172, 220, 160))

        # Traffic rows + cars.
        for lane in self.lanes:
            for c in range(self.grid_width):
                paint_cell(lane.row, c, (220, 220, 220))
            for x in lane.cars:
                c = int(round(x)) % self.grid_width
                paint_cell(lane.row, c, (216, 74, 74))

        # Agent (blue).
        paint_cell(self.agent_row, self.agent_col, (52, 120, 246))

        return frame

    def _render_human(self) -> None:
        try:
            import pygame
        except ImportError as exc:
            raise RuntimeError(
                "render_mode='human' requires pygame. Install with: pip install pygame"
            ) from exc

        cell = self._cell_px * self._human_scale
        pad = 14
        grid_w_px = self.grid_width * cell
        grid_h_px = self.grid_height * cell
        panel_w = self._panel_width_px
        window_w = grid_w_px + panel_w + pad * 3
        window_h = grid_h_px + pad * 2

        if self._window is None:
            pygame.init()
            self._human_closed = False
            self._window = pygame.display.set_mode((window_w, window_h))
            pygame.display.set_caption("Frogger Lite - Live Preview")
            self._clock = pygame.time.Clock()
            self._font_small = pygame.font.SysFont("consolas", 19)
            self._font_large = pygame.font.SysFont("consolas", 28, bold=True)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.close()
                return
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                self.close()
                return

        if self._window is None:
            return

        colors = {
            "bg": (245, 246, 250),
            "safe": (226, 239, 226),
            "start": (201, 229, 210),
            "goal": (164, 218, 165),
            "lane": (77, 84, 94),
            "lane_mark": (205, 212, 219),
            "grid_line": (164, 172, 182),
            "car": (236, 93, 87),
            "car_window": (196, 236, 252),
            "agent": (73, 205, 102),
            "agent_eye": (22, 35, 28),
            "panel_bg": (26, 32, 44),
            "panel_title": (245, 247, 255),
            "panel_text": (209, 217, 232),
            "panel_accent": (88, 176, 255),
            "ok": (88, 214, 134),
            "warn": (255, 211, 105),
            "bad": (255, 118, 117),
        }

        canvas = self._window
        grid_rect = pygame.Rect(pad, pad, grid_w_px, grid_h_px)
        panel_rect = pygame.Rect(grid_rect.right + pad, pad, panel_w, grid_h_px)
        canvas.fill(colors["bg"])

        # Draw row backgrounds.
        for row in range(self.grid_height):
            row_rect = pygame.Rect(grid_rect.left, grid_rect.top + row * cell, grid_w_px, cell)
            if row == self.goal_row:
                color = colors["goal"]
            elif row == self.start_row:
                color = colors["start"]
            elif row in self._lane_by_row:
                color = colors["lane"]
            else:
                color = colors["safe"]
            pygame.draw.rect(canvas, color, row_rect)

            # dashed lane markings for traffic rows
            if row in self._lane_by_row:
                y = row_rect.top + cell // 2
                dash = max(cell // 3, 8)
                gap = max(cell // 4, 6)
                x = row_rect.left + 6
                while x < row_rect.right - 6:
                    pygame.draw.line(
                        canvas,
                        colors["lane_mark"],
                        (x, y),
                        (min(x + dash, row_rect.right - 6), y),
                        2,
                    )
                    x += dash + gap

        # grid lines
        for c in range(self.grid_width + 1):
            x = grid_rect.left + c * cell
            pygame.draw.line(canvas, colors["grid_line"], (x, grid_rect.top), (x, grid_rect.bottom), 1)
        for r in range(self.grid_height + 1):
            y = grid_rect.top + r * cell
            pygame.draw.line(canvas, colors["grid_line"], (grid_rect.left, y), (grid_rect.right, y), 1)

        # goal pads
        for c in range(0, self.grid_width, 2):
            cx = grid_rect.left + c * cell + cell // 2
            cy = grid_rect.top + cell // 2
            pygame.draw.circle(canvas, (129, 197, 112), (cx, cy), max(cell // 7, 4))

        # cars
        for lane in self.lanes:
            row_y = grid_rect.top + lane.row * cell
            for x in lane.cars:
                col = int(round(x)) % self.grid_width
                col_x = grid_rect.left + col * cell

                margin_x = max(cell // 8, 4)
                margin_y = max(cell // 6, 4)
                car_rect = pygame.Rect(
                    col_x + margin_x,
                    row_y + margin_y,
                    cell - 2 * margin_x,
                    cell - 2 * margin_y,
                )
                pygame.draw.rect(canvas, colors["car"], car_rect, border_radius=max(cell // 6, 4))

                window_rect = pygame.Rect(
                    car_rect.left + max(cell // 10, 3),
                    car_rect.top + max(cell // 10, 3),
                    max(car_rect.width // 2, 4),
                    max(car_rect.height // 3, 3),
                )
                pygame.draw.rect(
                    canvas, colors["car_window"], window_rect, border_radius=max(cell // 12, 3)
                )

                wheel_r = max(cell // 10, 2)
                pygame.draw.circle(
                    canvas, (34, 34, 34), (car_rect.left + wheel_r + 2, car_rect.bottom - wheel_r - 1), wheel_r
                )
                pygame.draw.circle(
                    canvas, (34, 34, 34), (car_rect.right - wheel_r - 2, car_rect.bottom - wheel_r - 1), wheel_r
                )

                # direction indicator
                mid_y = car_rect.centery
                if lane.direction > 0:
                    points = [
                        (car_rect.right - 2, mid_y),
                        (car_rect.right - 10, mid_y - 5),
                        (car_rect.right - 10, mid_y + 5),
                    ]
                else:
                    points = [
                        (car_rect.left + 2, mid_y),
                        (car_rect.left + 10, mid_y - 5),
                        (car_rect.left + 10, mid_y + 5),
                    ]
                pygame.draw.polygon(canvas, (255, 244, 184), points)

        # agent (frog-like marker)
        agent_x = grid_rect.left + self.agent_col * cell + cell // 2
        agent_y = grid_rect.top + self.agent_row * cell + cell // 2
        radius = max(cell // 3, 6)
        pygame.draw.circle(canvas, colors["agent"], (agent_x, agent_y), radius)
        eye_offset = max(radius // 2, 3)
        eye_r = max(radius // 6, 2)
        pygame.draw.circle(
            canvas, colors["agent_eye"], (agent_x - eye_offset, agent_y - eye_offset // 2), eye_r
        )
        pygame.draw.circle(
            canvas, colors["agent_eye"], (agent_x + eye_offset, agent_y - eye_offset // 2), eye_r
        )

        # Best row progress marker.
        if self.best_row_reached < self.start_row:
            y = grid_rect.top + self.best_row_reached * cell
            pygame.draw.line(
                canvas,
                colors["panel_accent"],
                (grid_rect.left + 2, y + 2),
                (grid_rect.right - 2, y + 2),
                3,
            )

        pygame.draw.rect(canvas, colors["panel_bg"], panel_rect, border_radius=14)
        pygame.draw.rect(canvas, (42, 51, 69), panel_rect, width=2, border_radius=14)
        pygame.draw.rect(canvas, (42, 51, 69), grid_rect, width=2, border_radius=8)

        if self._font_small is None or self._font_large is None:
            self._font_small = pygame.font.SysFont("consolas", 19)
            self._font_large = pygame.font.SysFont("consolas", 28, bold=True)

        def draw_text(text: str, y: int, color: tuple[int, int, int]) -> int:
            line = self._font_small.render(text, True, color)
            canvas.blit(line, (panel_rect.left + 16, y))
            return y + line.get_height() + 8

        title = self._font_large.render("FROGGER LITE", True, colors["panel_title"])
        y_text = panel_rect.top + 18
        canvas.blit(title, (panel_rect.left + 16, y_text))
        y_text += title.get_height() + 14

        success_rate = self.total_goals / max(self.completed_episodes, 1)
        progress = 1.0 - (self.best_row_reached / max(self.start_row, 1))

        y_text = draw_text(f"Episode:      {self.episode_index}", y_text, colors["panel_text"])
        y_text = draw_text(
            f"Step:         {self.step_count}/{self.max_steps}",
            y_text,
            colors["panel_text"],
        )
        y_text = draw_text(
            f"Ep Reward:    {self.current_episode_reward:7.2f}",
            y_text,
            colors["panel_text"],
        )
        y_text = draw_text(
            f"Last Reward:  {self.last_episode_reward:7.2f}",
            y_text,
            colors["panel_text"],
        )
        y_text += 8
        y_text = draw_text(
            f"Best Progress:{progress * 100:6.1f}%",
            y_text,
            colors["panel_accent"],
        )
        y_text = draw_text(
            f"Goals:        {self.total_goals}",
            y_text,
            colors["ok"],
        )
        y_text = draw_text(
            f"Collisions:   {self.total_collisions}",
            y_text,
            colors["bad"],
        )
        y_text = draw_text(
            f"Success Rate: {success_rate * 100:6.1f}%",
            y_text,
            colors["warn"],
        )
        y_text += 8
        y_text = draw_text(
            f"Agent: ({self.agent_col}, {self.agent_row})",
            y_text,
            colors["panel_text"],
        )
        y_text = draw_text(
            f"Outcome: {self.last_outcome}",
            y_text,
            colors["panel_text"],
        )

        hint = self._font_small.render("ESC / close = hide preview", True, (170, 181, 199))
        canvas.blit(hint, (panel_rect.left + 16, panel_rect.bottom - hint.get_height() - 16))

        pygame.display.flip()
        if self._clock is not None:
            self._clock.tick(self.render_fps)
