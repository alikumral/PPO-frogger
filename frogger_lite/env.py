from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import gymnasium as gym
import numpy as np
from gymnasium import spaces

LaneType = Literal["normal", "burst", "stop_go", "conveyor", "hazard"]
EventName = Literal[
    "EMERGENCY_SWEEP",
    "BLACKOUT",
    "ROADBLOCK_SHIFT",
    "TRAFFIC_SURGE",
]
Profile = Literal["v1_classic", "v2_arcade"]

LANE_TYPES: tuple[LaneType, ...] = (
    "normal",
    "burst",
    "stop_go",
    "conveyor",
    "hazard",
)
EVENT_NAMES: tuple[EventName, ...] = (
    "EMERGENCY_SWEEP",
    "BLACKOUT",
    "ROADBLOCK_SHIFT",
    "TRAFFIC_SURGE",
)


@dataclass
class Lane:
    row: int
    direction: int
    base_speed: float
    cars: list[float]
    lane_type: LaneType = "normal"
    phase_time: float = 0.0
    hazard_active: bool = False
    conveyor_tick: int = 0


class FroggerLiteEnv(gym.Env[np.ndarray, int]):
    metadata = {"render_modes": ["ansi", "rgb_array", "human"], "render_fps": 15}

    def __init__(
        self,
        grid_width: int = 10,
        grid_height: int = 12,
        n_lanes: int = 8,
        car_density: float = 0.28,
        max_steps: int = 200,
        render_mode: str | None = None,
        render_fps: int | None = None,
        env_profile: Profile = "v1_classic",
    ) -> None:
        super().__init__()

        if render_mode is not None and render_mode not in self.metadata["render_modes"]:
            raise ValueError(f"Unsupported render_mode={render_mode}")
        if env_profile not in {"v1_classic", "v2_arcade"}:
            raise ValueError("env_profile must be 'v1_classic' or 'v2_arcade'")
        if grid_width < 6 or grid_height < 8:
            raise ValueError("grid_width >= 6 and grid_height >= 8 required")
        if n_lanes < 0:
            raise ValueError("n_lanes must be >= 0")
        if not (0.05 <= car_density <= 0.6):
            raise ValueError("car_density should be in [0.05, 0.6]")

        self.env_profile = env_profile
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
            raise ValueError("render_fps must be positive")

        self.n_lanes = n_lanes
        self.car_density = car_density
        self.max_lane_speed = 0.6
        self.collision_threshold = 0.45

        self.traffic_row_start = 2
        self.traffic_row_end = grid_height - 3
        self.traffic_rows = list(range(self.traffic_row_start, self.traffic_row_end + 1))
        if n_lanes > len(self.traffic_rows):
            raise ValueError(f"n_lanes max is {len(self.traffic_rows)} for current grid")

        obs_size = 16 if env_profile == "v1_classic" else 26
        self.action_space = spaces.Discrete(5 if env_profile == "v1_classic" else 6)
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(obs_size,), dtype=np.float32
        )

        self.rng = np.random.default_rng()
        self.lanes: list[Lane] = []
        self._lane_by_row: dict[int, Lane] = {}

        self.agent_row = self.start_row
        self.agent_col = self.grid_width // 2
        self.best_row_reached = self.start_row
        self.step_count = 0
        self.episode_index = 0
        self.current_episode_reward = 0.0
        self.last_episode_reward = 0.0
        self.last_outcome = "running"
        self.completed_episodes = 0

        # V2 state
        self.tick_dt = 0.2
        self.max_goals_per_episode = 3
        self.goals_completed = 0
        self.difficulty_level = 0
        self.combo_multiplier = 1.0
        self.combo_sum = 0.0
        self.near_miss_count = 0
        self.dash_uses = 0
        self.dash_cooldown_steps = 0
        self.dash_cooldown_max_steps = 10
        self.event_phase: Literal["idle", "telegraph", "active"] = "idle"
        self.telegraph_event_name: EventName | None = None
        self.active_event_name: EventName | None = None
        self.last_event_name: EventName | None = None
        self.event_time_remaining = 0.0
        self.next_event_in = np.inf
        self.event_active_lane_row: int | None = None
        self.event_interval_reduction = 0.0
        self.events_started = 0
        self.events_survived = 0
        self.roadblock_cells: set[tuple[int, int]] = set()
        self.roadblock_shift_timer = 0.0

        self._window: Any | None = None
        self._clock: Any | None = None
        self._font_small: Any | None = None
        self._font_medium: Any | None = None
        self._font_title: Any | None = None
        self._human_closed = False
        self._last_step_reward = 0.0
        self._step_reward_trace: list[float] = []
        self._episode_reward_history: list[float] = []

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        if self.step_count > 0 or self.current_episode_reward != 0:
            self.last_episode_reward = self.current_episode_reward

        self.episode_index += 1
        self.agent_row = self.start_row
        self.agent_col = self.grid_width // 2
        self.best_row_reached = self.start_row
        self.step_count = 0
        self.current_episode_reward = 0.0
        self.last_outcome = "running"
        self._last_step_reward = 0.0
        self._step_reward_trace = []

        self.goals_completed = 0
        self.difficulty_level = 0
        self.combo_multiplier = 1.0
        self.combo_sum = 0.0
        self.near_miss_count = 0
        self.dash_uses = 0
        self.dash_cooldown_steps = 0
        self.event_phase = "idle"
        self.telegraph_event_name = None
        self.active_event_name = None
        self.last_event_name = None
        self.event_time_remaining = 0.0
        self.next_event_in = self._sample_next_event_interval() if self.env_profile == "v2_arcade" else np.inf
        self.event_active_lane_row = None
        self.event_interval_reduction = 0.0
        self.events_started = 0
        self.events_survived = 0
        self.roadblock_cells = set()
        self.roadblock_shift_timer = 0.0

        self._init_lanes()
        return self._get_obs(), {
            "episode_index": self.episode_index,
            "env_profile": self.env_profile,
            "last_outcome": self.last_outcome,
        }

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        return self._step_v2(action) if self.env_profile == "v2_arcade" else self._step_v1(action)

    def render(self) -> np.ndarray | str | None:
        if self.render_mode is None:
            return None
        if self.render_mode == "ansi":
            return self._render_ansi()
        if self.render_mode == "rgb_array":
            return self._render_rgb_array()
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
        self._font_medium = None
        self._font_title = None
        self._human_closed = True

    def _step_v1(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        if not self.action_space.contains(action):
            raise ValueError(f"Invalid action: {action}")

        prev_dist = self.agent_row - self.goal_row
        if action == 1:
            self.agent_row -= 1
        elif action == 2:
            self.agent_row += 1
        elif action == 3:
            self.agent_col -= 1
        elif action == 4:
            self.agent_col += 1

        self.agent_row = int(np.clip(self.agent_row, self.goal_row, self.start_row))
        self.agent_col = int(np.clip(self.agent_col, 0, self.grid_width - 1))
        self.step_count += 1
        self._move_cars_v1()

        collision = self._is_collision_v1()
        reached_goal = self.agent_row == self.goal_row
        terminated = collision or reached_goal
        truncated = self.step_count >= self.max_steps and not terminated

        reward = -0.01 + 0.10 * float(prev_dist - (self.agent_row - self.goal_row))
        if self.agent_row < self.best_row_reached:
            self.best_row_reached = self.agent_row
            reward += 0.20
        if collision:
            reward -= 1.0
        if reached_goal:
            reward += 2.0
        if truncated:
            reward -= 0.2

        return self._finalize_step(reward, collision, reached_goal, terminated, truncated, False)

    def _step_v2(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        if not self.action_space.contains(action):
            raise ValueError(f"Invalid action: {action}")
        if self.dash_cooldown_steps > 0:
            self.dash_cooldown_steps -= 1

        prev_row = self.agent_row
        prev_best_row = self.best_row_reached
        if action == 1:
            self.agent_row -= 1
        elif action == 2:
            self.agent_row += 1
        elif action == 3:
            self.agent_col -= 1
        elif action == 4:
            self.agent_col += 1
        elif action == 5:
            if self.dash_cooldown_steps == 0:
                self.agent_row -= 2
                self.dash_cooldown_steps = self.dash_cooldown_max_steps
                self.dash_uses += 1
            else:
                self.agent_row -= 1

        self.agent_row = int(np.clip(self.agent_row, self.goal_row, self.start_row))
        self.agent_col = int(np.clip(self.agent_col, 0, self.grid_width - 1))
        self.step_count += 1

        self._update_lane_phases(self.tick_dt)
        self._advance_event_director(self.tick_dt)
        self._apply_conveyor_drift()
        if self.agent_row < self.best_row_reached:
            self.best_row_reached = self.agent_row
        round_best_row = self.best_row_reached
        near_miss = self._move_cars_v2()

        scoring_row = self.agent_row
        signed_progress = prev_row - scoring_row
        backward_rows = max(0, scoring_row - prev_row)
        frontier_gain = max(0, prev_best_row - round_best_row)
        collision = self._is_collision_v2()
        reached_goal = scoring_row == self.goal_row and not collision

        if near_miss and not collision:
            self.near_miss_count += 1

        final_goal = False
        if reached_goal:
            self.goals_completed += 1
            final_goal = self.goals_completed >= self.max_goals_per_episode
            if not final_goal:
                self._increase_difficulty_after_goal()
                self.agent_row = self.start_row
                self.agent_col = self.grid_width // 2
                self.best_row_reached = self.start_row

        if collision or backward_rows > 0 or action == 0:
            self.combo_multiplier = 1.0
        elif frontier_gain > 0:
            self.combo_multiplier = min(2.0, self.combo_multiplier + 0.1)

        reward = -0.01
        if frontier_gain > 0:
            reward += float(frontier_gain) * self.combo_multiplier
        if backward_rows > 0:
            reward -= 0.25 * float(backward_rows)
        if near_miss and not collision and signed_progress > 0:
            reward += 0.2
        if reached_goal:
            reward += 3.0
        if action == 0:
            reward -= 0.02
        if collision:
            reward -= 1.2

        terminated = collision or final_goal
        truncated = self.step_count >= self.max_steps and not terminated
        return self._finalize_step(reward, collision, reached_goal, terminated, truncated, near_miss and not collision)

    def _finalize_step(
        self,
        reward: float,
        collision: bool,
        reached_goal: bool,
        terminated: bool,
        truncated: bool,
        near_miss: bool,
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        step_reward = float(reward)
        self.current_episode_reward += step_reward
        self.combo_sum += self.combo_multiplier
        self._last_step_reward = step_reward
        self._step_reward_trace.append(step_reward)
        if len(self._step_reward_trace) > 160:
            self._step_reward_trace = self._step_reward_trace[-160:]

        if terminated or truncated:
            self.completed_episodes += 1
            self.last_episode_reward = self.current_episode_reward
            self._episode_reward_history.append(float(self.current_episode_reward))
            if len(self._episode_reward_history) > 80:
                self._episode_reward_history = self._episode_reward_history[-80:]
            if collision:
                self.last_outcome = "collision"
            elif reached_goal and (self.env_profile == "v1_classic" or self.goals_completed >= self.max_goals_per_episode):
                self.last_outcome = "goal"
            else:
                self.last_outcome = "timeout"

        info = {
            "collision": collision,
            "reached_goal": reached_goal,
            "steps": self.step_count,
            "best_row": self.best_row_reached,
            "episode_index": self.episode_index,
            "episode_reward": self.current_episode_reward,
            "last_outcome": self.last_outcome,
            "env_profile": self.env_profile,
            "near_miss": near_miss,
            "near_miss_count": self.near_miss_count,
            "combo_multiplier": self.combo_multiplier,
            "combo_avg": self.combo_sum / max(self.step_count, 1),
            "goals_completed": self.goals_completed,
            "dash_uses": self.dash_uses,
            "dash_cooldown": self.dash_cooldown_steps,
            "events_started": self.events_started,
            "events_survived": self.events_survived,
            "event_phase": self.event_phase,
            "event_name": self.active_event_name or self.telegraph_event_name,
            "event_time_remaining": self.event_time_remaining if self.event_phase != "idle" else 0.0,
        }
        return self._get_obs(), float(reward), terminated, truncated, info

    def _init_lanes(self) -> None:
        rows = self.traffic_rows[: self.n_lanes]
        lane_types = self._lane_types_for_count(len(rows)) if self.env_profile == "v2_arcade" else ["normal"] * len(rows)
        self.lanes = []
        for i, row in enumerate(rows):
            lane = Lane(
                row=row,
                direction=1 if i % 2 == 0 else -1,
                base_speed=float(self.rng.uniform(0.25, self.max_lane_speed)),
                cars=[float(self.rng.uniform(0, self.grid_width - 1)) for _ in range(max(1, int(round(self.grid_width * self.car_density))))],
                lane_type=lane_types[i],
            )
            if lane.lane_type == "burst":
                lane.phase_time = float(self.rng.uniform(0.0, 5.0))
            elif lane.lane_type == "stop_go":
                lane.phase_time = float(self.rng.uniform(0.0, 3.2))
            elif lane.lane_type == "hazard":
                lane.phase_time = float(self.rng.uniform(0.0, 2.4))
            elif lane.lane_type == "conveyor":
                lane.conveyor_tick = int(self.rng.integers(0, 3))
            self.lanes.append(lane)
        self._lane_by_row = {lane.row: lane for lane in self.lanes}

    def _lane_types_for_count(self, lane_count: int) -> list[LaneType]:
        if lane_count == 0:
            return []
        shuffled = list(LANE_TYPES)
        self.rng.shuffle(shuffled)
        if lane_count <= len(shuffled):
            return shuffled[:lane_count]
        out = shuffled[:]
        for _ in range(lane_count - len(shuffled)):
            out.append(LANE_TYPES[int(self.rng.integers(0, len(LANE_TYPES)))])
        return out

    def _move_cars_v1(self) -> None:
        for lane in self.lanes:
            lane.cars = [float((x + lane.direction * lane.base_speed) % self.grid_width) for x in lane.cars]

    def _move_cars_v2(self) -> bool:
        near_miss = False
        for lane in self.lanes:
            speed = self._effective_lane_speed(lane)
            if speed <= 0:
                continue
            moved: list[float] = []
            for old_x in lane.cars:
                new_x = float((old_x + lane.direction * speed) % self.grid_width)
                moved.append(new_x)
                if lane.row == self.agent_row:
                    threshold = self._collision_threshold_for_lane(lane)
                    near_band = threshold + 0.8
                    old_d = self._circular_dist(self.agent_col, old_x)
                    new_d = self._circular_dist(self.agent_col, new_x)
                    crossed = (old_d > near_band and new_d <= near_band) or (old_d <= near_band and new_d > near_band)
                    if crossed and min(old_d, new_d) > threshold:
                        near_miss = True
            lane.cars = moved
        return near_miss

    def _effective_lane_speed(self, lane: Lane) -> float:
        speed = lane.base_speed * (1.0 + 0.08 * self.difficulty_level)
        if lane.lane_type == "burst":
            speed *= 0.5 if (lane.phase_time % 5.0) < 2.5 else 1.8
        elif lane.lane_type == "stop_go":
            if (lane.phase_time % 3.2) >= 2.0:
                speed = 0.0
        if self.active_event_name == "TRAFFIC_SURGE":
            speed *= 1.3
        if self.active_event_name == "EMERGENCY_SWEEP" and self.event_active_lane_row is not None and lane.row == self.event_active_lane_row:
            speed *= 2.2
        return float(speed)

    def _update_lane_phases(self, dt: float) -> None:
        for lane in self.lanes:
            lane.phase_time += dt
            if lane.lane_type == "hazard":
                lane.hazard_active = (lane.phase_time % 2.4) < 1.2
            if lane.lane_type == "conveyor":
                lane.conveyor_tick += 1

    def _apply_conveyor_drift(self) -> None:
        lane = self._lane_by_row.get(self.agent_row)
        if lane and lane.lane_type == "conveyor" and lane.conveyor_tick % 3 == 0:
            self.agent_col = int(np.clip(self.agent_col + lane.direction, 0, self.grid_width - 1))

    def _collision_threshold_for_lane(self, lane: Lane) -> float:
        threshold = self.collision_threshold
        if lane.lane_type == "hazard" and lane.hazard_active:
            threshold *= 1.25
        return threshold

    def _is_collision_v1(self) -> bool:
        lane = self._lane_by_row.get(self.agent_row)
        return bool(lane and any(self._circular_dist(self.agent_col, x) <= self.collision_threshold for x in lane.cars))

    def _is_collision_v2(self) -> bool:
        if (self.agent_row, self.agent_col) in self.roadblock_cells:
            return True
        lane = self._lane_by_row.get(self.agent_row)
        if lane is None:
            return False
        threshold = self._collision_threshold_for_lane(lane)
        return any(self._circular_dist(self.agent_col, x) <= threshold for x in lane.cars)

    def _increase_difficulty_after_goal(self) -> None:
        self.difficulty_level += 1
        self.event_interval_reduction = min(6.0, self.event_interval_reduction + 1.0)
        if self.event_phase == "idle":
            self.next_event_in = min(self.next_event_in, self._sample_next_event_interval())

    def _sample_next_event_interval(self) -> float:
        low = max(8.0, 12.0 - self.event_interval_reduction)
        high = max(low + 1.0, 18.0 - self.event_interval_reduction)
        return float(self.rng.uniform(low, high))

    def _sample_event(self) -> EventName:
        candidates = [e for e in EVENT_NAMES if e != self.last_event_name]
        return candidates[int(self.rng.integers(0, len(candidates)))]

    def _advance_event_director(self, dt: float) -> None:
        if self.env_profile != "v2_arcade":
            return
        if self.event_phase == "idle":
            self.next_event_in -= dt
            if self.next_event_in <= 0:
                self.event_phase = "telegraph"
                self.telegraph_event_name = self._sample_event()
                self.event_time_remaining = 1.5
            return
        if self.event_phase == "telegraph":
            self.event_time_remaining -= dt
            if self.event_time_remaining <= 0:
                self.event_phase = "active"
                self.active_event_name = self.telegraph_event_name
                self.telegraph_event_name = None
                self.event_time_remaining = float(self.rng.uniform(4.0, 6.0))
                self.events_started += 1
                self._activate_event()
            return

        self.event_time_remaining -= dt
        if self.active_event_name == "ROADBLOCK_SHIFT":
            self.roadblock_shift_timer += dt
            if self.roadblock_shift_timer >= 1.0:
                self.roadblock_shift_timer = 0.0
                self._generate_roadblocks()
        if self.event_time_remaining <= 0:
            self.events_survived += 1
            if self.active_event_name is not None:
                self.last_event_name = self.active_event_name
            self._deactivate_event()
            self.event_phase = "idle"
            self.next_event_in = self._sample_next_event_interval()

    def _activate_event(self) -> None:
        self.event_active_lane_row = None
        self.roadblock_cells = set()
        self.roadblock_shift_timer = 0.0
        if self.active_event_name == "EMERGENCY_SWEEP" and self.lanes:
            self.event_active_lane_row = self.lanes[int(self.rng.integers(0, len(self.lanes)))].row
        elif self.active_event_name == "ROADBLOCK_SHIFT":
            self._generate_roadblocks()

    def _deactivate_event(self) -> None:
        self.active_event_name = None
        self.event_time_remaining = 0.0
        self.event_active_lane_row = None
        self.roadblock_cells = set()
        self.roadblock_shift_timer = 0.0

    def _generate_roadblocks(self) -> None:
        self.roadblock_cells = set()
        rows = [r for r in self.traffic_rows if r not in {self.goal_row, self.start_row}]
        if not rows:
            return
        chosen_rows = self.rng.choice(rows, size=min(len(rows), max(2, self.grid_height // 4)), replace=False)
        for row in chosen_rows:
            cols = self.rng.choice(np.arange(self.grid_width), size=min(2, max(1, self.grid_width // 6)), replace=False)
            for col in cols:
                self.roadblock_cells.add((int(row), int(col)))

    def _get_obs(self) -> np.ndarray:
        return self._get_obs_v2() if self.env_profile == "v2_arcade" else self._get_obs_v1()

    def _get_obs_v1(self) -> np.ndarray:
        base = self._base_obs(False)
        return np.clip(np.asarray(base, dtype=np.float32), -1.0, 1.0)

    def _get_obs_v2(self) -> np.ndarray:
        blackout = self.active_event_name == "BLACKOUT" and self.event_phase == "active"
        features = self._base_obs(blackout)
        event_oh = [0.0, 0.0, 0.0, 0.0]
        if self.active_event_name in EVENT_NAMES:
            event_oh[EVENT_NAMES.index(self.active_event_name)] = 1.0
        features.extend(event_oh)
        if self.event_phase == "telegraph":
            features.append(float(np.clip(self.event_time_remaining / 1.5, 0.0, 1.0)))
        elif self.event_phase == "active":
            features.append(float(np.clip(self.event_time_remaining / 6.0, 0.0, 1.0)))
        else:
            features.append(0.0)
        features.append(float(self.dash_cooldown_steps / max(self.dash_cooldown_max_steps, 1)))
        scan_rows = [self.agent_row, self.agent_row - 1, self.agent_row - 2]
        for row in scan_rows:
            lane = self._lane_by_row.get(row)
            if lane is None:
                features.append(0.0)
            else:
                idx = LANE_TYPES.index(lane.lane_type)
                features.append(float((idx / (len(LANE_TYPES) - 1)) * 2.0 - 1.0))
        features.append(1.0 if (self.agent_row, self.agent_col) in self.roadblock_cells else 0.0)
        return np.clip(np.asarray(features, dtype=np.float32), -1.0, 1.0)

    def _base_obs(self, hide_hints: bool) -> list[float]:
        row_norm = self.agent_row / max(self.start_row, 1)
        col_norm = self.agent_col / max(self.grid_width - 1, 1)
        dist_goal_norm = (self.agent_row - self.goal_row) / max(self.start_row - self.goal_row, 1)
        step_frac = self.step_count / max(self.max_steps, 1)
        features = [float(row_norm), float(col_norm), float(dist_goal_norm), float(step_frac)]
        for row in [self.agent_row, self.agent_row - 1, self.agent_row - 2]:
            lane = self._lane_by_row.get(row)
            if lane is None:
                features.extend([0.0, 0.0, 0.0, 0.0])
                continue
            nearest = min(self._circular_dist(self.agent_col, x) for x in lane.cars)
            danger = 1.0 - min(nearest / (self.grid_width / 2.0), 1.0)
            occ_th = self._collision_threshold_for_lane(lane) if self.env_profile == "v2_arcade" else self.collision_threshold
            occupied = 1.0 if nearest <= occ_th else 0.0
            if hide_hints:
                lane_dir = 0.0
                lane_speed = 0.0
            else:
                lane_dir = float(lane.direction)
                lane_speed = min(
                    (self._effective_lane_speed(lane) if self.env_profile == "v2_arcade" else lane.base_speed)
                    / (self.max_lane_speed * 2.5),
                    1.0,
                )
            features.extend([float(danger), lane_dir, float(lane_speed), float(occupied)])
        return features

    def _circular_dist(self, a: float, b: float) -> float:
        d = abs(a - b)
        return min(d, self.grid_width - d)

    def _render_ansi(self) -> str:
        grid = [[" " for _ in range(self.grid_width)] for _ in range(self.grid_height)]
        for c in range(self.grid_width):
            grid[self.goal_row][c] = "G"
        for lane in self.lanes:
            lane_char = "."
            if self.env_profile == "v2_arcade":
                lane_char = {"normal": ".", "burst": "b", "stop_go": "s", "conveyor": "v", "hazard": "h"}[lane.lane_type]
            for c in range(self.grid_width):
                if grid[lane.row][c] == " ":
                    grid[lane.row][c] = lane_char
            for x in lane.cars:
                grid[lane.row][int(round(x)) % self.grid_width] = "C"
        for r, c in self.roadblock_cells:
            if 0 <= r < self.grid_height and 0 <= c < self.grid_width:
                grid[r][c] = "#"
        grid[self.agent_row][self.agent_col] = "A"
        lines = ["+" + "-" * self.grid_width + "+"]
        lines.extend("|" + "".join(row) + "|" for row in grid)
        lines.append("+" + "-" * self.grid_width + "+")
        if self.env_profile == "v2_arcade":
            evt = self.active_event_name or self.telegraph_event_name or "none"
            lines.append(
                f"event={evt} phase={self.event_phase} t={self.event_time_remaining:.1f} "
                f"goals={self.goals_completed}/{self.max_goals_per_episode} combo={self.combo_multiplier:.1f}"
            )
        return "\n".join(lines)

    def _render_rgb_array(self) -> np.ndarray:
        cell = 18
        h, w = self.grid_height * cell, self.grid_width * cell
        frame = np.full((h, w, 3), 238, dtype=np.uint8)
        lane_colors = {
            "normal": (205, 209, 217),
            "burst": (226, 201, 150),
            "stop_go": (172, 188, 201),
            "conveyor": (170, 202, 232),
            "hazard": (225, 177, 160),
        }

        def paint(r: int, c: int, color: tuple[int, int, int]) -> None:
            y0, x0 = r * cell, c * cell
            frame[y0 : y0 + cell, x0 : x0 + cell] = color

        for c in range(self.grid_width):
            paint(self.goal_row, c, (146, 210, 142))
        for lane in self.lanes:
            color = lane_colors[lane.lane_type] if self.env_profile == "v2_arcade" else (220, 220, 220)
            if self.env_profile == "v2_arcade":
                if lane.lane_type == "hazard" and lane.hazard_active:
                    color = (236, 140, 122)
                elif lane.lane_type == "burst" and (lane.phase_time % 5.0) >= 2.5:
                    color = (236, 177, 98)
                elif lane.lane_type == "stop_go" and (lane.phase_time % 3.2) >= 2.0:
                    color = (140, 154, 170)
                if (
                    self.active_event_name == "EMERGENCY_SWEEP"
                    and self.event_active_lane_row is not None
                    and lane.row == self.event_active_lane_row
                ):
                    color = (244, 171, 74)
            for c in range(self.grid_width):
                paint(lane.row, c, color)
            for x in lane.cars:
                paint(lane.row, int(round(x)) % self.grid_width, (207, 62, 62))
        for r, c in self.roadblock_cells:
            if 0 <= r < self.grid_height and 0 <= c < self.grid_width:
                paint(r, c, (244, 157, 46))
        paint(self.agent_row, self.agent_col, (66, 128, 240))
        return frame

    def _render_human(self) -> None:
        try:
            import pygame
        except ImportError as exc:
            raise RuntimeError("render_mode='human' requires pygame") from exc

        frame = self._render_rgb_array()
        h, w = frame.shape[0], frame.shape[1]
        scale = 3
        lane_label_w = 90
        board_x = 20 + lane_label_w
        board_y = 82
        board_w = w * scale
        board_h = h * scale
        panel_x = board_x + board_w + 20
        panel_w = 700
        win_w = panel_x + panel_w + 20
        win_h = max(board_y + board_h + 20, 760)

        if self._window is None or self._window.get_size() != (win_w, win_h):
            pygame.init()
            self._window = pygame.display.set_mode((win_w, win_h))
            pygame.display.set_caption("Frogger Lite - V2 HUD Preview")
            self._clock = pygame.time.Clock()
            self._font_small = pygame.font.SysFont("consolas", 16)
            self._font_medium = pygame.font.SysFont("consolas", 22, bold=True)
            self._font_title = pygame.font.SysFont("consolas", 42, bold=True)
            self._human_closed = False

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.close()
                return
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                self.close()
                return

        if self._window is None:
            return

        canvas = self._window
        canvas.fill((11, 16, 27))
        pygame.draw.rect(canvas, (15, 23, 39), pygame.Rect(0, 0, win_w, 64))
        pygame.draw.line(canvas, (50, 67, 93), (0, 64), (win_w, 64), width=2)

        surf = pygame.surfarray.make_surface(np.transpose(frame, (1, 0, 2)))
        surf = pygame.transform.scale(surf, (board_w, board_h))

        board_rect = pygame.Rect(board_x, board_y, board_w, board_h)
        pygame.draw.rect(canvas, (20, 28, 46), board_rect.inflate(8, 8), border_radius=12)
        canvas.blit(surf, (board_x, board_y))
        pygame.draw.rect(canvas, (70, 90, 120), board_rect, width=2, border_radius=6)

        if self._font_small is None:
            self._font_small = pygame.font.SysFont("consolas", 16)
        if self._font_medium is None:
            self._font_medium = pygame.font.SysFont("consolas", 22, bold=True)
        if self._font_title is None:
            self._font_title = pygame.font.SysFont("consolas", 42, bold=True)

        title = self._font_title.render("FROGGER V2", True, (226, 236, 252))
        subtitle = self._font_small.render("arcade preview", True, (155, 176, 206))
        canvas.blit(title, (board_x, 8))
        canvas.blit(subtitle, (board_x, 44))

        cell_px = board_w // self.grid_width
        for c in range(self.grid_width + 1):
            x = board_x + c * cell_px
            pygame.draw.line(canvas, (95, 110, 134), (x, board_y), (x, board_y + board_h), width=1)
        for r in range(self.grid_height + 1):
            y = board_y + r * cell_px
            pygame.draw.line(canvas, (95, 110, 134), (board_x, y), (board_x + board_w, y), width=1)

        blackout_active = self.active_event_name == "BLACKOUT" and self.event_phase == "active"
        for lane in self.lanes:
            lane_y = board_y + lane.row * cell_px + cell_px // 2 - 8
            if blackout_active:
                lane_txt = f"r{lane.row:02d} ???"
            else:
                d = "R" if lane.direction > 0 else "L"
                lane_txt = f"r{lane.row:02d} {lane.lane_type[:3].upper()} {d}"
            txt = self._font_small.render(lane_txt, True, (166, 186, 212))
            canvas.blit(txt, (board_x - lane_label_w, lane_y))

        agent_cx = board_x + int((self.agent_col + 0.5) * cell_px)
        agent_cy = board_y + int((self.agent_row + 0.5) * cell_px)
        pulse = 2.0 + (1.0 + float(np.sin(self.step_count * 0.4))) * 1.6
        pygame.draw.circle(
            canvas,
            (238, 245, 255),
            (agent_cx, agent_cy),
            max(8, cell_px // 2 - 8 + int(round(pulse))),
            width=2,
        )

        event_name = self.active_event_name or self.telegraph_event_name or "NONE"
        event_label = event_name.replace("_", " ").title()
        phase_colors = {
            "idle": (99, 112, 132),
            "telegraph": (235, 188, 84),
            "active": (236, 120, 86),
        }
        banner_color = phase_colors.get(self.event_phase, (99, 112, 132))
        banner_rect = pygame.Rect(panel_x + 4, 14, panel_w - 8, 36)
        pygame.draw.rect(canvas, banner_color, banner_rect, border_radius=10)
        banner_txt = self._font_medium.render(
            f"{event_label} [{self.event_phase.upper()}]", True, (24, 22, 24)
        )
        canvas.blit(banner_txt, (banner_rect.x + 12, banner_rect.y + 4))

        def blit_fit_text(
            text: str,
            font: Any,
            color: tuple[int, int, int],
            x: int,
            y: int,
            max_width: int,
        ) -> None:
            out = text
            if font.size(out)[0] <= max_width:
                canvas.blit(font.render(out, True, color), (x, y))
                return
            ell = "..."
            while out and font.size(out + ell)[0] > max_width:
                out = out[:-1]
            canvas.blit(font.render((out + ell) if out else ell, True, color), (x, y))

        def draw_card(rect: pygame.Rect, title_text: str) -> None:
            pygame.draw.rect(canvas, (20, 28, 45), rect, border_radius=12)
            pygame.draw.rect(canvas, (56, 76, 104), rect, width=1, border_radius=12)
            txt = self._font_medium.render(title_text, True, (224, 233, 246))
            canvas.blit(txt, (rect.x + 12, rect.y + 8))

        def draw_meter(
            x: int,
            y: int,
            width: int,
            label: str,
            value: float,
            fill_color: tuple[int, int, int],
            value_txt: str,
        ) -> int:
            txt = self._font_small.render(f"{label}: {value_txt}", True, (187, 205, 229))
            canvas.blit(txt, (x, y))
            bar_y = y + txt.get_height() + 4
            bar_h = 12
            pygame.draw.rect(canvas, (48, 60, 79), pygame.Rect(x, bar_y, width, bar_h), border_radius=6)
            fill_w = int(round(np.clip(value, 0.0, 1.0) * width))
            if fill_w > 0:
                pygame.draw.rect(
                    canvas,
                    fill_color,
                    pygame.Rect(x, bar_y, fill_w, bar_h),
                    border_radius=6,
                )
            return bar_y + bar_h + 8

        def draw_chart(rect: pygame.Rect, values: list[float]) -> None:
            header_h = 44
            plot = pygame.Rect(
                rect.x + 12,
                rect.y + header_h,
                rect.width - 24,
                rect.height - header_h - 12,
            )
            pygame.draw.rect(canvas, (12, 18, 30), plot, border_radius=8)
            pygame.draw.rect(canvas, (51, 63, 83), plot, width=1, border_radius=8)
            if len(values) < 2:
                txt = self._font_small.render("collecting reward samples...", True, (144, 160, 185))
                canvas.blit(txt, (plot.x + 10, plot.y + plot.height // 2 - txt.get_height() // 2))
                return
            min_v = float(min(values))
            max_v = float(max(values))
            if max_v - min_v < 1e-6:
                max_v = min_v + 1.0
            pts: list[tuple[int, int]] = []
            for i, val in enumerate(values):
                t = i / max(len(values) - 1, 1)
                x = int(round(plot.x + 8 + t * (plot.width - 16)))
                y = int(round(plot.y + 8 + (max_v - val) / (max_v - min_v) * (plot.height - 16)))
                pts.append((x, y))
            if min_v < 0.0 < max_v:
                zero_t = (max_v - 0.0) / (max_v - min_v)
                zero_y = int(round(plot.y + 8 + zero_t * (plot.height - 16)))
                pygame.draw.line(canvas, (95, 107, 130), (plot.x + 8, zero_y), (plot.right - 8, zero_y), 1)
            pygame.draw.lines(canvas, (96, 214, 168), False, pts, width=2)
            last = values[-1]
            lbl = self._font_small.render(
                f"min {min_v:.2f}  max {max_v:.2f}  last {last:.2f}",
                True,
                (167, 183, 210),
            )
            canvas.blit(lbl, (plot.x + 8, plot.bottom - lbl.get_height() - 4))

        row_gap = 12
        col_gap = 16
        col_w = (panel_w - col_gap) // 2
        row1_h = 146
        row2_h = 214
        row1_y = board_y
        row2_y = row1_y + row1_h + row_gap
        row3_y = row2_y + row2_h + row_gap
        row3_h = max(140, board_y + board_h - row3_y)

        run_card = pygame.Rect(panel_x, row1_y, col_w, row1_h)
        draw_card(run_card, "Run State")
        run_lines = [
            f"profile: {self.env_profile}",
            f"ep: {self.episode_index}   step: {self.step_count}/{self.max_steps}",
            f"reward: {self.current_episode_reward:.2f}   dR: {self._last_step_reward:+.2f}",
            f"goals: {self.goals_completed}/{self.max_goals_per_episode}   best_row: {self.best_row_reached}",
        ]
        yy = run_card.y + 38
        for line in run_lines:
            blit_fit_text(line, self._font_small, (188, 205, 230), run_card.x + 12, yy, run_card.width - 24)
            yy += self._font_small.get_height() + 2

        events_card = pygame.Rect(panel_x + col_w + col_gap, row1_y, col_w, row1_h)
        draw_card(events_card, "Events")
        event_lines = [
            f"event: {event_label}",
            f"phase: {self.event_phase}   t: {self.event_time_remaining:.1f}s",
            f"survival: {self.events_survived}/{max(self.events_started, 1)}",
            f"difficulty: {self.difficulty_level}",
            f"near-miss: {self.near_miss_count}",
        ]
        yy = events_card.y + 38
        for line in event_lines:
            blit_fit_text(line, self._font_small, (188, 205, 230), events_card.x + 12, yy, events_card.width - 24)
            yy += self._font_small.get_height() + 2

        bars_card = pygame.Rect(panel_x, row2_y, col_w, row2_h)
        draw_card(bars_card, "Action & Progress")
        round_progress = 1.0 - (self.agent_row - self.goal_row) / max(self.start_row - self.goal_row, 1)
        combo_norm = (self.combo_multiplier - 1.0) / 1.0
        dash_norm = 1.0 - (self.dash_cooldown_steps / max(self.dash_cooldown_max_steps, 1))
        if self.event_phase == "telegraph":
            event_norm = np.clip(self.event_time_remaining / 1.5, 0.0, 1.0)
            event_text = f"telegraph {self.event_time_remaining:.1f}s"
        elif self.event_phase == "active":
            event_norm = np.clip(self.event_time_remaining / 6.0, 0.0, 1.0)
            event_text = f"active {self.event_time_remaining:.1f}s"
        else:
            event_norm = 0.0
            event_text = f"next {self.next_event_in:.1f}s" if self.env_profile == "v2_arcade" else "none"
        yy = bars_card.y + 36
        meter_w = bars_card.width - 24
        yy = draw_meter(
            bars_card.x + 12,
            yy,
            meter_w,
            "round progress",
            float(round_progress),
            (98, 210, 124),
            f"{round_progress * 100:.0f}%",
        )
        yy = draw_meter(
            bars_card.x + 12,
            yy,
            meter_w,
            "combo",
            float(combo_norm),
            (94, 182, 252),
            f"x{self.combo_multiplier:.1f}",
        )
        yy = draw_meter(
            bars_card.x + 12,
            yy,
            meter_w,
            "dash ready",
            float(dash_norm),
            (243, 196, 96),
            f"cd {self.dash_cooldown_steps}",
        )
        _ = draw_meter(bars_card.x + 12, yy, meter_w, "event", float(event_norm), (232, 124, 90), event_text)

        legend_card = pygame.Rect(panel_x + col_w + col_gap, row2_y, col_w, row2_h)
        draw_card(legend_card, "Legend")
        lane_colors = {
            "normal lane": (205, 209, 217),
            "burst lane": (226, 201, 150),
            "stop-go lane": (172, 188, 201),
            "conveyor lane": (170, 202, 232),
            "hazard lane": (225, 177, 160),
            "car": (207, 62, 62),
            "agent": (66, 128, 240),
            "goal": (146, 210, 142),
            "roadblock": (244, 157, 46),
        }
        yy = legend_card.y + 40
        items = list(lane_colors.items())
        col2_x = legend_card.x + legend_card.width // 2 + 2
        col_text_w = legend_card.width // 2 - 34
        for i, (label, color) in enumerate(items):
            block_x = legend_card.x + 12 if i < 5 else col2_x + 4
            text_x = block_x + 22
            row_y = yy + (i if i < 5 else i - 5) * 28
            pygame.draw.rect(canvas, color, pygame.Rect(block_x, row_y + 2, 14, 14), border_radius=3)
            blit_fit_text(label, self._font_small, (188, 205, 230), text_x, row_y, col_text_w)

        chart_card = pygame.Rect(panel_x, row3_y, panel_w, row3_h)
        draw_card(chart_card, "Reward Trace (current episode)")
        draw_chart(chart_card, self._step_reward_trace[-120:])

        hint_txt = self._font_small.render("esc / close = hide preview", True, (164, 182, 207))
        canvas.blit(hint_txt, (board_x, board_y + board_h + 4))

        pygame.display.flip()
        if self._clock is not None:
            self._clock.tick(self.render_fps)
