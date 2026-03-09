from __future__ import annotations

import math

from frogger_lite.env import FroggerLiteEnv, LANE_TYPES


def test_lane_archetypes_cover_all_types_when_possible() -> None:
    env = FroggerLiteEnv(env_profile="v2_arcade", n_lanes=5)
    try:
        env.reset(seed=123)
        assigned = {lane.lane_type for lane in env.lanes}
        assert assigned == set(LANE_TYPES)
    finally:
        env.close()


def test_dash_up_has_cooldown_and_fallback_to_up() -> None:
    env = FroggerLiteEnv(env_profile="v2_arcade", n_lanes=0, max_steps=50)
    try:
        env.reset(seed=42)
        start_row = env.agent_row

        _, _, _, _, info1 = env.step(5)  # dash
        assert env.agent_row == start_row - 2
        assert info1["dash_uses"] == 1
        assert info1["dash_cooldown"] == 10

        row_before = env.agent_row
        _, _, _, _, info2 = env.step(5)  # cooldown active -> fallback up
        assert env.agent_row == row_before - 1
        assert info2["dash_uses"] == 1
        assert info2["dash_cooldown"] == 9
    finally:
        env.close()


def test_combo_increases_and_resets_on_idle() -> None:
    env = FroggerLiteEnv(env_profile="v2_arcade", n_lanes=0, max_steps=50)
    try:
        env.reset(seed=7)

        _, reward1, _, _, info1 = env.step(1)
        assert math.isclose(reward1, 1.1, rel_tol=1e-5)
        assert math.isclose(info1["combo_multiplier"], 1.1, rel_tol=1e-5)

        _, reward2, _, _, info2 = env.step(1)
        assert math.isclose(reward2, 1.2, rel_tol=1e-5)
        assert math.isclose(info2["combo_multiplier"], 1.2, rel_tol=1e-5)

        _, reward3, _, _, info3 = env.step(0)
        assert math.isclose(reward3, -0.02, rel_tol=1e-5)
        assert math.isclose(info3["combo_multiplier"], 1.0, rel_tol=1e-5)
    finally:
        env.close()


def test_event_scheduler_does_not_repeat_same_event_back_to_back() -> None:
    env = FroggerLiteEnv(env_profile="v2_arcade", n_lanes=4, max_steps=500)
    try:
        env.reset(seed=99)
        env.next_event_in = 0.0

        completed_events: list[str] = []
        last_seen = None

        for _ in range(500):
            _, _, done, truncated, _ = env.step(0)
            if env.last_event_name is not None and env.last_event_name != last_seen:
                completed_events.append(env.last_event_name)
                last_seen = env.last_event_name
            if done or truncated:
                break

        assert len(completed_events) >= 2
        for i in range(1, len(completed_events)):
            assert completed_events[i] != completed_events[i - 1]
    finally:
        env.close()


def test_roadblocks_never_spawn_on_start_or_goal_rows() -> None:
    env = FroggerLiteEnv(env_profile="v2_arcade", n_lanes=6)
    try:
        env.reset(seed=5)
        env.active_event_name = "ROADBLOCK_SHIFT"
        env._generate_roadblocks()  # noqa: SLF001 - explicit invariant test

        for row, col in env.roadblock_cells:
            assert row not in {env.start_row, env.goal_row}
            assert 0 <= col < env.grid_width
    finally:
        env.close()


def _rollout_signature(seed: int) -> tuple[list[float], list[str | None]]:
    env = FroggerLiteEnv(env_profile="v2_arcade", n_lanes=4, max_steps=120)
    rewards: list[float] = []
    events: list[str | None] = []
    actions = [1, 4, 1, 3, 0, 5, 1, 0, 4, 1]

    try:
        env.reset(seed=seed)
        for i in range(60):
            a = actions[i % len(actions)]
            _, r, done, truncated, info = env.step(a)
            rewards.append(round(float(r), 4))
            events.append(info.get("event_name"))
            if done or truncated:
                break
    finally:
        env.close()

    return rewards, events


def test_fixed_seed_replay_is_deterministic() -> None:
    r1, e1 = _rollout_signature(2026)
    r2, e2 = _rollout_signature(2026)
    assert r1 == r2
    assert e1 == e2
