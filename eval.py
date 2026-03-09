from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO

import frogger_lite  # noqa: F401 - env registration side effect


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate PPO model on FroggerLite-v0")
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("models/best/best_model.zip"),
        help="Path to a .zip SB3 model",
    )
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument(
        "--env-profile",
        type=str,
        default="v1_classic",
        choices=["v1_classic", "v2_arcade"],
        help="Environment profile for evaluation.",
    )
    parser.add_argument(
        "--render-mode",
        type=str,
        default="none",
        choices=["none", "ansi", "human"],
        help="Visualization mode: none, ansi (terminal), human (pygame window).",
    )
    parser.add_argument(
        "--render-fps",
        type=int,
        default=8,
        help="Render FPS for human mode (lower = slower).",
    )
    parser.add_argument("--sleep", type=float, default=0.08)
    parser.add_argument(
        "--metrics-out",
        type=Path,
        default=None,
        help="Optional path to save evaluation metrics as JSON.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.model.exists():
        raise FileNotFoundError(f"Model not found: {args.model}")

    render_mode = None if args.render_mode == "none" else args.render_mode
    env = gym.make(
        "FroggerLite-v0",
        max_steps=args.max_steps,
        render_mode=render_mode,
        render_fps=args.render_fps,
        env_profile=args.env_profile,
    )
    model = PPO.load(str(args.model))

    rewards: list[float] = []
    successes = 0
    total_steps = 0
    total_near_miss = 0
    total_dash_uses = 0
    total_goals = 0
    total_events_started = 0
    total_events_survived = 0
    combo_avgs: list[float] = []

    for ep in range(1, args.episodes + 1):
        obs, _ = env.reset()
        done = False
        ep_reward = 0.0

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(int(action))
            done = terminated or truncated
            ep_reward += reward

            if args.render_mode != "none":
                frame = env.render()
                if args.render_mode == "ansi" and isinstance(frame, str):
                    print("\033[2J\033[H", end="")
                    print(frame)
                    print(f"episode={ep} reward={ep_reward:.2f}")
                    time.sleep(args.sleep)
                elif args.render_mode == "human" and args.sleep > 0:
                    time.sleep(args.sleep)

        rewards.append(ep_reward)
        if info.get("reached_goal", False):
            successes += 1
        total_steps += int(info.get("steps", 0))
        total_near_miss += int(info.get("near_miss_count", 0))
        total_dash_uses += int(info.get("dash_uses", 0))
        total_goals += int(info.get("goals_completed", 0))
        total_events_started += int(info.get("events_started", 0))
        total_events_survived += int(info.get("events_survived", 0))
        combo_avgs.append(float(info.get("combo_avg", 1.0)))

        print(
            f"Episode {ep:02d}: reward={ep_reward:.2f}, "
            f"goal={'yes' if info.get('reached_goal', False) else 'no'}"
        )

    mean_reward = float(np.mean(rewards)) if rewards else 0.0
    success_rate = successes / max(args.episodes, 1)
    near_miss_rate = total_near_miss / max(total_steps, 1)
    combo_avg = float(np.mean(combo_avgs)) if combo_avgs else 1.0
    goals_per_episode = total_goals / max(args.episodes, 1)
    event_survival_rate = total_events_survived / max(total_events_started, 1)
    dash_usage_rate = total_dash_uses / max(total_steps, 1)

    print("\nSummary")
    print(f"Profile:        {args.env_profile}")
    print(f"Average reward: {mean_reward:.3f}")
    print(f"Success rate:   {success_rate:.1%}")
    print(f"Near-miss rate: {near_miss_rate:.3f}")
    print(f"Combo avg:      {combo_avg:.3f}")
    print(f"Goals/episode:  {goals_per_episode:.3f}")
    print(f"Event survive:  {event_survival_rate:.1%}")
    print(f"Dash usage:     {dash_usage_rate:.3f}")

    metrics = {
        "env_profile": args.env_profile,
        "episodes": args.episodes,
        "average_reward": mean_reward,
        "success_rate": success_rate,
        "near_miss_rate": near_miss_rate,
        "combo_avg": combo_avg,
        "goals_per_episode": goals_per_episode,
        "event_survival_rate": event_survival_rate,
        "dash_usage_rate": dash_usage_rate,
        "total_steps": total_steps,
        "model": str(args.model),
    }
    if args.metrics_out is not None:
        args.metrics_out.parent.mkdir(parents=True, exist_ok=True)
        args.metrics_out.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        print(f"Saved metrics to: {args.metrics_out}")

    env.close()


if __name__ == "__main__":
    main()
