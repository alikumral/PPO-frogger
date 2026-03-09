from __future__ import annotations

import argparse
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
    )
    model = PPO.load(str(args.model))

    rewards: list[float] = []
    successes = 0

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

        print(
            f"Episode {ep:02d}: reward={ep_reward:.2f}, "
            f"goal={'yes' if info.get('reached_goal', False) else 'no'}"
        )

    mean_reward = float(np.mean(rewards)) if rewards else 0.0
    success_rate = successes / max(args.episodes, 1)

    print("\nSummary")
    print(f"Average reward: {mean_reward:.3f}")
    print(f"Success rate:   {success_rate:.1%}")

    env.close()


if __name__ == "__main__":
    main()
