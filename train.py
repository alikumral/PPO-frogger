from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any

import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import (
    BaseCallback,
    CallbackList,
    EvalCallback,
)
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor

import frogger_lite  # noqa: F401 - env registration side effect


class PreviewCallback(BaseCallback):
    """Render a short terminal preview episode every N training timesteps."""

    def __init__(
        self,
        max_steps: int,
        preview_mode: str = "ansi",
        preview_freq: int = 50_000,
        sleep: float = 0.08,
        preview_render_fps: int = 8,
        verbose: int = 1,
    ) -> None:
        super().__init__(verbose=verbose)
        self.max_steps = max_steps
        self.preview_mode = preview_mode
        self.preview_freq = max(preview_freq, 1)
        self.sleep = max(sleep, 0.0)
        self.preview_render_fps = max(int(preview_render_fps), 1)
        self._last_preview = 0
        self._preview_env: gym.Env[Any, Any] | None = None

    def _on_training_start(self) -> None:
        self._preview_env = gym.make(
            "FroggerLite-v0",
            max_steps=self.max_steps,
            render_mode=self.preview_mode,
            render_fps=self.preview_render_fps,
        )

    def _on_step(self) -> bool:
        if self._preview_env is None:
            return True

        if (self.num_timesteps - self._last_preview) < self.preview_freq:
            return True

        self._last_preview = self.num_timesteps
        self._run_preview_episode()
        return True

    def _on_training_end(self) -> None:
        if self._preview_env is not None:
            self._preview_env.close()
            self._preview_env = None

    def _run_preview_episode(self) -> None:
        if self._preview_env is None:
            return

        obs, _ = self._preview_env.reset()
        done = False
        total_reward = 0.0
        step_count = 0
        reached_goal = False

        while not done:
            action, _ = self.model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = self._preview_env.step(
                int(action)
            )
            done = terminated or truncated
            total_reward += reward
            step_count += 1
            reached_goal = bool(info.get("reached_goal", False))

            frame = self._preview_env.render()
            if self.preview_mode == "ansi" and isinstance(frame, str):
                # Clear screen + move cursor to top-left.
                print("\033[2J\033[H", end="")
                print(frame)
                print(
                    f"[preview] timesteps={self.num_timesteps} "
                    f"steps={step_count} reward={total_reward:.2f}"
                )
                time.sleep(self.sleep)
            elif self.preview_mode == "human" and self.sleep > 0:
                time.sleep(self.sleep)

        print(
            f"[preview done] timesteps={self.num_timesteps} "
            f"reward={total_reward:.2f} goal={'yes' if reached_goal else 'no'}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train PPO on FroggerLite-v0")
    parser.add_argument("--total-timesteps", type=int, default=500_000)
    parser.add_argument("--n-envs", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--models-dir", type=Path, default=Path("models"))
    parser.add_argument("--log-dir", type=Path, default=Path("runs"))
    parser.add_argument(
        "--preview",
        action="store_true",
        help="Render a terminal preview episode periodically while training.",
    )
    parser.add_argument(
        "--preview-mode",
        type=str,
        default="ansi",
        choices=["ansi", "human"],
        help="Preview output mode: 'ansi' (terminal) or 'human' (pygame window).",
    )
    parser.add_argument(
        "--preview-freq",
        type=int,
        default=50_000,
        help="Training timesteps between preview episodes.",
    )
    parser.add_argument(
        "--preview-sleep",
        type=float,
        default=0.08,
        help="Delay between preview frames in seconds.",
    )
    parser.add_argument(
        "--preview-render-fps",
        type=int,
        default=8,
        help="Render FPS for human preview window (lower = slower).",
    )
    return parser.parse_args()


def build_env(max_steps: int):
    def _init():
        env = gym.make("FroggerLite-v0", max_steps=max_steps)
        return Monitor(env)

    return _init


def main() -> None:
    args = parse_args()

    args.models_dir.mkdir(parents=True, exist_ok=True)
    args.log_dir.mkdir(parents=True, exist_ok=True)

    vec_env = make_vec_env(
        build_env(args.max_steps),
        n_envs=args.n_envs,
        seed=args.seed,
    )

    eval_env = Monitor(gym.make("FroggerLite-v0", max_steps=args.max_steps))

    model = PPO(
        policy="MlpPolicy",
        env=vec_env,
        learning_rate=3e-4,
        n_steps=1024,
        batch_size=256,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        tensorboard_log=str(args.log_dir),
        seed=args.seed,
        verbose=1,
    )

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(args.models_dir / "best"),
        log_path=str(args.log_dir / "eval"),
        eval_freq=max(10_000 // args.n_envs, 1),
        n_eval_episodes=10,
        deterministic=True,
        render=False,
    )

    callbacks: list[BaseCallback] = [eval_callback]
    if args.preview:
        callbacks.append(
            PreviewCallback(
                max_steps=args.max_steps,
                preview_mode=args.preview_mode,
                preview_freq=args.preview_freq,
                sleep=args.preview_sleep,
                preview_render_fps=args.preview_render_fps,
            )
        )

    callback: BaseCallback = callbacks[0]
    if len(callbacks) > 1:
        callback = CallbackList(callbacks)

    model.learn(total_timesteps=args.total_timesteps, callback=callback)

    final_path = args.models_dir / "ppo_frogger_lite_final"
    model.save(str(final_path))

    vec_env.close()
    eval_env.close()

    print(f"Saved final model to: {final_path}.zip")


if __name__ == "__main__":
    main()
