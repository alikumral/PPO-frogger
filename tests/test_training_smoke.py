from __future__ import annotations

import gymnasium as gym
import pytest

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor

import frogger_lite  # noqa: F401


@pytest.mark.integration
def test_ppo_smoke_train_and_save(tmp_path) -> None:
    def make_env():
        return Monitor(
            gym.make(
                "FroggerLite-v0",
                env_profile="v2_arcade",
                max_steps=80,
                n_lanes=4,
            )
        )

    vec_env = make_vec_env(make_env, n_envs=1, seed=11)
    model = PPO(
        "MlpPolicy",
        vec_env,
        learning_rate=1e-3,
        n_steps=32,
        batch_size=32,
        n_epochs=1,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.015,
        vf_coef=0.5,
        max_grad_norm=0.5,
        seed=11,
        verbose=0,
    )

    try:
        model.learn(total_timesteps=96)
        out = tmp_path / "smoke_model"
        model.save(str(out))
        assert (tmp_path / "smoke_model.zip").exists()
    finally:
        vec_env.close()
