from gymnasium.envs.registration import register

from frogger_lite.env import FroggerLiteEnv

# Register env once for gym.make("FroggerLite-v0") convenience.
try:
    register(
        id="FroggerLite-v0",
        entry_point="frogger_lite.env:FroggerLiteEnv",
        max_episode_steps=200,
    )
except Exception:
    # Ignore if already registered (re-imports in notebooks/tests).
    pass

__all__ = ["FroggerLiteEnv"]
