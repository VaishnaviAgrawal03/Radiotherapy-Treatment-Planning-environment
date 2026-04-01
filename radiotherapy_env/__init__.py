"""
RadiotherapyPlanningEnv
=======================
A Gymnasium-compatible RL environment for cancer radiotherapy treatment planning.

Quick start:
    import gymnasium as gym
    import radiotherapy_env

    env = gym.make("RadiotherapyEnv-prostate-v1", render_mode="rgb_array")
    obs, info = env.reset(seed=42)

    for _ in range(50):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            break

    env.close()
"""

from .env import RadiotherapyEnv
import gymnasium as gym

# Register all three difficulty variants
gym.register(
    id="RadiotherapyEnv-prostate-v1",
    entry_point="radiotherapy_env:RadiotherapyEnv",
    kwargs={"task": "prostate", "max_steps": 50},
    max_episode_steps=50,
)

gym.register(
    id="RadiotherapyEnv-headneck-v1",
    entry_point="radiotherapy_env:RadiotherapyEnv",
    kwargs={"task": "head_neck", "max_steps": 60},
    max_episode_steps=60,
)

gym.register(
    id="RadiotherapyEnv-pediatricbrain-v1",
    entry_point="radiotherapy_env:RadiotherapyEnv",
    kwargs={"task": "pediatric_brain", "max_steps": 70},
    max_episode_steps=70,
)

__version__ = "1.0.0"
__all__ = ["RadiotherapyEnv"]
