from .grid_world import GridWorldEnv, ForbiddenAreaCfg, RewardCfg

from gymnasium.envs.registration import register

register(
    id="gridworld-v0",
    entry_point="envs:GridWorldEnv",
)

__all__ = ["GridWorldEnv", "ForbiddenAreaCfg", "RewardCfg"]