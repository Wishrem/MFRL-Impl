from .grid_world import GridWorldEnv, ForbiddenAreaCfg, RewardCfg

import gymnasium as gym
from gymnasium.envs.registration import register

register(
    id="gridworld-v0",
    entry_point="envs:GridWorldEnv",
)


def make_gridworld(
    start_loc: tuple[int, int] = (0, 0),
    target_loc: tuple[int, int] | None = None,
    size: int | tuple[int, int] = 5,
    forbidden_area_cfg: ForbiddenAreaCfg = ForbiddenAreaCfg(),
    reward_cfg: RewardCfg = RewardCfg(),
):
    return gym.make(
        "gridworld-v0",
        start_loc=start_loc,
        target_loc=target_loc,
        size=size,
        forbidden_area_cfg=forbidden_area_cfg,
        reward_cfg=reward_cfg,
    )


__all__ = ["GridWorldEnv", "ForbiddenAreaCfg", "RewardCfg", "make_gridworld"]
