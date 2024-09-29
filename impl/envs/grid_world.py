from typing import Any
from dataclasses import dataclass

import numpy as np
import gymnasium as gym
from gymnasium import spaces


@dataclass
class ForbiddenAreaCfg:
    """forbidden area configuration
    locs (list[tuple[int, int]]): list of forbidden area locations, if None, random `num` locations are generated
    num (int): number of forbidden areas, if None, `locs` must be provided

    `locs` has higher priority than `num`
    """

    locs: list[tuple[int, int]] | None = None
    num: int | None = 5  # for random state

    def __post_init__(self):
        if self.locs is None and self.num is None:
            raise ValueError("`locs` and `num` cannot be None at the same time")

    def get_locs(self, size: tuple[int, int]) -> list[tuple[int, int]]:
        """Get forbidden area locations

        Args:
            size (tuple[int, int]): size of the grid world

        Raises:
            ValueError: if location of forbidden area is out of bound

        Returns:
            np.ndarray: forbidden area locations
        """
        if self.locs is None:
            assert self.num is not None
            if self.num >= size[0] * size[1]:
                raise ValueError(
                    "Number of forbidden areas is greater than or equal to the number of grid world locations"
                )
            return self._generate_random_locs(size)
        for x, y in self.locs:
            if (x <= -1 or x >= size[0]) or (y <= -1 or y >= size[1]):
                raise ValueError(
                    f"Location ({x}, {y}) of forbidden area is out of bound"
                )
        if len(self.locs) >= size[0] * size[1]:
            raise ValueError(
                "Number of forbidden areas is greater than or equal to the number of grid world locations"
            )
        return self.locs

    def _generate_random_locs(self, size: tuple[int, int]) -> list[tuple[int, int]]:
        locs = [(i, j) for i in range(size[0]) for j in range(size[1])]
        np.random.shuffle(locs)
        return locs[: self.num]


@dataclass
class RewardCfg:
    out_of_bound: int = -1
    forbidden_area: int = -1
    target: int = 1
    move: int = 0


class GridWorldEnv(gym.Env):
    def __init__(
        self,
        start_loc: tuple[int, int] = (0, 0),
        target_loc: tuple[int, int] | None = None,
        size: int | tuple[int, int] = 5,
        forbidden_area_cfg: ForbiddenAreaCfg = ForbiddenAreaCfg(),
        reward_cfg: RewardCfg = RewardCfg(),
    ):
        """Grid world environment

        Args:
            start_loc (tuple[int, int], optional): the start location for agent. Defaults to (0, 0).
            target_loc (tuple[int, int] | None, optional): the target location. It will not with forbidden area. Defaults to None.
            size (int | tuple[int, int], optional): the size of grid world. If `size` is int type, the grid world is square. Defaults to 5.
            forbidden_area_cfg (ForbiddenAreaCfg, optional): forbidden area configuration. Defaults to ForbiddenAreaCfg().
            reward_cfg (RewardCfg, optional): reward configuration. Defaults to RewardCfg().
        """
        self._size = size if isinstance(size, tuple) else (size, size)
        self._rwd_cfg = reward_cfg
        self._start_loc = np.array(start_loc, dtype=np.int32)
        self._agent_loc = self._start_loc

        forbidden_locs = forbidden_area_cfg.get_locs(self._size)
        if target_loc is None:
            target_loc = self._generate_random_target_loc(forbidden_locs)
        elif target_loc in forbidden_locs:
            raise ValueError(
                "Target location is in forbidden area. If you use random forbidden area, you'd better set forbidden location manually"
            )
        self._target_loc = np.array(target_loc)
        self._forbidden_locs = np.array(forbidden_locs)

        self.observation_space = spaces.Dict(
            {
                "agent_loc": spaces.Box(
                    low=np.array([0, 0]),
                    high=np.array(self._size),
                    shape=(2,),
                    dtype=np.int32,
                ),
            }
        )

        # 0: up, 1: right, 2: down, 3: left, 4: stay
        self.action_space = spaces.Discrete(5)
        self._action_to_direction = {
            0: np.array([-1, 0], dtype=np.int32),
            1: np.array([0, 1], dtype=np.int32),
            2: np.array([1, 0], dtype=np.int32),
            3: np.array([0, -1], dtype=np.int32),
            4: np.array([0, 0], dtype=np.int32),
        }

    def step(self, action: int):
        direction = self._action_to_direction[action]
        new_loc = self._agent_loc + direction
        if np.any(new_loc < 0) or np.any(new_loc >= self._size):
            reward = self._rwd_cfg.out_of_bound
        elif np.all(new_loc == self._target_loc):
            reward = self._rwd_cfg.target
            self._agent_loc = new_loc
        elif np.any(np.all(new_loc == self._forbidden_locs, axis=1)):
            reward = self._rwd_cfg.forbidden_area
            self._agent_loc = new_loc
        else:
            reward = self._rwd_cfg.move
            self._agent_loc = new_loc

        # observation, reward, terminated, truncated, info
        return self._get_obs(), reward, False, False, {}

    def _get_obs(self):
        return {"agent_loc": self._agent_loc}

    def reset(self, seed: int | None = None, options: None | dict[str, Any] = None):
        super().reset(seed=seed)

        if options is None:
            self._agent_loc = self._start_loc
            return self._get_obs(), {}

        if options["agent_loc"] is not None:
            self._agent_loc = np.array(options["agent_loc"], dtype=np.int32)

        return self._get_obs(), {}

    @property
    def target_loc(self):
        return self._target_loc

    @property
    def start_loc(self):
        return self._start_loc

    @property
    def size(self):
        return self._size

    def _generate_random_target_loc(
        self, forbidden_locs: list[tuple[int, int]]
    ) -> tuple[int, int]:
        locs = [(i, j) for i in range(self._size[0]) for j in range(self._size[1])]
        locs = [loc for loc in locs if loc not in forbidden_locs]
        return locs[np.random.randint(len(locs))]
