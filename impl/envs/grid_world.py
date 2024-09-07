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

    def get_locs(self, size: tuple[int, int], seed: int = 0) -> np.ndarray:
        """Get forbidden area locations

        Args:
            size (tuple[int, int]): size of the grid world
            seed (int, optional): random seed. Defaults to 0.

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
            np.random.seed(seed)
            return self._generate_random_locs(size)
        for x, y in self.locs:
            if (x <= -1 or x >= size[0]) or (y <= -1 or y >= size[1]):
                raise ValueError(
                    f"Location ({x}, {y}) of forbidden area is out of bound"
                )
        if len(self.locs) >= size[0] * size[1] - 1:  # minus one for target location
            raise ValueError(
                "Number of forbidden areas is greater than or equal to the number of grid world locations"
            )
        return np.array(self.locs)

    def _generate_random_locs(self, size: tuple[int, int]) -> np.ndarray:
        locs = [[i, j] for i in range(size[0]) for j in range(size[1])]
        np.random.shuffle(locs)
        return np.array(locs[: self.num])


class GridWorldEnv(gym.Env):

    def __init__(
        self,
        start_loc: tuple[int, int] = (0, 0),
        target_loc: tuple[int, int] | None = None,
        size: int | tuple[int, int] = 5,
        forbidden_area_cfg: ForbiddenAreaCfg = ForbiddenAreaCfg(),
    ):
        """Grid world environment

        Args:
            start_loc (tuple[int, int], optional): the start location for agent. Defaults to (0, 0).
            target_loc (tuple[int, int] | None, optional): the target location. It will not with forbidden area. Defaults to None.
            size (int | tuple[int, int], optional): the size of grid world. If `size` is int type, the grid world is square. Defaults to 5.
            forbidden_area_cfg (ForbiddenAreaCfg, optional): forbidden area configuration. Defaults to ForbiddenAreaCfg().
        """
        self.size = size if isinstance(size, tuple) else (size, size)

        forbidden_locs = forbidden_area_cfg.get_locs(self.size)
        if target_loc is None:
            target_loc = self._generate_random_target_loc(forbidden_locs)
        elif target_loc in forbidden_locs:
            raise ValueError(
                "Target location is in forbidden area. If you use random forbidden area, you'd better set forbidden location manually"
            )
        self._target_loc = np.array(target_loc)
        self._forbidden_locs = forbidden_locs

        # 0: up, 1: right, 2: down, 3: left, 4: stay
        self.action_space = spaces.Discrete(5)
        self._action_to_direction = {
            0: np.array([0, 1]),
            1: np.array([1, 0]),
            2: np.array([0, -1]),
            3: np.array([-1, 0]),
            4: np.array([0, 0]),
        }

    def _generate_random_target_loc(self, forbidden_locs: np.ndarray) -> np.array:
        locs = [[i, j] for i in range(self.size[0]) for j in range(self.size[1])]
        locs = [loc for loc in locs if loc not in forbidden_locs]
        return np.array(locs[np.random.randint(len(locs))])
