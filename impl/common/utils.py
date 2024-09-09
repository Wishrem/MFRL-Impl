from dataclasses import dataclass

import numpy as np
import gymnasium as gym

from .policy import Policy


@dataclass
class SARPair:
    """State-Action-Reward Pair"""

    state: np.ndarray
    action: int
    reward: float

    def __post_init__(self):
        self.state = np.array(self.state)

    def __eq__(self, other) -> bool:
        if not isinstance(other, SARPair):
            return False
        if self.state.shape != other.state.shape:
            return False
        return (
            (self.state == other.state).all()
            and self.action == other.action
            and bool(np.isclose(self.reward, other.reward))
        )


def get_episode(
    loc: tuple[int, int],
    action: int,
    env: gym.Env,
    policy: Policy,
    episode_length: int = 100,
) -> list[SARPair]:
    size: tuple[int, int] = env.unwrapped.size  # type: ignore
    if not (0 <= loc[0] < size[0] and 0 <= loc[1] < size[1]):
        raise ValueError(f"loc should be in [0, {size[0]}) x [0, {size[1]})")

    cur_loc = np.array(loc)
    cur_action = action
    episode = []

    env.reset(options={"agent_loc": loc})
    for _ in range(episode_length):
        obs, rwd_, *_ = env.step(cur_action)

        episode.append(SARPair(cur_loc, cur_action, float(rwd_)))

        cur_loc = obs["agent_loc"]
        cur_action = policy.get_action(cur_loc)

    return episode


def get_return(
    loc: tuple[int, int],
    action: int,
    env: gym.Env,
    policy: Policy,
    episode_length: int = 100,
    gamma: float = 0.9,
) -> float:
    episode = get_episode(loc, action, env, policy, episode_length)
    episode.reverse()

    return_val = 0.0
    for _, sar in enumerate(episode):
        return_val = return_val * gamma + sar.reward

    return return_val
