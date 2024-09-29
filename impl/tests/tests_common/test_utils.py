import numpy as np

from envs import make_gridworld
from envs.grid_world import ForbiddenAreaCfg, RewardCfg
from common.utils import get_return, get_episode, SARPair
from common.policy import Policy


class TestUtils:
    env = make_gridworld(
        size=2,
        target_loc=(1, 1),
        forbidden_area_cfg=ForbiddenAreaCfg(locs=[(0, 1)]),
        reward_cfg=RewardCfg(move=0, target=1, forbidden_area=-2, out_of_bound=-1),
    )

    def test_get_episode(self):
        env = self.env
        episode_length = 5
        action_stars = np.array([[2, 2], [1, 4]])

        policy = Policy(env, eps=0)
        policy.action_stars = action_stars

        trajectory = get_episode((0, 0), 0, env, policy, episode_length)
        target_trajectory = [
            SARPair(state=(0, 0), action=0, reward=-1.0),
            SARPair(state=(0, 0), action=2, reward=0.0),
            SARPair(state=(1, 0), action=1, reward=1.0),
            SARPair(state=(1, 1), action=4, reward=1.0),
            SARPair(state=(1, 1), action=4, reward=1.0),
        ]
        assert trajectory == target_trajectory

    def test_get_return(self):
        env = self.env
        episode_length = 10
        action_stars = np.array([[2, 2], [1, 4]])

        policy = Policy(env, eps=0)
        policy.action_stars = action_stars
        q_value = get_return((0, 0), 0, env, policy, episode_length, gamma=0.9)
        assert np.isclose(q_value, 3.61, atol=5e-3)

        q_value = get_return((0, 0), 1, env, policy, episode_length, gamma=0.9)
        assert np.isclose(q_value, 3.51, atol=5e-3)

        q_value = get_return((0, 0), 2, env, policy, episode_length, gamma=0.9)
        assert np.isclose(q_value, 5.51, atol=5e-3)

        q_value = get_return((0, 0), 3, env, policy, episode_length, gamma=0.9)
        assert np.isclose(q_value, 3.61, atol=5e-3)

        q_value = get_return((0, 0), 4, env, policy, episode_length, gamma=0.9)
        assert np.isclose(q_value, 4.61, atol=5e-3)
