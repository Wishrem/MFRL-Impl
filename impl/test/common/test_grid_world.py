import numpy as np
import envs  # for registering the environment
import gymnasium as gym

from test.utils import is_same_array
from envs.grid_world import ForbiddenAreaCfg, RewardCfg
from common.grid_world import Policy, get_return


class TestPolicy:
    env = gym.make("gridworld-v0", size=5)

    def test_init(self):
        policy = Policy(self.env, eps=0.1)
        assert policy.size == (5, 5)
        assert policy.action_stars.shape == (5, 5)

    def test_get_action_probs(self):
        # non-greedy
        policy = Policy(self.env, eps=0)
        policy.action_stars = np.array([[0, 1], [2, 3]])
        actions, probs = policy.get_action_probs((0, 0))
        assert is_same_array(actions, [0, 1, 2, 3, 4], True)
        assert is_same_array(probs, [1, 0, 0, 0, 0], True)

        # greedy
        policy = Policy(self.env, eps=1)
        policy.action_stars = np.array([[0, 1], [2, 3]])
        actions, probs = policy.get_action_probs((0, 0))
        assert is_same_array(actions, [0, 1, 2, 3, 4], True)
        for p in probs:
            if not np.isclose(p, 0.2):
                assert False

    def test_get_action(self):
        # non-greedy
        policy = Policy(self.env, eps=0)
        policy.action_stars = np.array([[0, 1], [2, 3]])
        assert 0 == policy.get_action((0, 0))

    def test_improve(self):
        policy = Policy(self.env, eps=0)
        policy.action_stars = np.array([[0, 1], [2, 3]])
        q_values = np.array([0, 1, 0, 0, 0])
        policy.improve((0, 0), q_values)
        assert policy.action_stars[0, 0] == 1


def test_get_return():
    env = gym.make(
        "gridworld-v0",
        size=2,
        target_loc=(1, 1),
        forbidden_area_cfg=ForbiddenAreaCfg(locs=[(0, 1)]),
        reward_cfg=RewardCfg(move=0, target=1, forbidden_area=-2, out_of_bound=-1),
    )
    episode_length = 10
    action_stars = np.array([[2, 2], [1, 4]])
    
    policy = Policy(env, eps=0)
    policy.action_stars = action_stars
    q_value = get_return((0, 0), 0, env, policy, episode_length, gamma=0.9)
    assert np.isclose(q_value, 3.61, atol=1e-2)
    
    q_value = get_return((0, 0), 1, env, policy, episode_length, gamma=0.9)
    assert np.isclose(q_value, 3.51, atol=1e-2)
    
    q_value = get_return((0, 0), 2, env, policy, episode_length, gamma=0.9)
    assert np.isclose(q_value, 5.51, atol=1e-2)
    
    q_value = get_return((0, 0), 3, env, policy, episode_length, gamma=0.9)
    assert np.isclose(q_value, 3.61, atol=1e-2)
    
    q_value = get_return((0, 0), 4, env, policy, episode_length, gamma=0.9)
    assert np.isclose(q_value, 4.61, atol=1e-2)
