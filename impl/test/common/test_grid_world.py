import numpy as np
import envs # for registering the environment
import gymnasium as gym

from common.grid_world import Policy


class TestPolicy:
    env = gym.make("gridworld-v0", size=5)
    
    def test_init(self):
        policy = Policy(self.env, eps=0.1)
        assert policy.size == (5, 5)
        assert policy.action_stars.shape == (5, 5)

        
    def test_get_action(self):
        # non-greedy
        policy = Policy(self.env, eps=0)
        policy.action_stars = np.array([[0, 1], [2, 3]])
        assert 0 == policy.get_action((0, 0))
        
        # greedy
        policy = Policy(self.env, eps=1)
        policy.action_stars = np.array([[0, 1], [2, 3]])
        n = 1000
        cnt = 0
        for _ in range(n):
            if policy.get_action((0, 0)) == 0:
                cnt += 1
        p = cnt / n
        assert (p - 1/5) < 0.0001

    def test_improve(self):
        policy = Policy(self.env, eps=0)
        policy.action_stars = np.array([[0, 1], [2, 3]])
        q_values = np.array([0, 1, 0, 0, 0])
        policy.improve((0, 0), q_values)
        assert policy.action_stars[0, 0] == 1