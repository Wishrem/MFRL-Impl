import numpy as np

from envs import make_gridworld
from tests.utils import is_same_array
from common.policy import Policy


class TestPolicy:
    env = make_gridworld(size=5)

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
