import gymnasium as gym

import numpy as np


class Policy:
    version = 0  # increment this when exectue Policy Improvement

    def __init__(self, env: gym.Env, eps: float = 0):
        """Initialize policy for grid world

        Args:
            env (gym.Env): GridWorldEnv
            eps (float, optional): Epsilion for epsilion-greed policy. Set it to 0 for non-greedy. Defaults to 0.

        Raises:
            ValueError: eps should be in [0, 1]
        """

        size = env.unwrapped.size  # type: ignore
        self.size = size if isinstance(size, tuple) else (size, size)
        # record the action index whose action value is maximum
        self.action_stars = np.random.randint(
            0, 4, size=size
        )  # 0: up, 1: right, 2: down, 3: left, 4: stay

        if eps < 0 or eps > 1:
            raise ValueError("eps should be in [0, 1], 0 for non-greedy")

        self.eps = eps

    @staticmethod
    def _check_loc(loc: tuple[int, int] | np.ndarray):
        if isinstance(loc, np.ndarray):
            if loc.shape != (2,):
                raise ValueError(f"loc shape should be (2,), but got {loc.shape}")
            if not np.issubdtype(loc.dtype, np.integer):
                raise ValueError(f"loc dtype should be np.integer, but got {loc.dtype}")

    def get_action_probs(
        self, loc: tuple[int, int] | np.ndarray
    ) -> tuple[list[int], list[float]]:
        self._check_loc(loc)

        num_act = 5
        fact = 1 / num_act * self.eps
        prob_star = 1 - (num_act - 1) * fact
        prob = 1 * fact

        actions = [i for i in range(num_act)]
        probs = []

        idx = tuple(loc)
        action_star = self.action_stars[idx]
        for a in actions:
            if (a == action_star).all():
                probs.append(prob_star)
            else:
                probs.append(prob)
        return actions, probs

    def get_action(self, loc: tuple[int, int] | np.ndarray) -> int:
        self._check_loc(loc)

        actions, probs = self.get_action_probs(loc)

        return np.random.choice(actions, 1, p=probs).item()

    def improve(self, loc: tuple[int, int] | np.ndarray, q_values: np.ndarray):
        self._check_loc(loc)

        if q_values.shape != (5,):
            raise ValueError(
                f"q_values shape should be {(5, )}, but got {q_values.shape}"
            )
        idx = tuple(loc)
        self.version += 1
        self.action_stars[idx] = np.argmax(q_values)
