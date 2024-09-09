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


def get_return(
    loc: tuple[int, int],
    action: int,
    env: gym.Env,
    policy: Policy,
    episode_length: int = 100,
    gamma: float = 0.9,
) -> float:
    size: tuple[int, int] = env.unwrapped.size  # type: ignore
    if not (0 <= loc[0] < size[0] and 0 <= loc[1] < size[1]):
        raise ValueError(f"loc should be in [0, {size[0]}) x [0, {size[1]})")

    cur_loc = np.array(loc)
    cur_action = action
    return_val = 0.0
    tranjctory_length = 0

    env.reset(options={"agent_loc": loc})
    for _ in range(episode_length):
        obs, rwd_, *_ = env.step(cur_action)

        rwd = float(rwd_)
        return_val += rwd * gamma**tranjctory_length
        tranjctory_length += 1

        cur_loc = obs["agent_loc"]
        cur_action = policy.get_action(cur_loc)

    return return_val


class Visualizer:
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap

    # white for normal area, yellow for forbidden area, blue for target area
    cmap = ListedColormap(["#F8F1E5", "#FFECB3", "#6FA3EF"])  # white, yellow, blue
    arrow_directions = {
        0: (0.0, 1.0),  # up
        1: (1.0, 0.0),  # right
        2: (0.0, -1.0),  # down
        3: (-1.0, 0.0),  # left
    }

    def __init__(self, env: gym.Env):
        self.env = env
        # tuple type for indexing and sizing
        self.size: tuple[int, int] = env.unwrapped.size  # type: ignore
        self.target_loc = tuple(env.unwrapped._target_loc)  # type: ignore
        self.forbidden_locs: np.ndarray = env.unwrapped._forbidden_locs  # type: ignore
        self.start_loc = tuple(env.unwrapped._start_loc)  # type: ignore

    def _draw_mesh(self, ax: plt.Axes):
        num_row, num_col = self.size
        x, y = np.linspace(0, num_row, num_row + 1), np.linspace(
            0, num_col, num_col + 1
        )
        X, Y = np.meshgrid(x, y)

        Z = np.zeros(self.size, dtype=np.int8)

        Z[self.target_loc] = 2
        for loc in self.forbidden_locs:
            idx = tuple(loc)  # tuple type for indexing
            Z[idx] = 1
        Z = np.flipud(Z)  # (0, 0) is at the upper left; (w, h) is at the bottom right

        # draw mesh
        ax.pcolormesh(X, Y, Z, cmap=self.cmap, shading="flat", edgecolors="face")
        # crop the graph
        ax.set_xlim(x.min(), x.max())
        ax.set_ylim(y.min(), y.max())

        # draw grid line
        ax.grid(True, which="both", color="black", linestyle="-", linewidth=2)
        ax.tick_params(labelbottom=False, labelleft=False)

    def _draw_arrow(
        self, ax: plt.Axes, loc: tuple[int, int], actions: list[int], probs: list[float]
    ):
        from matplotlib import patches

        # (0, 0) is at the upper left; (w, h) is at the bottom right
        x_center, y_center = loc[1] + 0.5, self.size[0] - loc[0] - 0.5

        for a, p in zip(actions[:4], probs[:4]):
            dx, dy = self.arrow_directions[a]
            arrow_len = 0.2 * p
            dx, dy = dx * arrow_len, dy * arrow_len
            ax.arrow(
                x_center,
                y_center,
                dx,
                dy,
                head_width=0.2 * arrow_len,
                head_length=0.4 * arrow_len,
                fc="darkgreen",
                ec="darkgreen",
            )

        # action 5: stay
        radius = 0.2 * probs[4]
        circle = patches.Circle(
            (x_center, y_center), radius, color="darkgreen", fill=False, lw=2
        )
        ax.add_patch(circle)

    def draw_strategy(self, policy: Policy):
        """Draw the strategy of grid world. Used for IPython Note Book.

        Args:
            policy (Policy): Policy for GridWorldEnv
        """
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()

        self._draw_mesh(ax)
        # draw arrow with probability
        for i in range(self.size[0]):
            for j in range(self.size[1]):
                loc = (i, j)
                actions, probs = policy.get_action_probs(loc)
                self._draw_arrow(ax, loc, actions, probs)
        ax.set_aspect("equal", adjustable="box")

        fig.tight_layout()
        plt.show()

    def draw_state_values(
        self, policy: Policy, episode_length: int, num_episodes: int, gamma: float = 0.9
    ):
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()

        self._draw_mesh(ax)
        # draw state value
        for i in range(self.size[0]):
            for j in range(self.size[1]):
                loc = (i, j)
                actions, probs = policy.get_action_probs(loc)
                q_values = np.zeros(5, dtype=np.float32)
                for a in actions:
                    q = [
                        get_return(loc, a, self.env, policy, episode_length, gamma)
                        for _ in range(num_episodes)
                    ]
                    q_values[a] = np.mean(q, dtype=np.float32)
                v: np.ndarray = (q_values * probs).sum(dtype=np.float32)
                x_center, y_center = j + 0.5, self.size[0] - i - 0.5
                ax.text(
                    x_center,
                    y_center,
                    f"{v:.2f}",
                    ha="center",
                    va="center",
                    fontsize=16,
                )
        ax.set_aspect("equal", adjustable="box")

        fig.tight_layout()
        plt.show()
