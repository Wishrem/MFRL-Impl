import numpy as np
import gymnasium as gym

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from .policy import Policy
from .utils import get_return

class Visualizer:

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
