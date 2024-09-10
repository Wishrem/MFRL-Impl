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

    def _sample_q_values(self, episode_length: int, policy: Policy, gamma: float):
        q_values = np.empty((*self.size, 5), dtype=np.float32)
        for i in range(self.size[0]):
            for j in range(self.size[1]):
                loc = (i, j)
                actions, probs = policy.get_action_probs(loc)
                for a in actions:
                    q = get_return(loc, a, self.env, policy, episode_length, gamma)
                    q_values[loc][a] = q
        return q_values

    def draw_state_values(
        self,
        policy: Policy,
        q_values: np.ndarray | None = None,
        episode_length: int | None = None,
        num_episodes: int | None = None,
        gamma: float = 0.9,
    ):
        """Draw the state values of grid world. Used for IPython Note Book.

        q_values or episode_length and num_episodes must be provided.
        q_values has a higher priority.

        Args:
            policy (Policy): Policy for GridWorldEnv
            q_values (np.ndarray | None, optional): The q values for all state-action pairs.. Defaults to None.
            episode_length (int | None, optional): The length of episode used for sampling a q value. Defaults to None.
            num_episodes (int | None, optional): The number of episodes used for sampling a q value. Defaults to None.
            gamma (float, optional): The discount rate. Defaults to 0.9.

        Raises:
            ValueError: If q_values is None, episode_length and num_episodes must be provided. If q_values is provided, it must have the shape (*self.size, 5).
        """
        if q_values is None:
            if episode_length is None or num_episodes is None:
                raise ValueError(
                    "episode_length and num_episodes must be provided, if q_values is None."
                )
            q_values = self._sample_q_values(episode_length, policy, gamma)
        elif q_values.shape != (*self.size, 5):
            raise ValueError(
                f"q_values must have the shape {(*self.size, 5)}, but got {q_values.shape}."
            )
        elif not np.issubdtype(q_values.dtype, np.floating):
            print("Warning: q_values is not float type. Convert it to float")
            q_values = q_values.astype(np.float32)

        if q_values is not None and (
            episode_length is not None or num_episodes is not None
        ):
            print(
                "Warning: q_values is provided. episode_length and num_episodes are ignored."
            )

        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()

        self._draw_mesh(ax)
        # draw state values
        for i in range(self.size[0]):
            for j in range(self.size[1]):
                loc = (i, j)
                _, probs = policy.get_action_probs(loc)
                q = q_values[loc]
                v: np.ndarray = (q * probs).sum(dtype=np.float32)
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
