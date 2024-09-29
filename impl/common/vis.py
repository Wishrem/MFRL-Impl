import numpy as np
import gymnasium as gym

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from envs.grid_world import GridWorldEnv
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
    star_coords = np.array(
        [
            (0, 1),
            (0.2245, 0.309),
            (0.9511, 0.309),
            (0.3633, -0.118),
            (0.5878, -0.809),
            (0, -0.382),
            (-0.5878, -0.809),
            (-0.3633, -0.118),
            (-0.9511, 0.309),
            (-0.2245, 0.309),
            (0, 1),
        ],
        dtype=np.float16,
    )

    def __init__(self, env: gym.Env):
        self.env = env
        # tuple type for indexing and sizing
        grid_world: GridWorldEnv = env.unwrapped  # type: ignore
        self.size = grid_world.size
        self.target_loc: tuple[int, int] = tuple(grid_world.target_loc)
        self.forbidden_locs = grid_world._forbidden_locs
        self.start_loc: tuple[int, int] = tuple(grid_world.start_loc)
        self.action_to_direction = grid_world._action_to_direction

        self._check_grid_world_properties()

    def _check_grid_world_properties(self):
        if len(self.size) != 2:
            raise ValueError("size must have 2 elements")
        if len(self.target_loc) != 2:
            raise ValueError("target_loc must have 2 elements")
        if len(self.start_loc) != 2:
            raise ValueError("start_loc must have 2 elements")

    def _draw_mesh(self, ax: plt.Axes):
        num_row, num_col = self.size
        x, y = (
            np.linspace(0, num_row, num_row + 1),
            np.linspace(0, num_col, num_col + 1),
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

    def _draw_start_loc(self, loc: tuple[int, int], ax: plt.Axes):
        from matplotlib import patches

        x_center, y_center = loc[1] + 0.5, self.size[0] - loc[0] - 0.5

        scale = 0.2
        scaled_coords = [
            (x * scale + x_center, y * scale + y_center) for x, y in self.star_coords
        ]
        polygon = patches.Polygon(scaled_coords, closed=True, color="orange")
        ax.add_patch(polygon)

    def draw_strategy(self, policy: Policy, with_trajectory: bool = False):
        """Draw the strategy of grid world. Used for IPython Note Book.

        Args:
            policy (Policy): Policy for GridWorldEnv
        """
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()

        self._draw_mesh(ax)

        if with_trajectory:
            trajectory: list[tuple[int, int]] = [
                self.start_loc
            ]  # convenient for drawing the trajectory
            nxt_loc = self.start_loc
            self._draw_start_loc(self.start_loc, ax)
            while not self.reach_end(trajectory[1:], nxt_loc):
                cur_loc = nxt_loc
                last_loc = trajectory[-1]
                trajectory.append(cur_loc)

                x1, x2 = cur_loc[1] + 0.5, last_loc[1] + 0.5
                y1, y2 = (
                    self.size[0] - cur_loc[0] - 0.5,
                    self.size[0] - last_loc[0] - 0.5,
                )
                ax.plot([x1, x2], [y1, y2], "-", color="blue", lw=0.5)
                action = policy.get_action(cur_loc, True)
                nxt_loc = tuple(
                    np.add(nxt_loc, self.action_to_direction[action]).astype(int)
                )

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
                for a in range(5):
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
        if q_values is not None and (
            episode_length is not None or num_episodes is not None
        ):
            print(
                "Warning: q_values is provided. episode_length and num_episodes are ignored."
            )

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

    def reach_end(
        self, trajectory: list[tuple[int, int]], nxt_loc: tuple[int, int]
    ) -> bool:
        """Check if the trajectory reaches the path end, including forming a circle or getting out of the grid world.
        Args:
            trajectory (list[np.ndarray]): The trajectory of the agent.
            action (int): The action to be taken on a specific state.
        """

        if len(trajectory) == 0:
            return False

        cur_loc = trajectory[-1]
        if cur_loc == self.target_loc:
            return True

        if nxt_loc in trajectory:
            return True

        if (
            nxt_loc[0] < 0
            or nxt_loc[0] >= self.size[0]
            or nxt_loc[1] < 0
            or nxt_loc[1] >= self.size[1]
        ):
            return True

        return False
