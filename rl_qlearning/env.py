from typing import Tuple

ACTION_SPACE = {
    0: (-1, 0),  # N
    1: (0, 1),  # E
    2: (1, 0),  # S
    3: (0, -1),
}  # W


class GridWorld:
    """Deterministic 4x4 GridWorld with step cost -1 and goal +10."""

    def __init__(self, size: int = 4, goal: Tuple[int, int] = (3, 3)) -> None:
        """Initialize the GridWorld environment.

        Args:
            size (int, optional): The size of the grid (size x size). Defaults to 4.
            goal (Tuple[int, int], optional): The (row, col) coordinates of the goal
                                              state. Defaults to (3, 3).
        """
        self.size = size
        self.goal = goal
        self.state = (0, 0)

    def reset(self) -> int:
        self.state = (0, 0)
        return self._state_id(self.state)

    def step(self, action: int) -> Tuple[int, int, bool]:
        """Take an action in the environment.

        The agent moves one step in the specified direction. If the move
        goes off the grid, the agent stays in its current position.

        Args:
            action (int): The action to take, corresponding to an index in ACTION_SPACE
                          (0: North, 1: East, 2: South, 3: West).

        Returns:
            Tuple[int, int, bool]: A tuple containing:
                - next_state_id (int): The ID of the state after taking the action.
                - reward (int): The reward received for taking the action (-1 for a
                                normal step, +10 if the goal is reached).
                - done (bool): True if the goal state is reached, False otherwise.
        """
        dr, dc = ACTION_SPACE[action]
        r, c = self.state
        nr, nc = max(0, min(self.size - 1, r + dr)), max(0, min(self.size - 1, c + dc))
        self.state = (nr, nc)
        done = self.state == self.goal
        reward = 10 if done else -1
        return self._state_id(self.state), reward, done

    # helpers
    def _state_id(self, rc: Tuple[int, int]) -> int:
        """Convert (row, col) coordinates to a unique state ID."""
        return rc[0] * self.size + rc[1]
