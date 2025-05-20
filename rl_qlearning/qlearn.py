from typing import List

import numpy as np

from .env import ACTION_SPACE, GridWorld


def q_learning(
    env: GridWorld,
    alpha: float = 0.1,
    gamma: float = 0.9,
    epsilon: float = 0.1,
    episodes: int = 5000,
) -> np.ndarray:
    """Perform tabular Q-learning to find the optimal Q-function.

    Args:
        env (GridWorld): The environment to learn from.
        alpha (float, optional): Learning rate. Defaults to 0.1.
        gamma (float, optional): Discount factor. Defaults to 0.9.
        epsilon (float, optional): Exploration rate for epsilon-greedy policy.
                                 Defaults to 0.1.
        episodes (int, optional): Number of episodes to train for. Defaults to 5000.

    Returns:
        np.ndarray: The learned Q-table, a 2D array of shape (n_states, n_actions).
    """
    n_states, n_actions = env.size**2, len(ACTION_SPACE)
    Q = np.zeros((n_states, n_actions))

    rng = np.random.default_rng()
    for _ in range(episodes):
        s = env.reset()
        done = False
        while not done:
            # ε-greedy
            if rng.random() < epsilon:
                a = rng.integers(n_actions)
            else:
                a = int(np.argmax(Q[s]))
            s_next, r, done = env.step(a)
            Q[s, a] += alpha * (r + gamma * np.max(Q[s_next]) - Q[s, a])
            s = s_next
    return Q


def greedy_policy(Q: np.ndarray) -> List[int]:
    """Extract the greedy policy from a Q-table.

    For each state, this function selects the action with the highest Q-value.

    Args:
        Q (np.ndarray): The Q-table, a 2D array where Q[s, a] is the Q-value
                        of taking action a in state s.

    Returns:
        List[int]: A list where the i-th element is the greedy action to take
                   in state i.
    """
    return list(np.argmax(Q, axis=1))
