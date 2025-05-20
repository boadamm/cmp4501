import numpy as np
import pytest

from rl_qlearning.env import ACTION_SPACE, GridWorld
from rl_qlearning.qlearn import greedy_policy, q_learning


@pytest.fixture
def env() -> GridWorld:
    """Provides a GridWorld environment instance."""
    return GridWorld()


@pytest.fixture
def q_table_learned(env: GridWorld) -> np.ndarray:
    """Provides a Q-table learned with a sufficient number of episodes for tests to pass."""
    # Episodes might need adjustment depending on the Q-learning implementation's efficiency
    return q_learning(env, episodes=5000, alpha=0.1, epsilon=0.1)


def test_q_table_shape(q_table_learned: np.ndarray, env: GridWorld):
    """Tests if the Q-table has the correct shape (e.g., 16x4 for a 4x4 grid)."""
    expected_shape = (env.size * env.size, len(ACTION_SPACE))
    assert q_table_learned.shape == expected_shape, (
        f"Q-table shape is {q_table_learned.shape}, expected {expected_shape}"
    )


def test_greedy_policy_reaches_goal(q_table_learned: np.ndarray, env: GridWorld):
    """
    Tests if the greedy policy derived from the Q-table can reach the goal
    from state (0,0) within a maximum number of steps on average over multiple trials.
    """
    policy = greedy_policy(q_table_learned)
    num_trials = 100
    max_steps_to_goal = (
        10  # Max steps allowed to reach the goal, as per original prompt
    )
    total_steps_taken = 0
    successful_reaches = 0

    for _ in range(num_trials):
        state_id = env.reset()  # env.reset() returns the state_id
        current_state_tuple = env.state  # Keep track of tuple for goal check
        steps_this_trial = 0
        done = False
        # print(f"Trial {_ + 1}: Start State {current_state_tuple} (ID: {state_id})")
        while (
            not done and steps_this_trial < max_steps_to_goal * 3
        ):  # Increased safety break slightly
            action = policy[state_id]
            # print(f"  Step {steps_this_trial + 1}: State {current_state_tuple} (ID: {state_id}), Action: {action}")
            next_state_id, reward, done = env.step(action)
            current_state_tuple = env.state  # Update tuple state after step
            state_id = next_state_id
            steps_this_trial += 1

            if done and current_state_tuple == env.goal:
                total_steps_taken += steps_this_trial
                successful_reaches += 1
                # print(f"  Goal reached in {steps_this_trial} steps.")
                break
            elif done:  # Reached terminal state but not THE goal (should not happen with this env)
                # print(f"  Terminal state {current_state_tuple} reached but not goal after {steps_this_trial} steps.")
                break
        # if steps_this_trial >= max_steps_to_goal * 3:
        # print(f"  Trial timed out at state {current_state_tuple}")

    # The policy should be good enough now.
    # We expect a high success rate and average steps within the limit.
    min_successful_trials = num_trials * 0.9  # Expect at least 90% success rate

    print(f"Successful reaches: {successful_reaches}/{num_trials}")
    assert successful_reaches >= min_successful_trials, (
        f"Greedy policy only reached goal in {successful_reaches}/{num_trials} trials. Expected at least {min_successful_trials}."
    )

    if successful_reaches > 0:
        average_steps = total_steps_taken / successful_reaches
        print(f"Average steps to goal: {average_steps:.2f}")
        assert average_steps <= max_steps_to_goal, (
            f"Greedy policy reached goal in {average_steps:.2f} steps on average, expected <= {max_steps_to_goal}."
        )
    # If successful_reaches is 0 (and less than min_successful_trials), the first assert would have failed.
