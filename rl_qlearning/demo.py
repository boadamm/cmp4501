from .env import GridWorld
from .qlearn import greedy_policy, q_learning

if __name__ == "__main__":
    env = GridWorld()
    Q = q_learning(env, episodes=3000)
    policy = greedy_policy(Q)
    arrows = {0: "↑", 1: "→", 2: "↓", 3: "←"}  # N, E, S, W
    # Correcting action mapping if ACTION_SPACE in env.py is {0:N, 1:E, 2:S, 3:W}
    # The policy will output 0, 1, 2, 3. These need to map to the arrows.
    # If ACTION_SPACE is {0: (-1,0) #N, 1: (0,1) #E, 2: (1,0) #S, 3: (0,-1) #W}
    # And policy gives action index, then arrows map is correct.

    print("Learned Q-table:")
    print(Q)
    print("\nDerived Policy (Action ID per state):")
    print(policy)
    print("\nPolicy Visualization (Arrows):")
    for r in range(env.size):
        row_actions = []
        for c in range(env.size):
            state_id = env._state_id((r, c))
            action_id = policy[state_id]
            row_actions.append(arrows.get(action_id, "?"))  # Use .get for safety
        print(" ".join(row_actions))

    # Simulate a run with the learned policy
    print("\nSimulating agent with learned policy:")
    current_state = env.reset()
    done = False
    total_reward_demo = 0
    path_taken = []
    action_names = {0: "N", 1: "E", 2: "S", 3: "W"}

    for i in range(env.size * env.size * 2):  # Max steps to prevent infinite loop
        action = policy[current_state]
        path_taken.append((env.state, action_names.get(action, "ERR")))
        next_state, reward, done = env.step(action)
        total_reward_demo += reward
        print(
            f"Step {i + 1}: State {env.state} (ID: {current_state}), "
            f"Action: {action_names.get(action, 'ERR')}, Reward: {reward}, "
            f"Next State ID: {next_state}, Done: {done}"
        )
        current_state = next_state
        if done:
            path_taken.append((env.state, "GOAL"))
            print(f"Goal reached! Total reward: {total_reward_demo}")
            break
    if not done:
        print("Agent did not reach goal within max steps.")

    print("\nPath visualization:")
    grid_viz = [["." for _ in range(env.size)] for _ in range(env.size)]
    for i_viz, ((r_viz, c_viz), a_viz) in enumerate(path_taken):
        if grid_viz[r_viz][c_viz] == ".":
            grid_viz[r_viz][c_viz] = str(i_viz) if a_viz != "GOAL" else "G"
    for row_viz in grid_viz:
        print(" ".join(row_viz))
