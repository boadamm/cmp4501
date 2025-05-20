# pragma: no cover
import matplotlib.pyplot as plt

from rl_qlearning.env import GridWorld
from rl_qlearning.qlearn import q_learning

env = GridWorld()
_, returns = q_learning(
    env, episodes=3000, epsilon=0.2, alpha=0.1, gamma=0.9, log_returns=True
)  # Unpack Q and returns
plt.plot(returns)
plt.axhline(0, linestyle="--")
plt.xlabel("Episode")
plt.ylabel("Cumulative reward")
plt.title("Q-learning convergence")
plt.savefig("figures/qlearning_returns.png", dpi=200)
