# pragma: no cover
import matplotlib.pyplot as plt

from decision_tree.datasets import load_toy_split
from decision_tree.tree import DecisionTree

X_tr, y_tr, X_te, y_te = load_toy_split()
depths, acc = [], []
for d in range(1, 7):
    clf = DecisionTree(max_depth=d).fit(X_tr, y_tr)
    acc.append((clf.predict(X_te) == y_te).mean())
    depths.append(d)

plt.plot(depths, acc, marker="o")
plt.xlabel("Max depth")
plt.ylabel("Accuracy")
plt.title("Decision-Tree depth study")
plt.savefig("figures/tree_depth.png", dpi=200)
