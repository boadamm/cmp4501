from typing import Optional, Tuple

import numpy as np


class Node:
    def __init__(self,
                 feature: Optional[int] = None,
                 threshold: Optional[float] = None,
                 left: Optional["Node"] = None,
                 right: Optional["Node"] = None,
                 *, label: Optional[int] = None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.label = label

    def is_leaf(self) -> bool:
        return self.label is not None

class DecisionTree:
    """
    Simple CART-style binary decision tree (no pruning yet).
    """
    def __init__(self, max_depth: int = 3, min_samples_split: int = 2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.root: Optional[Node] = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "DecisionTree":
        self.root = self._grow(X, y, depth=0)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.array([self._predict_row(row, self.root) for row in X])

    def _gini(self, y: np.ndarray) -> float:
        if y.size == 0:
            return 0.0
        p1 = (y == 1).mean()
        return 1.0 - (p1**2 + (1 - p1)**2)

    def _best_split(self, X: np.ndarray, y: np.ndarray) -> Tuple[int, float, float]:
        best_gain, best_feat, best_thresh = -1, -1, -1.0
        parent_gini = self._gini(y)
        n_samples = y.size
        for feat in range(X.shape[1]):
            thresholds = np.unique(X[:, feat])
            for t in thresholds:
                left_idx = X[:, feat] <= t
                right_idx = ~left_idx
                if (left_idx.sum() < self.min_samples_split or
                        right_idx.sum() < self.min_samples_split):
                    continue
                g_left = self._gini(y[left_idx])
                g_right = self._gini(y[right_idx])
                gain = parent_gini - (left_idx.sum() / n_samples) * g_left - \
                       (right_idx.sum() / n_samples) * g_right
                if gain > best_gain:
                    best_gain, best_feat, best_thresh = gain, feat, t
        return best_feat, best_thresh, best_gain

    def _grow(self, X: np.ndarray, y: np.ndarray, depth: int) -> Node:
        # make leaf if pure or max depth reached
        if (depth >= self.max_depth or
                self._gini(y) == 0.0 or
                y.size < self.min_samples_split):
            label = int(np.round(y.mean()))
            return Node(label=label)

        feat, thresh, gain = self._best_split(X, y)
        if gain <= 0: # If no best split found or gain is not positive, make a leaf
            label = int(np.round(y.mean()))
            return Node(label=label)

        left_mask = X[:, feat] <= thresh
        # Add a check to ensure that the split results in at least one sample
        # in each child node
        if not np.any(left_mask) or not np.any(~left_mask):
            label = int(np.round(y.mean()))
            return Node(label=label)

        left = self._grow(X[left_mask], y[left_mask], depth+1)
        right = self._grow(X[~left_mask], y[~left_mask], depth+1)
        return Node(feature=feat, threshold=thresh, left=left, right=right)

    def _predict_row(self, x: np.ndarray, node: Node) -> int:
        while not node.is_leaf():
            node = node.left if x[node.feature] <= node.threshold else node.right
        return node.label

    def export_text(self, node: Optional[Node]=None, depth: int = 0) -> str:
        if node is None:
            node = self.root
        indent = "  " * depth
        if node.is_leaf():
            return f"{indent}Predict {node.label}\n"
        txt = f"{indent}X[{node.feature}] <= {node.threshold:.2f}?\n"
        txt += self.export_text(node.left, depth+1)
        txt += self.export_text(node.right, depth+1)
        return txt
