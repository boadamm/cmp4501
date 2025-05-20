from decision_tree.datasets import load_toy_split
from decision_tree.tree import DecisionTree


def test_tree_accuracy():
    X_tr, y_tr, X_te, y_te = load_toy_split()
    clf = DecisionTree(max_depth=3).fit(X_tr, y_tr)
    y_pred = clf.predict(X_te)
    acc = (y_pred == y_te).mean()
    assert acc >= 0.9
    assert y_pred.shape == y_te.shape
