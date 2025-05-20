from decision_tree.datasets import load_digits_binary
from decision_tree.tree import DecisionTreeClassifier

MIN_ACCURACY_TREE = 0.9

def test_dt_accuracy():
    X_tr, y_tr, X_te, y_te = load_digits_binary()
    clf = DecisionTreeClassifier(max_depth=10, min_samples_leaf=5).fit(X_tr, y_tr)
    y_pred = clf.predict(X_te)
    acc = (y_pred == y_te).mean()
    assert acc >= MIN_ACCURACY_TREE
    assert y_pred.shape == y_te.shape
