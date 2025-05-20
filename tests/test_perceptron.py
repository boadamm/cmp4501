from perceptron.datasets import load_linear
from perceptron.model import Perceptron


def test_perceptron_accuracy():
    X_tr, y_tr, X_te, y_te = load_linear()
    clf = Perceptron(hidden_size=6, lr=0.1, epochs=2000).fit(X_tr, y_tr)
    y_pred = clf.predict(X_te)
    acc = (y_pred == y_te).mean()
    assert acc >= 0.9
    assert y_pred.shape == y_te.shape
