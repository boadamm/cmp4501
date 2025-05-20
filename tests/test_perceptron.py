from sklearn.model_selection import train_test_split

from perceptron.datasets import load_linear, xor_dataset
from perceptron.model import Perceptron

MIN_ACCURACY_THRESHOLD = 0.9


def test_perceptron_accuracy():
    X_tr, y_tr, X_te, y_te = load_linear()
    clf = Perceptron(hidden_size=6, lr=0.1, epochs=2000).fit(X_tr, y_tr)
    y_pred = clf.predict(X_te)
    acc = (y_pred == y_te).mean()
    assert acc >= MIN_ACCURACY_THRESHOLD
    assert y_pred.shape == y_te.shape


def test_perceptron_xor():
    X, y = xor_dataset()
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=42)
    clf = Perceptron(hidden_size=8, lr=0.1, epochs=5000, seed=1).fit(X_tr, y_tr)
    y_pred = clf.predict(X_te)
    acc = (y_pred == y_te).mean()
    assert acc >= MIN_ACCURACY_THRESHOLD
    assert y_pred.shape == y_te.shape
