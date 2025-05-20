from naive_bayes.datasets import load_toy_data
from naive_bayes.model import MultinomialNB


def test_nb_accuracy():
    X_tr, y_tr, X_te, y_te = load_toy_data()
    clf = MultinomialNB(alpha=1.0).fit(X_tr, y_tr)
    y_pred = clf.predict(X_te)
    assert (y_pred == y_te).mean() >= 0.90
    assert y_pred.shape == y_te.shape
