# pragma: no cover
from naive_bayes.datasets import load_toy_data
from naive_bayes.model import MultinomialNB

if __name__ == "__main__":
    X_tr, y_tr, X_te, y_te = load_toy_data()
    clf = MultinomialNB().fit(X_tr, y_tr)
    acc = (clf.predict(X_te) == y_te).mean()
    print(f"accuracy: {acc}")  # Added f-string for clarity
