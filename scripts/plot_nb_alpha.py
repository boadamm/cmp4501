# pragma: no cover
import matplotlib.pyplot as plt
import numpy as np

from naive_bayes.datasets import load_toy_data
from naive_bayes.model import MultinomialNB

X_tr, y_tr, X_te, y_te = load_toy_data()
alphas = np.logspace(-2, 1, 9)
acc = []
for a in alphas:
    clf = MultinomialNB(alpha=a).fit(X_tr, y_tr)
    acc.append((clf.predict(X_te) == y_te).mean())

plt.semilogx(alphas, acc, marker="o")
plt.xlabel("Laplace alpha")
plt.ylabel("Accuracy")
plt.title("Naïve Bayes smoothing curve")
plt.savefig("figures/nb_alpha_curve.png", dpi=200)
