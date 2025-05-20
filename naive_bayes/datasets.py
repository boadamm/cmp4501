from typing import Tuple

import numpy as np


def load_toy_data(seed: int = 42) -> Tuple[np.ndarray, ...]:
    """
    Generate an easy binary bag-of-words dataset:
    • Features 0-19 are 'spam keywords'  → high rate for spam, low for ham
    • Features 20-39 are 'ham keywords'  → high rate for ham, low for spam
    • Features 40-99 are noise           → same low rate in both classes
    Returns X_train, y_train, X_test, y_test.
    """
    rng = np.random.default_rng(seed)
    n_feat = 100
    n_each = 50

    # base rates
    X_spam = rng.poisson(0.3, size=(n_each, n_feat))
    X_ham = rng.poisson(0.3, size=(n_each, n_feat))

    # bump class-specific keyword counts
    X_spam[:, :20] += rng.poisson(2.0, size=(n_each, 20))  # spam words
    X_ham[:, 20:40] += rng.poisson(2.0, size=(n_each, 20))  # ham words

    X = np.vstack([X_spam, X_ham])
    y = np.array([1] * n_each + [0] * n_each)

    # shuffle & split 70/30
    idx = rng.permutation(len(y))
    X, y = X[idx], y[idx]
    split = int(0.7 * len(y))
    return X[:split], y[:split], X[split:], y[split:]
