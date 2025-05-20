from typing import Tuple

import numpy as np


def load_toy_split(seed: int = 123) -> Tuple[np.ndarray, ...]:
    """
    Binary-class 2-feature toy dataset that is perfectly separable
    (for deterministic unit tests).
    Returns X_train, y_train, X_test, y_test.
    """
    rng = np.random.default_rng(seed)
    X0 = rng.normal(loc=(-2, -2), scale=0.5, size=(25, 2))
    X1 = rng.normal(loc=(2, 2), scale=0.5, size=(25, 2))
    X = np.vstack([X0, X1])
    y = np.array([0] * 25 + [1] * 25)
    idx = rng.permutation(len(y))
    X, y = X[idx], y[idx]
    split = 40
    return X[:split], y[:split], X[split:], y[split:]
