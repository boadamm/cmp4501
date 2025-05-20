from typing import Tuple

import numpy as np


def load_linear(seed: int = 24) -> Tuple[np.ndarray, ...]:
    """
    Linearly separable 2-D dataset (100 samples, 2 classes).
    Returns X_train, y_train, X_test, y_test.
    """
    rng = np.random.default_rng(seed)
    X0 = rng.normal(loc=(-2, -2), scale=0.4, size=(50, 2))
    X1 = rng.normal(loc=(2, 2), scale=0.4, size=(50, 2))
    X = np.vstack([X0, X1])
    y = np.array([0] * 50 + [1] * 50)
    idx = rng.permutation(len(y))
    X, y = X[idx], y[idx]
    split = 80
    return X[:split], y[:split], X[split:], y[split:]
