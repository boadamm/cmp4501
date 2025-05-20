from typing import Optional

import numpy as np


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-z))


class Perceptron:
    """
    2-layer feed-forward network:
    input (d) → hidden (h) → sigmoid → output sigmoid (binary).
    """

    PREDICTION_THRESHOLD = 0.5

    def __init__(
        self, hidden_size: int = 8, lr: float = 0.1, epochs: int = 5000, seed: int = 1
    ):
        self.hidden_size = hidden_size
        self.lr = lr
        self.epochs = epochs
        rng = np.random.default_rng(seed)
        self.rng = rng
        # weights will be initialised in fit when input dim is known
        self.W1: Optional[np.ndarray] = None
        self.b1: Optional[np.ndarray] = None
        self.W2: Optional[np.ndarray] = None
        self.b2: Optional[float] = None

    # ── public ────────────────────────────────────────────
    def fit(self, X: np.ndarray, y: np.ndarray) -> "Perceptron":
        n_samples, n_feat = X.shape
        # Xavier init
        limit1 = np.sqrt(6 / (n_feat + self.hidden_size))
        self.W1 = self.rng.uniform(-limit1, limit1, (n_feat, self.hidden_size))
        self.b1 = np.zeros(self.hidden_size)

        limit2 = np.sqrt(6 / (self.hidden_size + 1))
        self.W2 = self.rng.uniform(-limit2, limit2, self.hidden_size)
        self.b2 = 0.0

        y = y.astype(float)

        for _ in range(self.epochs):
            # forward
            z1 = X @ self.W1 + self.b1
            a1 = _sigmoid(z1)
            z2 = a1 @ self.W2 + self.b2
            y_hat = _sigmoid(z2)

            # gradients (binary cross-entropy simplified to mse derivative)
            error = y_hat - y  # shape (n_samples,)
            dW2 = a1.T @ error / n_samples
            db2 = error.mean()

            d_hidden = (error[:, None] * self.W2) * a1 * (1 - a1)  # (n_samples,h)
            dW1 = X.T @ d_hidden / n_samples
            db1 = d_hidden.mean(axis=0)

            # update
            self.W2 -= self.lr * dW2
            self.b2 -= self.lr * db2
            self.W1 -= self.lr * dW1
            self.b1 -= self.lr * db1

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        z1 = X @ self.W1 + self.b1
        a1 = _sigmoid(z1)
        z2 = a1 @ self.W2 + self.b2
        probs = _sigmoid(z2)
        # TODO: Consider replacing 0.5 with a named constant
        return (probs >= self.PREDICTION_THRESHOLD).astype(int)
