# pragma: no cover
import matplotlib.pyplot as plt
import numpy as np

from perceptron.datasets import load_linear
from perceptron.model import (
    Perceptron,
    _sigmoid,  # Import _sigmoid
)

X_tr, y_tr, _, _ = load_linear()
losses = []
class PerceptronLog(Perceptron):
    def fit(self, X, y):
        n_samples, n_feat = X.shape
        # Xavier init
        # Ensure self.rng is initialized if not done by super().__init__
        if not hasattr(self, 'rng') or self.rng is None:
            # This is a fallback, ideally Perceptron.__init__ handles rng setup
            self.rng = np.random.default_rng(getattr(self, 'seed', None))

        limit1 = np.sqrt(6 / (n_feat + self.hidden_size))
        self.W1 = self.rng.uniform(-limit1, limit1, (n_feat, self.hidden_size))
        self.b1 = np.zeros(self.hidden_size)

        limit2 = np.sqrt(6 / (self.hidden_size + 1))
        self.W2 = self.rng.uniform(-limit2, limit2, self.hidden_size)
        self.b2 = 0.0

        y_true_float = y.astype(float)
        self.losses = [] # Initialize losses list

        for _ in range(self.epochs):
            # forward
            z1 = X @ self.W1 + self.b1
            a1 = _sigmoid(z1)
            z2 = a1 @ self.W2 + self.b2
            y_hat = _sigmoid(z2)

            # Calculate MSE loss
            mse = np.mean((y_hat - y_true_float)**2)
            self.losses.append(mse)

            # gradients (binary cross-entropy simplified to mse derivative)
            error = y_hat - y_true_float  # shape (n_samples,)
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

# Helper sigmoid function if not accessible from base class Perceptron
# This might be needed if _sigmoid is a private member of Perceptron
# For now, assuming Perceptron or its context provides _sigmoid
# def _sigmoid(z):
#     return 1.0 / (1.0 + np.exp(-z))

clf = PerceptronLog(epochs=2000, lr=0.1, hidden_size=6).fit(X_tr, y_tr)
# assume you stored self.losses inside original fit loop
plt.plot(clf.losses)
plt.xlabel("Epoch")
plt.ylabel("MSE")
plt.title("Perceptron training loss")
plt.savefig("figures/perceptron_loss.png", dpi=200)
