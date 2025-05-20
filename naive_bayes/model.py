from typing import Optional

import numpy as np


class MultinomialNB:
    """Bare-bones multinomial Naïve Bayes with Laplace smoothing."""

    def __init__(self, alpha: float = 1.0) -> None:
        self.alpha = alpha
        self.class_log_prior_: Optional[np.ndarray] = None
        self.feature_log_prob_: Optional[np.ndarray] = None
        # Add self.classes_ for consistency with scikit-learn and potential future use
        self.classes_: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "MultinomialNB":
        """Estimate P(class) and P(feature|class)."""
        self.classes_ = np.unique(y)
        n_classes = len(self.classes_)

        # Map original class labels to 0..n_classes-1 for indexing
        y_mapped = np.zeros_like(y, dtype=int)
        for i, cls_label in enumerate(self.classes_):
            y_mapped[y == cls_label] = i

        class_count = np.bincount(y_mapped, minlength=n_classes).astype(float)
        self.class_log_prior_ = np.log(class_count / class_count.sum())

        feature_count = np.zeros((n_classes, X.shape[1]), dtype=float)
        for i, cls_label in enumerate(self.classes_):
            # Use y_mapped for consistent indexing with feature_count
            feature_count[i] = X[y == cls_label].sum(axis=0) + self.alpha

        self.feature_log_prob_ = np.log(
            feature_count / feature_count.sum(axis=1, keepdims=True)
        )
        return self

    def _joint_log_likelihood(self, X: np.ndarray) -> np.ndarray:
        if self.feature_log_prob_ is None or self.class_log_prior_ is None:
            raise ValueError("Model has not been fitted yet.")
        return (X @ self.feature_log_prob_.T) + self.class_log_prior_

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return class with highest posterior for each sample."""
        if self.classes_ is None:
            raise ValueError("Model has not been fitted yet.")
        jll = self._joint_log_likelihood(X)
        # Map indexed predictions back to original class labels
        return self.classes_[np.argmax(jll, axis=1)]
