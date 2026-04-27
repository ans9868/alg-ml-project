"""PCA / truncated SVD baseline (proposal §6.5).

Fit only on training data; apply learned components to both train and test.
"""
from __future__ import annotations

import numpy as np
from sklearn.decomposition import TruncatedSVD


class PCAReducer:
    """Wrapper exposing the fit/transform interface required in proposal §14.3."""

    def __init__(self, k: int, random_state: int = 0):
        self.k = k
        self.random_state = random_state
        self._svd: TruncatedSVD | None = None

    def fit(self, X_train: np.ndarray) -> "PCAReducer":
        # n_components must be < n_features for TruncatedSVD; cap to be safe
        n_components = min(self.k, X_train.shape[1] - 1, X_train.shape[0] - 1)
        self._svd = TruncatedSVD(
            n_components=n_components, random_state=self.random_state
        )
        self._svd.fit(X_train)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self._svd is None:
            raise RuntimeError("PCAReducer.fit must be called before transform.")
        return self._svd.transform(X)

    def fit_transform(self, X_train: np.ndarray) -> np.ndarray:
        return self.fit(X_train).transform(X_train)

    @property
    def explained_variance_ratio(self) -> np.ndarray:
        if self._svd is None:
            raise RuntimeError("Not fit yet.")
        return self._svd.explained_variance_ratio_
