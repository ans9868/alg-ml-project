"""PCA / truncated SVD baseline (proposal §6.5).

Fit only on training data; apply learned components to both train and test.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from sklearn.decomposition import TruncatedSVD

from src.utils import load_npz_dict, save_npz_dict


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

    # -----------------------------------------------------------------
    # Persistence
    # -----------------------------------------------------------------
    def save(self, path: Path) -> None:
        if self._svd is None:
            raise RuntimeError("Cannot save an unfit PCAReducer.")
        save_npz_dict(
            path,
            method=np.array(["pca"]),
            k=np.array([self.k]),
            random_state=np.array([self.random_state]),
            components=self._svd.components_,            # (k, N)
            explained_variance=self._svd.explained_variance_,
            explained_variance_ratio=self._svd.explained_variance_ratio_,
            singular_values=self._svd.singular_values_,
        )

    @classmethod
    def load(cls, path: Path) -> "PCAReducer":
        d = load_npz_dict(path)
        obj = cls(k=int(d["k"][0]), random_state=int(d["random_state"][0]))
        # Reconstruct a TruncatedSVD shell with the stored components
        n_components = d["components"].shape[0]
        svd = TruncatedSVD(n_components=n_components, random_state=int(d["random_state"][0]))
        svd.components_ = d["components"]
        svd.explained_variance_ = d["explained_variance"]
        svd.explained_variance_ratio_ = d["explained_variance_ratio"]
        svd.singular_values_ = d["singular_values"]
        obj._svd = svd
        return obj
