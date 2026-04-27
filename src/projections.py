"""Random projection methods (proposal §6.3, §6.4).

Both projections are data-independent: `fit` only uses the input dimensionality
N (and the seed), not the actual data values. They expose the same fit /
transform / fit_transform interface as PCAReducer.
"""
from __future__ import annotations

import numpy as np


class DenseGaussianJL:
    """Dense Gaussian Johnson--Lindenstrauss projection (proposal §6.3).

    Builds A in R^{N x k} with entries A_ij ~ N(0, 1/k).
    Compressed representation: Z = X @ A.
    """

    def __init__(self, k: int, seed: int):
        self.k = k
        self.seed = seed
        self.A: np.ndarray | None = None

    def fit(self, X_train: np.ndarray) -> "DenseGaussianJL":
        N = X_train.shape[1]
        rng = np.random.default_rng(self.seed)
        self.A = rng.normal(loc=0.0, scale=np.sqrt(1.0 / self.k), size=(N, self.k))
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.A is None:
            raise RuntimeError("DenseGaussianJL.fit must be called before transform.")
        return X @ self.A

    def fit_transform(self, X_train: np.ndarray) -> np.ndarray:
        return self.fit(X_train).transform(X_train)

    @property
    def nnz(self) -> int:
        """Number of nonzero entries (always N*k for dense)."""
        return int(self.A.size) if self.A is not None else 0


class SparseJL:
    """Sparse Johnson--Lindenstrauss projection (proposal §6.4).

    Builds S in R^{N x k} where each row j has exactly s nonzero entries placed
    at uniformly-random column indices, each carrying random sign x 1/sqrt(s).
    Compressed representation: Z = X @ S.

    Note: must satisfy s <= k.
    """

    def __init__(self, k: int, s: int, seed: int):
        if s > k:
            raise ValueError(f"sparse JL requires s <= k, got s={s}, k={k}")
        self.k = k
        self.s = s
        self.seed = seed
        self.S: np.ndarray | None = None

    def fit(self, X_train: np.ndarray) -> "SparseJL":
        N = X_train.shape[1]
        rng = np.random.default_rng(self.seed)
        S = np.zeros((N, self.k), dtype=np.float64)
        scale = 1.0 / np.sqrt(self.s)

        for j in range(N):
            # Pick s columns uniformly without replacement
            cols = rng.choice(self.k, size=self.s, replace=False)
            # Random signs +/- 1
            signs = rng.choice([-1.0, 1.0], size=self.s)
            S[j, cols] = signs * scale

        self.S = S
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.S is None:
            raise RuntimeError("SparseJL.fit must be called before transform.")
        return X @ self.S

    def fit_transform(self, X_train: np.ndarray) -> np.ndarray:
        return self.fit(X_train).transform(X_train)

    @property
    def nnz(self) -> int:
        """Number of nonzero entries in the projection matrix."""
        return int(np.count_nonzero(self.S)) if self.S is not None else 0
