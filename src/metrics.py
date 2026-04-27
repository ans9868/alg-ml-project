"""Evaluation metrics (proposal §7).

Phase 0 implements only Metric 1 (pairwise distance distortion).
"""
from __future__ import annotations

import numpy as np


def sample_pair_indices(n: int, n_pairs: int, rng: np.random.Generator) -> np.ndarray:
    """Sample n_pairs unique (i, j) pairs with i < j from {0, ..., n-1}.

    Returns array of shape (n_pairs, 2).
    """
    max_pairs = n * (n - 1) // 2
    if n_pairs >= max_pairs:
        # Return all pairs
        i, j = np.triu_indices(n, k=1)
        return np.stack([i, j], axis=1)

    # Rejection-free sampling via uniform integers in [0, max_pairs) mapped to (i,j)
    chosen = rng.choice(max_pairs, size=n_pairs, replace=False)
    # Map linear index -> (i,j) with i<j: classic upper-triangular mapping
    # Using vectorized conversion: i = floor((-1 + sqrt(1 + 8*linear)) / 2) -- but
    # the simpler approach is to enumerate triu and gather.
    i, j = np.triu_indices(n, k=1)
    return np.stack([i[chosen], j[chosen]], axis=1)


def pairwise_distances(Z: np.ndarray, pair_idx: np.ndarray) -> np.ndarray:
    """Euclidean distances for the given (i,j) row pairs of Z.

    Z: (n_samples, n_features)
    pair_idx: (n_pairs, 2)
    Returns: (n_pairs,)
    """
    diffs = Z[pair_idx[:, 0]] - Z[pair_idx[:, 1]]
    return np.linalg.norm(diffs, axis=1)


def distance_distortion(
    Z_raw: np.ndarray,
    Z_compressed: np.ndarray,
    n_pairs: int = 50_000,
    seed: int = 0,
) -> dict:
    """Pairwise distance distortion (proposal §7.1).

    rho_ij = ||Z_compressed[i] - Z_compressed[j]|| / ||Z_raw[i] - Z_raw[j]||

    Returns a dict of summary statistics:
      mean_abs_distortion = mean(|rho - 1|)
      median_abs_distortion
      p95_abs_distortion
      mean_rho
    """
    assert Z_raw.shape[0] == Z_compressed.shape[0], "Row counts must match"
    n = Z_raw.shape[0]
    rng = np.random.default_rng(seed)
    pair_idx = sample_pair_indices(n, n_pairs, rng)

    d_raw = pairwise_distances(Z_raw, pair_idx)
    d_comp = pairwise_distances(Z_compressed, pair_idx)

    # Avoid divide-by-zero for any duplicate-day pairs (shouldn't happen but be safe)
    mask = d_raw > 1e-12
    rho = d_comp[mask] / d_raw[mask]
    abs_dev = np.abs(rho - 1.0)

    return {
        "n_pairs": int(mask.sum()),
        "mean_abs_distortion": float(abs_dev.mean()),
        "median_abs_distortion": float(np.median(abs_dev)),
        "p95_abs_distortion": float(np.quantile(abs_dev, 0.95)),
        "mean_rho": float(rho.mean()),
    }
