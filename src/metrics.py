"""Evaluation metrics (proposal §7).

Implements all five metrics:
  1. Pairwise distance distortion (§7.1)
  2. Nearest-neighbor preservation (§7.2)
  3. Clustering stability via Adjusted Rand Index (§7.3)
  4. Anomaly recall (§7.4)
  5. Runtime / sparsity (§7.5) -- captured by callers, summarized in metrics.

All metrics return a dict of summary statistics so the runner can flatten
them into a single result row.
"""
from __future__ import annotations

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
from sklearn.neighbors import NearestNeighbors


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


# ---------------------------------------------------------------------
# Metric 2: Nearest-Neighbor Preservation (proposal §7.2)
# ---------------------------------------------------------------------
def nearest_neighbor_overlap(
    Z_raw: np.ndarray,
    Z_compressed: np.ndarray,
    m_values: tuple[int, ...] = (5, 10),
) -> dict:
    """For each test point, find its m nearest neighbors in raw and compressed
    space; report mean fractional overlap across all points and m values.
    """
    assert Z_raw.shape[0] == Z_compressed.shape[0]
    n = Z_raw.shape[0]

    # Fit nearest-neighbor indexes on each space (k+1 because the point itself
    # is always its own closest neighbor at distance 0; we drop it after).
    max_m = max(m_values)
    nn_raw = NearestNeighbors(n_neighbors=max_m + 1).fit(Z_raw)
    nn_comp = NearestNeighbors(n_neighbors=max_m + 1).fit(Z_compressed)

    _, idx_raw = nn_raw.kneighbors(Z_raw)         # (n, max_m+1)
    _, idx_comp = nn_comp.kneighbors(Z_compressed)
    # Drop the self-match (column 0)
    idx_raw = idx_raw[:, 1:]
    idx_comp = idx_comp[:, 1:]

    out = {"n_points": int(n)}
    for m in m_values:
        # Per-point overlap fraction
        overlaps = np.array(
            [
                len(set(idx_raw[i, :m]) & set(idx_comp[i, :m])) / m
                for i in range(n)
            ]
        )
        out[f"nn_overlap_at_{m}"] = float(overlaps.mean())
        out[f"nn_overlap_at_{m}_p10"] = float(np.quantile(overlaps, 0.10))
    return out


# ---------------------------------------------------------------------
# Metric 3: Clustering Stability (Adjusted Rand Index, proposal §7.3)
# ---------------------------------------------------------------------
def clustering_ari(
    Z_raw: np.ndarray,
    Z_compressed: np.ndarray,
    cluster_counts: tuple[int, ...] = (3, 5, 8),
    seed: int = 0,
) -> dict:
    """For each cluster count C, run KMeans on raw and compressed, compare
    via Adjusted Rand Index. Higher ARI = clusterings agree more.
    """
    assert Z_raw.shape[0] == Z_compressed.shape[0]
    out: dict = {}
    for C in cluster_counts:
        km_raw = KMeans(n_clusters=C, n_init=10, random_state=seed).fit(Z_raw)
        km_comp = KMeans(n_clusters=C, n_init=10, random_state=seed).fit(Z_compressed)
        ari = float(adjusted_rand_score(km_raw.labels_, km_comp.labels_))
        out[f"ari_C{C}"] = ari
    out["ari_mean"] = float(np.mean(list(out.values())))
    return out


# ---------------------------------------------------------------------
# Metric 4: Anomaly Recall (proposal §7.4)
# ---------------------------------------------------------------------
def anomaly_recall(
    Z_raw: np.ndarray,
    Z_compressed: np.ndarray,
    top_pct: float = 0.05,
) -> dict:
    """Define unusual days as the top top_pct of rows by L2 norm.
    Compute set overlap between raw and compressed top-pct sets.
    Also report Spearman-style rank correlation of the anomaly scores.
    """
    assert Z_raw.shape[0] == Z_compressed.shape[0]
    n = Z_raw.shape[0]
    k_top = max(1, int(np.ceil(n * top_pct)))

    a_raw = np.linalg.norm(Z_raw, axis=1)
    a_comp = np.linalg.norm(Z_compressed, axis=1)

    top_raw = set(np.argsort(-a_raw)[:k_top].tolist())
    top_comp = set(np.argsort(-a_comp)[:k_top].tolist())

    overlap = len(top_raw & top_comp)
    recall = overlap / len(top_raw)
    precision = overlap / len(top_comp)

    # Score correlation across all points (Pearson on ranks ~ Spearman)
    rank_raw = np.argsort(np.argsort(a_raw))
    rank_comp = np.argsort(np.argsort(a_comp))
    if len(rank_raw) > 1:
        spearman = float(np.corrcoef(rank_raw, rank_comp)[0, 1])
    else:
        spearman = float("nan")

    # Top-10 unusual day overlap (separate from top-pct)
    k10 = min(10, n)
    top10_raw = set(np.argsort(-a_raw)[:k10].tolist())
    top10_comp = set(np.argsort(-a_comp)[:k10].tolist())
    top10_overlap = len(top10_raw & top10_comp) / k10

    return {
        "anomaly_n_top": int(k_top),
        "anomaly_recall_at_5pct": float(recall),
        "anomaly_precision_at_5pct": float(precision),
        "anomaly_score_spearman": spearman,
        "anomaly_top10_overlap": float(top10_overlap),
    }


# ---------------------------------------------------------------------
# Metric 5: Runtime / Sparsity (proposal §7.5)
# ---------------------------------------------------------------------
def runtime_sparsity_summary(
    fit_seconds: float,
    transform_seconds: float,
    metrics_seconds: float,
    nnz: int,
    matrix_shape: tuple[int, int] | None,
    method: str,
) -> dict:
    """Compose runtime / sparsity numbers into a metric dict. Caller is
    responsible for measuring fit/transform/metrics times; this just packages
    them into the standard summary schema.
    """
    if matrix_shape is not None:
        n_rows, n_cols = matrix_shape
        total_entries = n_rows * n_cols
        sparsity_ratio = (
            1.0 - (nnz / total_entries) if total_entries > 0 and nnz >= 0 else float("nan")
        )
    else:
        sparsity_ratio = float("nan")
    return {
        "fit_seconds": float(fit_seconds),
        "transform_seconds": float(transform_seconds),
        "metrics_seconds": float(metrics_seconds),
        "nnz": int(nnz),
        "sparsity_ratio": float(sparsity_ratio),
        "method_family": method,
    }
