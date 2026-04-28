"""Parallel execution backends: local (sequential) and Ray (proposal §9).

Both backends call the same `_run_single_config` worker so they produce
identical result schemas.
"""
from __future__ import annotations

import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src import metrics
from src.experiment_config import ExperimentConfig
from src.pca_baseline import PCAReducer
from src.projections import DenseGaussianJL, SparseJL
from src.utils import (
    config_to_dict,
    get_git_sha,
    now_utc_iso,
    save_json,
    save_npz_dict,
)


# ---------------------------------------------------------------------
# Per-config worker
# ---------------------------------------------------------------------
def _build_method(config: ExperimentConfig):
    """Instantiate a method object (or None for raw) from a config."""
    if config.method == "raw":
        return None
    if config.method == "pca":
        return PCAReducer(k=config.k, random_state=config.seed)
    if config.method == "dense_jl":
        return DenseGaussianJL(k=config.k, seed=config.seed)
    if config.method == "sparse_jl":
        return SparseJL(k=config.k, s=config.s, seed=config.seed)
    raise ValueError(f"Unknown method {config.method!r}")


def _run_single_config(
    config: ExperimentConfig,
    data: dict[str, Any],
    save_artifacts: bool = True,
    artifact_base: Path | None = None,
    git_sha: str = "unknown",
) -> dict:
    """Execute one config: fit -> transform -> compute metrics -> (save).

    Returns one dict (one row of the result DataFrame).
    """
    X_train = data["X_train"]   # standardized train (np.ndarray)
    X_test = data["X_test"]     # standardized test (np.ndarray)

    # --- Fit + transform (timed separately for runtime metric) ---
    method = _build_method(config)
    if method is None:
        fit_seconds = 0.0
        t0 = time.perf_counter()
        Z_test = X_test                 # raw / identity
        transform_seconds = time.perf_counter() - t0
        nnz = -1
        explained_var = None
        proj_shape = None
    else:
        t0 = time.perf_counter()
        method.fit(X_train)
        fit_seconds = time.perf_counter() - t0
        t0 = time.perf_counter()
        Z_test = method.transform(X_test)
        transform_seconds = time.perf_counter() - t0
        nnz = int(getattr(method, "nnz", -1))
        explained_var = (
            float(method.explained_variance_ratio.sum())
            if config.method == "pca"
            else None
        )
        # projection matrix shape for sparsity ratio
        if config.method == "pca":
            proj_shape = method._svd.components_.shape  # (k, N)
        elif config.method == "dense_jl":
            proj_shape = method.A.shape                 # (N, k)
        elif config.method == "sparse_jl":
            proj_shape = method.S.shape                 # (N, k)
        else:
            proj_shape = None

    # --- Metrics (timed together; individual metric times are sub-ms) ---
    t0 = time.perf_counter()
    dist = metrics.distance_distortion(
        Z_raw=X_test,
        Z_compressed=Z_test,
        n_pairs=config.n_pairs,
        seed=config.seed,
    )
    nn = metrics.nearest_neighbor_overlap(
        Z_raw=X_test,
        Z_compressed=Z_test,
        m_values=(5, 10),
    )
    ari = metrics.clustering_ari(
        Z_raw=X_test,
        Z_compressed=Z_test,
        cluster_counts=(3, 5, 8),
        seed=config.seed,
    )
    anom = metrics.anomaly_recall(
        Z_raw=X_test,
        Z_compressed=Z_test,
        top_pct=0.05,
    )
    metrics_seconds = time.perf_counter() - t0

    runtime = metrics.runtime_sparsity_summary(
        fit_seconds=fit_seconds,
        transform_seconds=transform_seconds,
        metrics_seconds=metrics_seconds,
        nnz=nnz if nnz >= 0 else 0,
        matrix_shape=proj_shape,
        method=config.method,
    )

    metrics_payload = {
        "distance_distortion": dist,
        "nearest_neighbor": nn,
        "clustering_ari": ari,
        "anomaly_recall": anom,
        "runtime_sparsity": runtime,
    }

    # --- Save artifacts ---
    if save_artifacts and artifact_base is not None:
        s_part = f"s{config.s}" if config.s is not None else "sNA"
        run_dir = (
            artifact_base
            / config.universe
            / config.slice
            / config.method
            / f"k{config.k}_{s_part}_seed{config.seed}"
        )
        run_dir.mkdir(parents=True, exist_ok=True)

        save_npz_dict(run_dir / "compressed.npz", Z_test=Z_test)
        if method is not None:
            method.save(run_dir / "projection.npz")
        save_json(run_dir / "metrics.json", metrics_payload)
        save_json(
            run_dir / "run_meta.json",
            {
                **asdict(config),
                "nnz": nnz,
                "explained_variance": explained_var,
                "fit_seconds": fit_seconds,
                "transform_seconds": transform_seconds,
                "metrics_seconds": metrics_seconds,
                "git_sha": git_sha,
                "timestamp_utc": now_utc_iso(),
            },
        )

    # --- Build result row (flatten all metrics with sensible prefixes) ---
    row = {
        **asdict(config),
        "nnz": nnz,
        "explained_variance": explained_var,
        "fit_seconds": fit_seconds,
        "transform_seconds": transform_seconds,
        "metrics_seconds": metrics_seconds,
        # distance distortion
        "dd_mean_abs": dist["mean_abs_distortion"],
        "dd_median_abs": dist["median_abs_distortion"],
        "dd_p95_abs": dist["p95_abs_distortion"],
        "dd_mean_rho": dist["mean_rho"],
        "dd_n_pairs": dist["n_pairs"],
        # nearest neighbor
        "nn_overlap_at_5": nn["nn_overlap_at_5"],
        "nn_overlap_at_5_p10": nn["nn_overlap_at_5_p10"],
        "nn_overlap_at_10": nn["nn_overlap_at_10"],
        "nn_overlap_at_10_p10": nn["nn_overlap_at_10_p10"],
        # clustering ARI
        "ari_C3": ari["ari_C3"],
        "ari_C5": ari["ari_C5"],
        "ari_C8": ari["ari_C8"],
        "ari_mean": ari["ari_mean"],
        # anomaly recall
        "anomaly_recall_5pct": anom["anomaly_recall_at_5pct"],
        "anomaly_precision_5pct": anom["anomaly_precision_at_5pct"],
        "anomaly_score_spearman": anom["anomaly_score_spearman"],
        "anomaly_top10_overlap": anom["anomaly_top10_overlap"],
        # runtime
        "sparsity_ratio": runtime["sparsity_ratio"],
    }
    return row


# ---------------------------------------------------------------------
# Local sequential backend (for debugging)
# ---------------------------------------------------------------------
def _run_local(
    configs: list[ExperimentConfig],
    prepared_data: dict[str, dict],
    save_artifacts: bool,
    artifact_base: Path | None,
    git_sha: str,
) -> pd.DataFrame:
    rows: list[dict] = []
    for c in configs:
        if c.universe not in prepared_data:
            raise KeyError(f"No prepared data for universe {c.universe!r}")
        rows.append(
            _run_single_config(c, prepared_data[c.universe], save_artifacts, artifact_base, git_sha)
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------
# Ray backend
# ---------------------------------------------------------------------
def _run_ray(
    configs: list[ExperimentConfig],
    prepared_data: dict[str, dict],
    num_cpus: int | None,
    save_artifacts: bool,
    artifact_base: Path | None,
    git_sha: str,
) -> pd.DataFrame:
    import ray

    ray.init(num_cpus=num_cpus, ignore_reinit_error=True, log_to_driver=False)

    # Put each universe's prepared data in the object store ONCE
    data_refs = {u: ray.put(d) for u, d in prepared_data.items()}

    @ray.remote
    def _remote(config_dict, data, save_artifacts, artifact_base_str, git_sha):
        # `data` arrives already dereferenced by Ray when the caller passes
        # an ObjectRef as an argument.
        from pathlib import Path as _Path

        from src.experiment_config import ExperimentConfig as _EC
        from src.parallel_backend import _run_single_config as _rsc

        config = _EC(**config_dict)
        ab = _Path(artifact_base_str) if artifact_base_str else None
        return _rsc(config, data, save_artifacts, ab, git_sha)

    futures = []
    artifact_base_str = str(artifact_base) if artifact_base else ""
    for c in configs:
        if c.universe not in data_refs:
            raise KeyError(f"No prepared data for universe {c.universe!r}")
        futures.append(
            _remote.remote(asdict(c), data_refs[c.universe], save_artifacts, artifact_base_str, git_sha)
        )

    rows = ray.get(futures)
    ray.shutdown()
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------
def run_experiment_grid(
    configs: list[ExperimentConfig],
    prepared_data: dict[str, dict],
    backend: str = "local",
    num_cpus: int | None = None,
    save_artifacts: bool = True,
    artifact_base: Path | None = None,
    git_sha: str | None = None,
) -> pd.DataFrame:
    """Run a grid of ExperimentConfigs and return a long-format DataFrame.

    `prepared_data` is keyed by universe name; each value is a dict with at
    least `X_train` and `X_test` numpy arrays (already standardized).
    """
    if git_sha is None:
        git_sha = get_git_sha()
    if backend == "local":
        return _run_local(configs, prepared_data, save_artifacts, artifact_base, git_sha)
    if backend == "ray":
        return _run_ray(configs, prepared_data, num_cpus, save_artifacts, artifact_base, git_sha)
    raise ValueError(f"Unknown backend {backend!r}")
