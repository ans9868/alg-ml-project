"""Phase 0 smoke test — end-to-end pipeline on top_100 with PCA k=20.

Exercises:
  - data load (returns_matrix.csv produced by run_data_pipeline.py)
  - top_100 universe selection
  - chronological 70/30 train/test split
  - z-score standardization with train statistics only
  - PCA fit on train, transform on test
  - distance distortion metric on subsampled test pairs

Goal: confirm the architecture produces a single coherent number, and that
the schema connecting modules is consistent.

Run from project root after run_data_pipeline.py:
    python scripts/run_phase0_smoke_test.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src import metrics
from src import preprocessing as pp
from src.crashsketch import CrashSketch
from src.pca_baseline import PCAReducer
from src.projections import DenseGaussianJL, SparseJL
from src.utils import (
    get_git_sha,
    now_utc_iso,
    save_json,
    save_npz_dict,
)

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
RESULTS_DIR = PROJECT_ROOT / "results"
ARTIFACT_DIR = RESULTS_DIR / "phase0" / "runs"

UNIVERSE = "top_100"
SLICE = "full_incl_covid"
K = 20
SPARSE_S = 3  # primary sparse JL value per proposal §6.4
TRAIN_FRAC = 0.7
N_PAIRS = 50_000
SEED = 0
SAVE_ARTIFACTS = True


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"[1/5] Loading returns matrix and {UNIVERSE} ticker list")
    returns = pd.read_csv(
        PROCESSED_DIR / "returns_matrix.csv", index_col=0, parse_dates=True
    )
    ranked = pd.read_csv(PROCESSED_DIR / "ranked_universe.csv")
    top_100 = ranked.head(100)["ticker"].tolist()
    print(f"      returns shape (T x N candidates):  {returns.shape}")
    print(f"      top_100 first 5: {top_100[:5]}")

    # Restrict to top_100 columns
    X = returns[top_100].copy()
    # Drop any rows that became all-NaN after filtering (shouldn't happen for survivors)
    X = X.dropna(how="all")
    print(f"      X shape after universe filter:     {X.shape}")
    assert X.isna().sum().sum() == 0, "no NaNs expected for survivor universe"

    print(f"[2/5] Chronological 70/30 train/test split")
    X_train, X_test = pp.chronological_train_test_split(X, train_frac=TRAIN_FRAC)
    print(f"      train shape: {X_train.shape}  ({X_train.index.min().date()} -> {X_train.index.max().date()})")
    print(f"      test shape:  {X_test.shape}  ({X_test.index.min().date()} -> {X_test.index.max().date()})")

    print("[3/5] Standardize with train statistics only")
    X_train_z, X_test_z, mu, sigma = pp.standardize_with_train_stats(X_train, X_test)
    print(f"      train mean(|mu|) ~ {mu.abs().mean():.6f}")
    print(f"      train mean(sigma) ~ {sigma.mean():.6f}")
    assert X_train_z.isna().sum().sum() == 0
    assert X_test_z.isna().sum().sum() == 0

    X_train_arr = X_train_z.values
    X_test_arr = X_test_z.values
    Z_raw_test = X_test_arr  # reference geometry (proposal §6.6)

    print(f"[4/5] Fit & transform all 4 methods at k={K}")
    methods: dict[str, dict] = {}

    # raw: identity (no method object)
    methods["raw"] = {"Z_test": Z_raw_test, "obj": None, "extras": {}}

    # PCA
    pca = PCAReducer(k=K, random_state=SEED).fit(X_train_arr)
    methods["pca"] = {
        "Z_test": pca.transform(X_test_arr),
        "obj": pca,
        "extras": {"explained_variance": float(pca.explained_variance_ratio.sum())},
    }

    # Dense Gaussian JL
    dense_jl = DenseGaussianJL(k=K, seed=SEED).fit(X_train_arr)
    methods["dense_jl"] = {
        "Z_test": dense_jl.transform(X_test_arr),
        "obj": dense_jl,
        "extras": {"nnz": dense_jl.nnz},
    }

    # Sparse JL (s=3)
    sparse_jl = SparseJL(k=K, s=SPARSE_S, seed=SEED).fit(X_train_arr)
    methods["sparse_jl_s3"] = {
        "Z_test": sparse_jl.transform(X_test_arr),
        "obj": sparse_jl,
        "extras": {"nnz": sparse_jl.nnz, "s": SPARSE_S},
    }

    # CrashSketch with rotation (the proposed novel method)
    cs_R = CrashSketch(k=K, s=SPARSE_S, seed=SEED, use_rotation=True).fit(X_train_arr)
    methods["crashsketch_R_s3"] = {
        "Z_test": cs_R.transform(X_test_arr),
        "obj": cs_R,
        "extras": {
            "nnz_S": cs_R.nnz,
            "nnz_total_with_R": cs_R.total_nnz_with_rotation,
            "s": SPARSE_S,
            "use_rotation": True,
        },
    }

    # CrashSketch without rotation (ablation: shows what rotation contributes)
    cs_noR = CrashSketch(k=K, s=SPARSE_S, seed=SEED, use_rotation=False).fit(X_train_arr)
    methods["crashsketch_noR_s3"] = {
        "Z_test": cs_noR.transform(X_test_arr),
        "obj": cs_noR,
        "extras": {
            "nnz_S": cs_noR.nnz,
            "s": SPARSE_S,
            "use_rotation": False,
        },
    }

    for name, m in methods.items():
        print(f"      {name:15s} -> Z_test shape {m['Z_test'].shape}  extras={m['extras']}")

    print(f"[5/5] All 5 metrics ({N_PAIRS:,} sampled pairs for distance, seed={SEED})")
    rows = []
    git_sha = get_git_sha(PROJECT_ROOT)
    for name, m in methods.items():
        dist = metrics.distance_distortion(
            Z_raw=Z_raw_test, Z_compressed=m["Z_test"],
            n_pairs=N_PAIRS, seed=SEED,
        )
        nn = metrics.nearest_neighbor_overlap(
            Z_raw=Z_raw_test, Z_compressed=m["Z_test"],
            m_values=(5, 10),
        )
        ari = metrics.clustering_ari(
            Z_raw=Z_raw_test, Z_compressed=m["Z_test"],
            cluster_counts=(3, 5, 8), seed=SEED,
        )
        anom = metrics.anomaly_recall(
            Z_raw=Z_raw_test, Z_compressed=m["Z_test"],
            top_pct=0.05,
        )
        row = {
            "method": name,
            "universe": UNIVERSE,
            "slice": SLICE,
            "n_assets": len(top_100),
            "k": K,
            "seed": SEED,
            "train_frac": TRAIN_FRAC,
            # distance distortion
            "dd_mean_abs": dist["mean_abs_distortion"],
            "dd_median_abs": dist["median_abs_distortion"],
            "dd_p95_abs": dist["p95_abs_distortion"],
            "dd_mean_rho": dist["mean_rho"],
            # nearest neighbor
            "nn_overlap_at_5": nn["nn_overlap_at_5"],
            "nn_overlap_at_10": nn["nn_overlap_at_10"],
            # clustering ari
            "ari_C3": ari["ari_C3"],
            "ari_C5": ari["ari_C5"],
            "ari_C8": ari["ari_C8"],
            "ari_mean": ari["ari_mean"],
            # anomaly
            "anomaly_recall_5pct": anom["anomaly_recall_at_5pct"],
            "anomaly_score_spearman": anom["anomaly_score_spearman"],
            "anomaly_top10_overlap": anom["anomaly_top10_overlap"],
            **m["extras"],
        }
        rows.append(row)

        if SAVE_ARTIFACTS:
            # Build per-config run directory
            uses_s = ("s" in name) or name.startswith("crashsketch")
            s_part = f"s{SPARSE_S}" if uses_s else "sNA"
            run_dir = (
                ARTIFACT_DIR
                / UNIVERSE
                / SLICE
                / name
                / f"k{K}_{s_part}_seed{SEED}"
            )
            run_dir.mkdir(parents=True, exist_ok=True)

            # Save compressed test matrix + train/test date indices
            save_npz_dict(
                run_dir / "compressed.npz",
                Z_test=m["Z_test"],
                test_dates=X_test_z.index.astype("int64").to_numpy(),
            )

            # Save projection matrix where applicable
            obj = m.get("obj")
            if obj is not None:
                obj.save(run_dir / "projection.npz")

            # Save metric results + run metadata
            save_json(run_dir / "metrics.json", {
                "distance_distortion": dist,
                "nearest_neighbor": nn,
                "clustering_ari": ari,
                "anomaly_recall": anom,
            })
            save_json(
                run_dir / "run_meta.json",
                {
                    "method": name,
                    "universe": UNIVERSE,
                    "slice": SLICE,
                    "k": K,
                    "s": SPARSE_S if uses_s else None,
                    "seed": SEED,
                    "n_assets": len(top_100),
                    "train_frac": TRAIN_FRAC,
                    "git_sha": git_sha,
                    "timestamp_utc": now_utc_iso(),
                    **m["extras"],
                },
            )

    df = pd.DataFrame(rows)
    print()
    print("=" * 110)
    print(f"PHASE 0 SMOKE TEST  -- universe={UNIVERSE}, N={len(top_100)}, k={K}, single seed={SEED}")
    print("=" * 110)
    print(
        df[[
            "method",
            "dd_mean_abs", "dd_mean_rho",
            "nn_overlap_at_5", "nn_overlap_at_10",
            "ari_mean",
            "anomaly_recall_5pct", "anomaly_top10_overlap",
        ]].to_string(index=False, float_format=lambda x: f"{x:.4f}")
    )
    print()
    print("Notes:")
    print("  - Single-seed numbers for randomized methods are noise-dominated; treat as sanity check, not result.")
    print("  - 'raw' is identity --> all metrics at perfect value by construction.")
    print("  - 'crashsketch_R_s3' is the proposed novel method (sparse + rotation); 'crashsketch_noR_s3' is the ablation.")
    print()

    df.to_csv(RESULTS_DIR / "phase0_smoke_test.csv", index=False)
    with open(RESULTS_DIR / "phase0_smoke_test.json", "w") as f:
        json.dump(rows, f, indent=2)
    print(f"Results written to results/phase0_smoke_test.csv and .json")


if __name__ == "__main__":
    main()
