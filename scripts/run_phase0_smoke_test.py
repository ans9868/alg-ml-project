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
from src.pca_baseline import PCAReducer

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
RESULTS_DIR = PROJECT_ROOT / "results"

UNIVERSE = "top_100"
K = 20
TRAIN_FRAC = 0.7
N_PAIRS = 50_000
SEED = 0


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

    Z_raw_test = X_test_z.values  # reference geometry (proposal §6.6)

    print(f"[4/5] PCA fit on train (k={K}), transform test")
    pca = PCAReducer(k=K, random_state=SEED).fit(X_train_z.values)
    Z_pca_test = pca.transform(X_test_z.values)
    print(f"      Z_raw_test shape: {Z_raw_test.shape}")
    print(f"      Z_pca_test shape: {Z_pca_test.shape}")
    print(f"      explained variance ratio (cumulative top {K}): {pca.explained_variance_ratio.sum():.4f}")

    print(f"[5/5] Distance distortion (sampled {N_PAIRS} pairs)")
    result = metrics.distance_distortion(
        Z_raw=Z_raw_test,
        Z_compressed=Z_pca_test,
        n_pairs=N_PAIRS,
        seed=SEED,
    )

    print()
    print("=" * 60)
    print("PHASE 0 SMOKE TEST RESULT")
    print("=" * 60)
    print(f"Universe:                      {UNIVERSE}")
    print(f"N (assets):                    {len(top_100)}")
    print(f"k (compressed dim):            {K}")
    print(f"PCA explained variance:        {pca.explained_variance_ratio.sum():.4f}")
    print(f"Test rows:                     {len(X_test)}")
    print(f"Pairs sampled:                 {result['n_pairs']:,}")
    print(f"  mean |rho - 1|               {result['mean_abs_distortion']:.4f}")
    print(f"  median |rho - 1|             {result['median_abs_distortion']:.4f}")
    print(f"  95th-pct |rho - 1|           {result['p95_abs_distortion']:.4f}")
    print(f"  mean rho                     {result['mean_rho']:.4f}")
    print()

    # Persist a small results CSV
    out = {
        "universe": UNIVERSE,
        "n_assets": len(top_100),
        "k": K,
        "method": "pca",
        "seed": SEED,
        "train_frac": TRAIN_FRAC,
        "n_pairs_sampled": result["n_pairs"],
        "explained_variance": float(pca.explained_variance_ratio.sum()),
        "mean_abs_distortion": result["mean_abs_distortion"],
        "median_abs_distortion": result["median_abs_distortion"],
        "p95_abs_distortion": result["p95_abs_distortion"],
        "mean_rho": result["mean_rho"],
    }
    pd.DataFrame([out]).to_csv(
        RESULTS_DIR / "phase0_smoke_test.csv", index=False
    )
    with open(RESULTS_DIR / "phase0_smoke_test.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"Result written to results/phase0_smoke_test.csv and .json")


if __name__ == "__main__":
    main()
