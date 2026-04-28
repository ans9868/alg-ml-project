"""Recompute metrics from saved compressed artifacts, no re-fitting needed.

Usage:
    python scripts/recompute_metrics.py results/phase0/runs/<universe>/<slice>/<method>/<run_dir>/

Loads `compressed.npz` and the corresponding raw test matrix from data/processed,
runs all metrics, and prints the result. The whole point of the saving upgrade
is that this works without touching the projection or re-fitting anything.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src import metrics
from src import preprocessing as pp
from src.utils import load_json, load_npz_dict

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
N_PAIRS = 50_000
SEED = 0
TRAIN_FRAC = 0.7


def main(run_dir_arg: str) -> None:
    run_dir = Path(run_dir_arg).resolve()
    if not run_dir.exists():
        raise SystemExit(f"Run dir does not exist: {run_dir}")

    meta = load_json(run_dir / "run_meta.json")
    print(f"Loaded run: {meta['method']}  universe={meta['universe']}  k={meta['k']}")

    compressed = load_npz_dict(run_dir / "compressed.npz")
    Z_compressed_test = compressed["Z_test"]
    print(f"  Z_test shape: {Z_compressed_test.shape}")

    # Reload the raw reference geometry the same way the smoke test does
    returns = pd.read_csv(
        PROCESSED_DIR / "returns_matrix.csv", index_col=0, parse_dates=True
    )
    ranked = pd.read_csv(PROCESSED_DIR / "ranked_universe.csv")
    universe_size = int(meta["universe"].split("_")[-1])
    tickers = ranked.head(universe_size)["ticker"].tolist()

    X = returns[tickers].dropna(how="all")
    X_train, X_test = pp.chronological_train_test_split(X, train_frac=TRAIN_FRAC)
    X_train_z, X_test_z, _, _ = pp.standardize_with_train_stats(X_train, X_test)
    Z_raw_test = X_test_z.values

    # Sanity: same row count
    assert Z_raw_test.shape[0] == Z_compressed_test.shape[0], (
        "Row mismatch between saved compressed test set and freshly built reference"
    )

    print(f"  Z_raw_test shape: {Z_raw_test.shape}")
    print()

    result = metrics.distance_distortion(
        Z_raw=Z_raw_test,
        Z_compressed=Z_compressed_test,
        n_pairs=N_PAIRS,
        seed=SEED,
    )

    print("Recomputed distance_distortion:")
    for kk, vv in result.items():
        print(f"  {kk:30s} {vv}")

    # Compare against the saved metric file as a regression check
    saved = load_json(run_dir / "metrics.json")["distance_distortion"]
    diff = {
        kk: abs(result[kk] - saved[kk]) for kk in saved if isinstance(saved[kk], float)
    }
    print()
    print("Diff vs saved (should be 0):")
    for kk, vv in diff.items():
        print(f"  {kk:30s} {vv}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(__doc__)
        sys.exit(2)
    main(sys.argv[1])
