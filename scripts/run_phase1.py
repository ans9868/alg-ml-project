"""Phase 1 grid runner.

Runs an ExperimentConfig grid via either local or Ray backend, saves
artifacts, and aggregates results into a single long-format CSV.

Usage:
    python scripts/run_phase1.py --backend local --subset smoke
    python scripts/run_phase1.py --backend ray   --subset core
    python scripts/run_phase1.py --backend ray   --subset full

Subsets:
  smoke  -- top_100, k in {20}, dense+sparse+pca+raw, seeds 0..2
            ~12 configs; finishes in seconds. Used to verify Ray works.
  core   -- top_100, full k grid, full s grid, seeds 0..9, single slice
            ~280 configs; finishes in 1-2 minutes locally.
  full   -- all 3 universes, all 3 slices, full grid, 50 seeds
            ~30K configs; intended for cloud.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src import preprocessing as pp
from src.experiment_config import ExperimentConfig, generate_experiment_grid
from src.parallel_backend import run_experiment_grid

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
RESULTS_DIR = PROJECT_ROOT / "results" / "phase1"
ARTIFACT_BASE = RESULTS_DIR / "runs"


# ---------------------------------------------------------------------
# Slice masks
# ---------------------------------------------------------------------
COVID_START = pd.Timestamp("2020-02-19")
COVID_END = pd.Timestamp("2020-04-30")


def slice_test(X_test: pd.DataFrame, slice_name: str) -> pd.DataFrame:
    if slice_name == "full_incl_covid":
        return X_test
    if slice_name == "full_excl_covid":
        mask = (X_test.index < COVID_START) | (X_test.index > COVID_END)
        return X_test.loc[mask]
    if slice_name == "covid_only":
        mask = (X_test.index >= COVID_START) & (X_test.index <= COVID_END)
        return X_test.loc[mask]
    raise ValueError(f"Unknown slice {slice_name!r}")


# ---------------------------------------------------------------------
# Universe loaders
# ---------------------------------------------------------------------
def load_universe_data(universe: str, slice_name: str, train_frac: float) -> dict:
    """Load returns for a universe, do split + standardize, slice the test set.

    Returns dict with X_train, X_test arrays and a few index helpers.
    """
    returns = pd.read_csv(
        PROCESSED_DIR / "returns_matrix.csv", index_col=0, parse_dates=True
    )
    ranked = pd.read_csv(PROCESSED_DIR / "ranked_universe.csv")

    if universe == "all_survivors":
        tickers = ranked["ticker"].tolist()
    elif universe.startswith("top_"):
        n = int(universe.split("_")[1])
        tickers = ranked.head(n)["ticker"].tolist()
    else:
        raise ValueError(f"Unknown universe {universe!r}")

    X = returns[tickers].dropna(how="all")
    X_train, X_test = pp.chronological_train_test_split(X, train_frac=train_frac)
    X_train_z, X_test_z, mu, sigma = pp.standardize_with_train_stats(X_train, X_test)
    X_test_z_sliced = slice_test(X_test_z, slice_name)

    return {
        "X_train": X_train_z.values,
        "X_test": X_test_z_sliced.values,
        "train_dates": X_train_z.index,
        "test_dates": X_test_z_sliced.index,
        "mu": mu.values,
        "sigma": sigma.values,
        "tickers": tickers,
    }


# ---------------------------------------------------------------------
# Subsets
# ---------------------------------------------------------------------
def grid_smoke() -> list[ExperimentConfig]:
    return generate_experiment_grid(
        universes=["top_100"],
        slices=["full_incl_covid"],
        methods=["raw", "pca", "dense_jl", "sparse_jl"],
        k_values=[20],
        s_values=[3],
        seeds=range(3),
    )


def grid_core() -> list[ExperimentConfig]:
    return generate_experiment_grid(
        universes=["top_100"],
        slices=["full_incl_covid"],
        methods=["raw", "pca", "dense_jl", "sparse_jl"],
        k_values=[5, 10, 20, 30, 50, 100],
        s_values=[1, 3, 5],
        seeds=range(10),
    )


def grid_top100_full() -> list[ExperimentConfig]:
    """Comprehensive top_100 grid: 50 seeds, all slices (covid_only auto-skipped)."""
    return generate_experiment_grid(
        universes=["top_100"],
        slices=["full_incl_covid", "full_excl_covid", "covid_only"],
        methods=["raw", "pca", "dense_jl", "sparse_jl"],
        k_values=[5, 10, 20, 30, 50, 100],
        s_values=[1, 3, 5],
        seeds=range(50),
    )


def grid_full() -> list[ExperimentConfig]:
    return generate_experiment_grid(
        universes=["top_100", "top_200", "all_survivors"],
        slices=["full_incl_covid", "full_excl_covid", "covid_only"],
        methods=["raw", "pca", "dense_jl", "sparse_jl"],
        k_values=[5, 10, 20, 30, 50, 100],
        s_values=[1, 3, 5],
        seeds=range(50),
    )


SUBSETS = {
    "smoke": grid_smoke,
    "core": grid_core,
    "top100_full": grid_top100_full,
    "full": grid_full,
}


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=["local", "ray"], default="local")
    parser.add_argument("--subset", choices=list(SUBSETS), default="smoke")
    parser.add_argument("--num-cpus", type=int, default=None)
    parser.add_argument("--no-save-artifacts", action="store_true")
    args = parser.parse_args()

    save_artifacts = not args.no_save_artifacts

    configs = SUBSETS[args.subset]()
    print(f"Subset: {args.subset}, total configs: {len(configs)}")

    # Pre-load all needed (universe, slice) pairs.
    needed_pairs = sorted({(c.universe, c.slice) for c in configs})
    prepared_data: dict[str, dict] = {}
    skipped_pairs: list[tuple[str, str]] = []
    print(f"Preparing data for {len(needed_pairs)} (universe, slice) pairs...")
    for universe, slice_ in needed_pairs:
        key = f"{universe}__{slice_}"
        d = load_universe_data(universe, slice_, train_frac=configs[0].train_frac)
        # Skip degenerate slices (e.g. covid_only with the standard 70/30 split
        # has 0 test rows because COVID lives in train). Per proposal §5.5
        # the proper fix is a pre-COVID training protocol; tracked as Phase 1.5.
        if d["X_test"].shape[0] < 10:
            print(f"  WARN: {universe}/{slice_} has {d['X_test'].shape[0]} test rows -- skipping")
            skipped_pairs.append((universe, slice_))
            continue
        prepared_data[key] = d
    if skipped_pairs:
        configs = [c for c in configs if (c.universe, c.slice) not in skipped_pairs]
        print(f"  Skipped configs after filter: {len(configs)} remaining")

    # Re-key configs by (universe, slice) so the runner finds them
    # (the runner's prepared_data is keyed by universe; we use a composite key).
    # Simpler: store under the universe key alone; the slice has been baked into
    # the test-set already, so each (universe, slice) is its own pseudo-universe
    # for the purposes of the runner.
    runner_data = {}
    new_configs: list[ExperimentConfig] = []
    for c in configs:
        pseudo_universe = f"{c.universe}__{c.slice}"
        runner_data[pseudo_universe] = prepared_data[pseudo_universe]
        new_configs.append(
            ExperimentConfig(
                universe=pseudo_universe,
                slice=c.slice,
                method=c.method,
                k=c.k,
                s=c.s,
                seed=c.seed,
                train_frac=c.train_frac,
                n_pairs=c.n_pairs,
            )
        )

    print(f"Running {len(new_configs)} configs on backend={args.backend}...")
    t0 = time.perf_counter()
    df = run_experiment_grid(
        configs=new_configs,
        prepared_data=runner_data,
        backend=args.backend,
        num_cpus=args.num_cpus,
        save_artifacts=save_artifacts,
        artifact_base=ARTIFACT_BASE,
    )
    elapsed = time.perf_counter() - t0

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_csv = RESULTS_DIR / f"phase1_{args.subset}_{args.backend}.csv"
    df.sort_values(["universe", "slice", "method", "k", "s", "seed"], inplace=True, na_position="first")
    df.to_csv(out_csv, index=False)

    print()
    print(f"DONE  configs={len(new_configs)}  elapsed={elapsed:.1f}s")
    print(f"      results -> {out_csv}")
    print()
    # Quick aggregate across seeds for randomized methods
    agg = (
        df.groupby(["method", "k", "s"], dropna=False)
        .agg(
            mean_dd_mean_abs=("dd_mean_abs", "mean"),
            std_dd_mean_abs=("dd_mean_abs", "std"),
            mean_dd_mean_rho=("dd_mean_rho", "mean"),
            std_dd_mean_rho=("dd_mean_rho", "std"),
            n=("seed", "size"),
        )
        .reset_index()
    )
    print("Aggregate (mean ± std across seeds):")
    print(agg.to_string(index=False, float_format=lambda x: f"{x:.4f}"))


if __name__ == "__main__":
    main()
