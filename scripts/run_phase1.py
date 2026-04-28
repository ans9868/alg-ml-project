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
# Time windows
# ---------------------------------------------------------------------
COVID_START = pd.Timestamp("2020-02-19")
COVID_END = pd.Timestamp("2020-04-30")
PRECOVID_TRAIN_END = pd.Timestamp("2019-12-31")


def slice_test_chrono(X_test: pd.DataFrame, slice_name: str) -> pd.DataFrame:
    """Slice the test set under the chrono70_30 protocol.
    Note: with the standard 70/30 split, the test starts ~2021-09; covid_only
    is empty (skipped upstream) and full_incl/excl are equivalent.
    """
    if slice_name == "full":
        return X_test
    if slice_name == "full_incl_covid":
        return X_test
    if slice_name == "full_excl_covid":
        mask = (X_test.index < COVID_START) | (X_test.index > COVID_END)
        return X_test.loc[mask]
    if slice_name == "covid_only":
        mask = (X_test.index >= COVID_START) & (X_test.index <= COVID_END)
        return X_test.loc[mask]
    raise ValueError(f"Unknown chrono slice {slice_name!r}")


def slice_test_precovid(X_test: pd.DataFrame, slice_name: str) -> pd.DataFrame:
    """Slice the test set under the preCOVID protocol.
    Test starts 2020-01-01 (after train end of 2019-12-31).
    """
    if slice_name == "all_test":
        return X_test
    if slice_name == "covid_only":
        mask = (X_test.index >= COVID_START) & (X_test.index <= COVID_END)
        return X_test.loc[mask]
    if slice_name == "post_covid":
        mask = X_test.index > COVID_END
        return X_test.loc[mask]
    raise ValueError(f"Unknown preCOVID slice {slice_name!r}")


# ---------------------------------------------------------------------
# Universe loaders
# ---------------------------------------------------------------------
def load_universe_data(
    universe: str,
    protocol: str,
    slice_name: str,
    train_frac: float = 0.7,
) -> dict:
    """Load returns for a universe under a given training protocol, then slice
    the test set per the slice name. Standardization is fit on the train slice
    that the protocol defines.
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

    if protocol == "chrono70_30":
        X_train, X_test = pp.chronological_train_test_split(X, train_frac=train_frac)
        X_train_z, X_test_z, mu, sigma = pp.standardize_with_train_stats(X_train, X_test)
        X_test_z_sliced = slice_test_chrono(X_test_z, slice_name)
    elif protocol == "preCOVID":
        # Train: 2014-01-01 to 2019-12-31 (~6 years)
        # Test:  2020-01-01 to 2024-12-31 (~5 years incl. COVID + post-COVID)
        train_mask = X.index <= PRECOVID_TRAIN_END
        X_train = X.loc[train_mask].copy()
        X_test = X.loc[~train_mask].copy()
        X_train_z, X_test_z, mu, sigma = pp.standardize_with_train_stats(X_train, X_test)
        X_test_z_sliced = slice_test_precovid(X_test_z, slice_name)
    else:
        raise ValueError(f"Unknown protocol {protocol!r}")

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
CHRONO = ("chrono70_30", ["full"])
PRECOVID = ("preCOVID", ["covid_only", "post_covid", "all_test"])


def grid_smoke() -> list[ExperimentConfig]:
    return generate_experiment_grid(
        universes=["top_100"],
        protocol_slices=[("chrono70_30", ["full"])],
        methods=["raw", "pca", "dense_jl", "sparse_jl"],
        k_values=[20],
        s_values=[3],
        seeds=range(3),
    )


def grid_core() -> list[ExperimentConfig]:
    return generate_experiment_grid(
        universes=["top_100"],
        protocol_slices=[CHRONO],
        methods=["raw", "pca", "dense_jl", "sparse_jl"],
        k_values=[5, 10, 20, 30, 50, 100],
        s_values=[1, 3, 5],
        seeds=range(10),
    )


def grid_top100_full() -> list[ExperimentConfig]:
    """Comprehensive top_100 grid: both protocols, all slices, 50 seeds."""
    return generate_experiment_grid(
        universes=["top_100"],
        protocol_slices=[CHRONO, PRECOVID],
        methods=["raw", "pca", "dense_jl", "sparse_jl"],
        k_values=[5, 10, 20, 30, 50, 100],
        s_values=[1, 3, 5],
        seeds=range(50),
    )


def grid_path_b() -> list[ExperimentConfig]:
    """CrashSketch evaluation grid: top_100, k=20, both protocols, all slices, 20 seeds.
    Compares all CrashSketch variants (R/noR x float/int8) against raw, PCA,
    dense JL, sparse JL.

    The interesting comparisons:
      - crashsketch_R_int8 vs crashsketch_noR_int8: does rotation help when paired
        with quantization? (The PolarQuant claim)
      - crashsketch_*_int8 vs sparse_jl: does adding quantization hurt much?
    """
    return generate_experiment_grid(
        universes=["top_100"],
        protocol_slices=[CHRONO, PRECOVID],
        methods=["raw", "pca", "dense_jl", "sparse_jl",
                 "crashsketch_R", "crashsketch_noR",
                 "crashsketch_R_int8", "crashsketch_noR_int8",
                 "crashsketch_R_1bit", "crashsketch_noR_1bit",
                 "crashsketch_R_normsign", "crashsketch_noR_normsign"],
        k_values=[20],
        s_values=[3],
        seeds=range(20),
    )


def grid_yolo() -> list[ExperimentConfig]:
    """The whole shebang. 5 universes (top_100/200/300/400/MAX),
    both protocols, all slices, 100 seeds. ~50K configs.
    """
    return generate_experiment_grid(
        universes=["top_100", "top_200", "top_300", "top_400", "all_survivors"],
        protocol_slices=[CHRONO, PRECOVID],
        methods=["raw", "pca", "dense_jl", "sparse_jl"],
        k_values=[5, 10, 20, 30, 50, 100],
        s_values=[1, 3, 5],
        seeds=range(100),
    )


def grid_full() -> list[ExperimentConfig]:
    """Alias for yolo (backwards compat)."""
    return grid_yolo()


SUBSETS = {
    "smoke": grid_smoke,
    "core": grid_core,
    "top100_full": grid_top100_full,
    "path_b": grid_path_b,
    "yolo": grid_yolo,
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

    # Pre-load all needed (universe, protocol, slice) triples.
    needed = sorted({(c.universe, c.protocol, c.slice) for c in configs})
    prepared_data: dict[str, dict] = {}
    skipped_triples: list[tuple[str, str, str]] = []
    print(f"Preparing data for {len(needed)} (universe, protocol, slice) triples...")
    for universe, protocol, slice_ in needed:
        key = f"{universe}__{protocol}__{slice_}"
        d = load_universe_data(universe, protocol, slice_, train_frac=configs[0].train_frac)
        # Skip degenerate slices: with chrono70_30, covid_only ends up with 0
        # rows because COVID is in train.
        if d["X_test"].shape[0] < 10:
            print(f"  WARN: {key} has {d['X_test'].shape[0]} test rows -- skipping")
            skipped_triples.append((universe, protocol, slice_))
            continue
        prepared_data[key] = d
    if skipped_triples:
        configs = [c for c in configs if (c.universe, c.protocol, c.slice) not in skipped_triples]
        print(f"  Skipped configs after filter: {len(configs)} remaining")

    # The runner keys prepared_data by "universe". We bake (protocol, slice) into
    # a composite pseudo-universe so each config has the correct sliced test set.
    runner_data = {}
    new_configs: list[ExperimentConfig] = []
    for c in configs:
        pseudo_universe = f"{c.universe}__{c.protocol}__{c.slice}"
        runner_data[pseudo_universe] = prepared_data[pseudo_universe]
        new_configs.append(
            ExperimentConfig(
                universe=pseudo_universe,
                protocol=c.protocol,
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
    df.sort_values(["universe", "protocol", "slice", "method", "k", "s", "seed"], inplace=True, na_position="first")
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
