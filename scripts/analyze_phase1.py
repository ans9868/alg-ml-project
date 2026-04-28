"""Generate all headline figures and tables from a Phase 1 results CSV.

Reads results/phase1/cloud/phase1_yolo_ray.csv (or whichever CSV path
is passed) and writes:
  figures/yolo_*.png
  results/phase1/cloud/yolo_*_summary.csv

Usage:
    python scripts/analyze_phase1.py [path/to/results.csv]
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_CSV = PROJECT_ROOT / "results" / "phase1" / "cloud" / "phase1_yolo_ray.csv"
FIG_DIR = PROJECT_ROOT / "figures"
SUMMARY_DIR = PROJECT_ROOT / "results" / "phase1" / "cloud"

PALETTE = {
    "raw": "#444444",
    "pca": "#1f77b4",
    "dense_jl": "#ff7f0e",
    "sparse_jl": "#2ca02c",
}


# ---------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------
def load_results(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    # Parse the composite universe key into universe_real (e.g. "top_100")
    df["universe_real"] = df["universe"].str.split("__").str[0]
    df["N"] = df["universe_real"].map({
        "top_100": 100, "top_200": 200, "top_300": 300,
        "top_400": 400, "all_survivors": 452,
    })
    return df


# ---------------------------------------------------------------------
# Common helpers
# ---------------------------------------------------------------------
def _agg_with_seed_bands(df, x_col, metric):
    """Return per-method (x, mean, std) DataFrame, sparse_jl pinned to s=3."""
    rows = []
    for method, color in PALETTE.items():
        ms = df[df.method == method]
        if method == "sparse_jl":
            ms = ms[ms.s == 3]
        agg = ms.groupby(x_col)[metric].agg(["mean", "std", "count"]).reset_index()
        agg["method"] = method
        agg["color"] = color
        rows.append(agg)
    return pd.concat(rows, ignore_index=True)


def _plot_lines(ax, agg, x_col, ylabel, title, xlabel=None):
    for method, color in PALETTE.items():
        sub = agg[agg.method == method]
        if len(sub) == 0:
            continue
        ax.plot(sub[x_col], sub["mean"], "-o", color=color, label=method, linewidth=2)
        if (sub["count"] > 1).any():
            ax.fill_between(
                sub[x_col],
                sub["mean"] - sub["std"].fillna(0),
                sub["mean"] + sub["std"].fillna(0),
                alpha=0.18,
                color=color,
            )
    ax.set_xlabel(xlabel or x_col)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)


# ---------------------------------------------------------------------
# Plot 1: 4-metric panel vs k (top_100, chrono/full)
# ---------------------------------------------------------------------
def fig_metric_panel_vs_k(df, universe="top_100", protocol="chrono70_30", slice_="full"):
    sub = df[
        (df.universe_real == universe)
        & (df.protocol == protocol)
        & (df["slice"] == slice_)
    ]
    metrics = [
        ("dd_mean_abs", "Distance distortion |ρ - 1|", True),
        ("nn_overlap_at_5", "5-NN overlap", False),
        ("ari_mean", "Clustering ARI (mean over C∈{3,5,8})", False),
        ("anomaly_recall_5pct", "Anomaly recall @ top 5%", False),
    ]
    fig, axes = plt.subplots(1, 4, figsize=(20, 4.5))
    for ax, (metric, ylabel, lower_better) in zip(axes, metrics):
        agg = _agg_with_seed_bands(sub, "k", metric)
        _plot_lines(ax, agg, "k", ylabel, ylabel, xlabel="compressed dim k")
    fig.suptitle(
        f"Metric panel: {universe}, {protocol}/{slice_}  (mean ± std across seeds)",
        fontsize=13,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    return fig


# ---------------------------------------------------------------------
# Plot 2: COVID stress comparison (covid_only vs post_covid panels)
# ---------------------------------------------------------------------
def fig_covid_stress(df, metric, ylabel, universe="top_100"):
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), sharey=True)
    for ax, slice_ in zip(axes, ["covid_only", "post_covid"]):
        sub = df[
            (df.universe_real == universe)
            & (df.protocol == "preCOVID")
            & (df["slice"] == slice_)
        ]
        agg = _agg_with_seed_bands(sub, "k", metric)
        _plot_lines(ax, agg, "k", ylabel, f"{slice_}", xlabel="compressed dim k")
    fig.suptitle(
        f"COVID stress: {ylabel}  —  {universe}, preCOVID protocol",
        fontsize=12,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    return fig


# ---------------------------------------------------------------------
# Plot 3: Universe scaling — dd vs N at k=20
# ---------------------------------------------------------------------
def fig_universe_scaling(df, metric, ylabel, k=20, protocol="chrono70_30", slice_="full"):
    sub = df[
        (df.protocol == protocol)
        & (df["slice"] == slice_)
        & (df.k == k)
    ]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    agg = _agg_with_seed_bands(sub, "N", metric)
    _plot_lines(ax, agg, "N", ylabel, f"{ylabel} vs Universe size (k={k})", xlabel="N (assets)")
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------
# Plot 4: Runtime / accuracy frontier
# ---------------------------------------------------------------------
def fig_runtime_frontier(df, metric, ylabel, universe="top_100", protocol="chrono70_30", slice_="full"):
    sub = df[
        (df.universe_real == universe)
        & (df.protocol == protocol)
        & (df["slice"] == slice_)
    ].copy()
    sub["total_time"] = sub.fit_seconds + sub.transform_seconds

    fig, ax = plt.subplots(figsize=(8, 5))
    for method, color in PALETTE.items():
        ms = sub[sub.method == method]
        if method == "sparse_jl":
            ms = ms[ms.s == 3]
        if len(ms) == 0:
            continue
        ax.scatter(ms.total_time + 1e-6, ms[metric], color=color, alpha=0.25, s=12, label=None)
        agg = ms.groupby("k").agg(t=("total_time", "mean"), m=(metric, "mean")).reset_index()
        ax.plot(agg.t + 1e-6, agg.m, "-o", color=color, linewidth=2, label=method)
        for _, row in agg.iterrows():
            ax.annotate(f"k={int(row.k)}", (row.t + 1e-6, row.m),
                        fontsize=7, xytext=(3, 3), textcoords="offset points",
                        color=color)

    ax.set_xscale("log")
    ax.set_xlabel("fit + transform time (sec, log scale)")
    ax.set_ylabel(ylabel)
    ax.set_title(f"Runtime vs accuracy — {universe}, {protocol}/{slice_}")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------
# Plot 5: Sparse JL sparsity tradeoff
# ---------------------------------------------------------------------
def fig_sparsity_tradeoff(df, universe="top_100", protocol="chrono70_30", slice_="full"):
    sub = df[
        (df.universe_real == universe)
        & (df.protocol == protocol)
        & (df["slice"] == slice_)
        & (df.method == "sparse_jl")
    ]
    fig, ax = plt.subplots(figsize=(8, 5))
    cmap = plt.get_cmap("viridis")
    k_values = sorted(sub.k.unique())
    for i, k in enumerate(k_values):
        ms = sub[sub.k == k]
        agg = ms.groupby("s").agg(
            sparsity=("sparsity_ratio", "mean"),
            dd=("dd_mean_abs", "mean"),
            dd_std=("dd_mean_abs", "std"),
        ).reset_index()
        color = cmap(i / max(1, len(k_values) - 1))
        ax.errorbar(
            agg.sparsity,
            agg.dd,
            yerr=agg.dd_std,
            fmt="-o",
            color=color,
            label=f"k={k}",
            linewidth=2,
        )
        for _, row in agg.iterrows():
            ax.annotate(f"s={int(row.s)}", (row.sparsity, row.dd),
                        fontsize=7, xytext=(3, 3), textcoords="offset points",
                        color=color)
    ax.set_xlabel("Sparsity ratio (fraction of zeros in projection matrix)")
    ax.set_ylabel("Distance distortion |ρ - 1|")
    ax.set_title("Sparse JL sparsity-accuracy tradeoff (top_100)")
    ax.legend(title="dim k", fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------
# Summary tables
# ---------------------------------------------------------------------
def best_method_table(df, universe="top_100", protocol="chrono70_30", slice_="full", k=20):
    metrics_dir = {
        "dd_mean_abs": "min",
        "nn_overlap_at_5": "max",
        "nn_overlap_at_10": "max",
        "ari_mean": "max",
        "anomaly_recall_5pct": "max",
        "anomaly_score_spearman": "max",
        "sparsity_ratio": "max",
    }
    sub = df[
        (df.universe_real == universe)
        & (df.protocol == protocol)
        & (df["slice"] == slice_)
        & (df.k == k)
    ]
    sub = sub[(sub.method != "sparse_jl") | (sub.s == 3)]
    rows = []
    for metric, direction in metrics_dir.items():
        agg = sub.groupby("method")[metric].mean()
        winner = agg.idxmin() if direction == "min" else agg.idxmax()
        rows.append({
            "metric": metric,
            "direction": direction,
            "winner": winner,
            "winner_value": round(agg[winner], 4),
            "raw": round(agg.get("raw", float("nan")), 4),
            "pca": round(agg.get("pca", float("nan")), 4),
            "dense_jl": round(agg.get("dense_jl", float("nan")), 4),
            "sparse_jl_s3": round(agg.get("sparse_jl", float("nan")), 4),
        })
    return pd.DataFrame(rows)


def covid_stress_delta_table(df, universe="top_100", k=20):
    """For each method, compute (covid_only - post_covid) for each metric."""
    sub = df[
        (df.universe_real == universe)
        & (df.protocol == "preCOVID")
        & (df.k == k)
        & ((df.method != "sparse_jl") | (df.s == 3))
    ]
    metrics = ["dd_mean_abs", "nn_overlap_at_5", "ari_mean", "anomaly_recall_5pct"]
    pivot = sub.groupby(["slice", "method"])[metrics].mean().unstack("slice")
    delta = pivot.xs("covid_only", level="slice", axis=1) - pivot.xs("post_covid", level="slice", axis=1)
    return delta.round(4)


def universe_scaling_table(df, k=20, protocol="chrono70_30", slice_="full"):
    sub = df[(df.protocol == protocol) & (df["slice"] == slice_) & (df.k == k)]
    sub = sub[(sub.method != "sparse_jl") | (sub.s == 3)]
    return (
        sub.groupby(["N", "method"])["dd_mean_abs"]
        .mean()
        .unstack("method")
        .round(4)
    )


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def main(csv_path: Path = DEFAULT_CSV) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    SUMMARY_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading {csv_path}")
    df = load_results(csv_path)
    print(f"  rows: {len(df):,}")

    print("\nGenerating plots...")

    figs = [
        ("yolo_metric_panel_top100_chrono.png",
         fig_metric_panel_vs_k(df, "top_100", "chrono70_30", "full")),
        ("yolo_metric_panel_all_survivors_chrono.png",
         fig_metric_panel_vs_k(df, "all_survivors", "chrono70_30", "full")),

        ("yolo_covid_dd.png",
         fig_covid_stress(df, "dd_mean_abs", "Distance distortion |ρ - 1|")),
        ("yolo_covid_anomaly_recall.png",
         fig_covid_stress(df, "anomaly_recall_5pct", "Anomaly recall @ top 5%")),
        ("yolo_covid_nn5.png",
         fig_covid_stress(df, "nn_overlap_at_5", "5-NN overlap")),
        ("yolo_covid_ari.png",
         fig_covid_stress(df, "ari_mean", "Clustering ARI")),

        ("yolo_universe_scaling_dd.png",
         fig_universe_scaling(df, "dd_mean_abs", "Distance distortion |ρ - 1|")),
        ("yolo_universe_scaling_nn5.png",
         fig_universe_scaling(df, "nn_overlap_at_5", "5-NN overlap")),
        ("yolo_universe_scaling_ari.png",
         fig_universe_scaling(df, "ari_mean", "Clustering ARI")),

        ("yolo_runtime_frontier_dd.png",
         fig_runtime_frontier(df, "dd_mean_abs", "Distance distortion |ρ - 1|")),
        ("yolo_runtime_frontier_anomaly.png",
         fig_runtime_frontier(df, "anomaly_recall_5pct", "Anomaly recall")),

        ("yolo_sparsity_tradeoff.png",
         fig_sparsity_tradeoff(df)),
    ]

    for filename, fig in figs:
        path = FIG_DIR / filename
        fig.savefig(path, dpi=140, bbox_inches="tight")
        print(f"  wrote {path.relative_to(PROJECT_ROOT)}")
        plt.close(fig)

    print("\nGenerating summary tables...")

    best = best_method_table(df)
    best.to_csv(SUMMARY_DIR / "yolo_best_method_top100_k20.csv", index=False)
    print(f"  wrote yolo_best_method_top100_k20.csv")

    covid = covid_stress_delta_table(df)
    covid.to_csv(SUMMARY_DIR / "yolo_covid_stress_delta_top100_k20.csv")
    print(f"  wrote yolo_covid_stress_delta_top100_k20.csv")

    scale = universe_scaling_table(df)
    scale.to_csv(SUMMARY_DIR / "yolo_universe_scaling_dd_k20.csv")
    print(f"  wrote yolo_universe_scaling_dd_k20.csv")

    print("\n=== Best-method table (top_100, chrono/full, k=20) ===")
    print(best.to_string(index=False))

    print("\n=== COVID stress delta (top_100, k=20, covid_only minus post_covid) ===")
    print(covid)

    print("\n=== Universe scaling (dd_mean_abs at k=20) ===")
    print(scale)


if __name__ == "__main__":
    csv = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_CSV
    main(csv)
