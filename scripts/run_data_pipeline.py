"""Phase 0 data pipeline: load Kaggle data, run elimination filters, save outputs.

Outputs (under data/processed/):
    candidate_universe.csv       -- input ticker list
    ranked_universe.csv          -- tickers sorted by avg dollar volume
    returns_matrix.csv           -- wide log-return matrix (all candidates)
    asset_list.csv               -- final selected tickers per slice
    preprocessing_report.json    -- diagnostic counts and dropped tickers

Run from project root:
    python scripts/run_data_pipeline.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Add project root to path so `import src.*` works when run as a script
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src import data_loader as dl
from src import preprocessing as pp


# ---------------------------------------------------------------------
# Configuration (matches AMLDS_Project_proposal-5 §2 frozen design)
# ---------------------------------------------------------------------
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
WINDOW_START = "2014-01-01"
WINDOW_END = "2024-12-31"   # effective end is data-dependent (Kaggle ends 2024-12-20)
COVERAGE_THRESHOLD = 0.95
# Phase 0 finding: 0.5 log-return threshold caught legitimate volatility events
# (COVID crash, PCG bankruptcy, etc.), not data errors. Raised to 1.0 (172% move,
# physically implausible without a data error). This amendment to proposal §4.3
# Stage 1d will be reflected in the next .tex revision.
EXTREME_RETURN = 1.0
N_FLAT = 10
CONSTANT_PRICE = 5

DATA_SOURCE = "yfinance"  # "kaggle" or "yfinance" -- Kaggle was found to be broken
KAGGLE_DATASET = "andrewmvd/sp-500-stocks"
DATA_DOWNLOAD_DATE = "2025-04-27"


def main() -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    print(f"[1/6] Loading {DATA_SOURCE} data from {RAW_DIR}")
    if DATA_SOURCE == "yfinance":
        prices_full, volume_full = dl.load_yfinance_wide(RAW_DIR)
        print(f"      yfinance prices shape: {prices_full.shape}")
        print(f"      raw date range: {prices_full.index.min().date()} .. {prices_full.index.max().date()}")
    else:
        raw_long = dl.load_kaggle_long(RAW_DIR)
        print(f"      raw rows: {len(raw_long):,}")

    print(f"[2/6] Filtering to date window [{WINDOW_START}, {WINDOW_END}]")
    if DATA_SOURCE == "yfinance":
        mask = (prices_full.index >= WINDOW_START) & (prices_full.index <= WINDOW_END)
        prices = prices_full.loc[mask].copy()
        volume = volume_full.loc[mask].copy()
        actual_end = prices.index.max()
    else:
        windowed = dl.filter_date_range(raw_long, WINDOW_START, WINDOW_END)
        actual_end = windowed["Date"].max()
        prices = dl.pivot_wide(windowed, "Adj Close")
        volume = dl.pivot_wide(windowed, "Volume")

    print(f"      effective window end: {actual_end.date()}")

    print("[3/6] Wide-format prices ready")
    print(f"      prices shape: {prices.shape}  (T x N)")

    print("[4/6] Computing returns and dollar volume")
    returns = dl.compute_log_returns(prices)
    # Dollar volume aligned with returns rows (skip first day to match)
    dollar_volume = dl.compute_dollar_volume(prices, volume)

    candidate_universe = sorted(prices.columns.tolist())
    print(f"      candidate pool size: {len(candidate_universe)}")

    print("[5/6] Running elimination pipeline")
    print("      Stage 1: data integrity")
    s1 = pp.run_stage1(
        prices=prices,
        returns=returns,
        window_end=actual_end,
        coverage_threshold=COVERAGE_THRESHOLD,
        extreme_return_threshold=EXTREME_RETURN,
    )
    print(f"        nan_in_window:        {len(s1.nan_in_window):>4}")
    print(f"        delisted_before_end:  {len(s1.delisted_before_end):>4}")
    print(f"        low_coverage:         {len(s1.low_coverage):>4}")
    print(f"        extreme_returns:      {len(s1.extreme_returns):>4}")
    print(f"        Stage 1 total drops:  {len(s1.all_dropped()):>4}")

    after_s1 = [t for t in candidate_universe if t not in s1.all_dropped()]

    # Stage 2 only operates on tickers that passed Stage 1
    prices_s2 = prices[after_s1]
    returns_s2 = returns[after_s1]

    print("      Stage 2: anomaly filter")
    s2 = pp.run_stage2(
        prices=prices_s2,
        returns=returns_s2,
        n_flat=N_FLAT,
        constant_price_threshold=CONSTANT_PRICE,
    )
    print(f"        consecutive_zero_returns: {len(s2.consecutive_zero_returns):>4}")
    print(f"        constant_price:           {len(s2.constant_price):>4}")
    print(f"        Stage 2 total drops:      {len(s2.all_dropped()):>4}")

    survivors = [t for t in after_s1 if t not in s2.all_dropped()]
    print(f"      survivors: {len(survivors)}")

    print("[6/6] Ranking and selecting slices")
    rankings = pp.rank_by_dollar_volume(dollar_volume[survivors])
    top_100 = pp.select_top_n(rankings, survivors, 100)
    top_200 = pp.select_top_n(rankings, survivors, 200)
    all_survivors = pp.select_top_n(rankings, survivors, len(survivors))
    print(f"      top_100[:5]:  {top_100[:5]}")
    print(f"      top_200[:5]:  {top_200[:5]}")
    print(f"      all_survivors size: {len(all_survivors)}")

    # ---------------------------------------------------------------------
    # Persist outputs
    # ---------------------------------------------------------------------
    print()
    print("Saving outputs to data/processed/")

    pd.DataFrame({"ticker": candidate_universe}).to_csv(
        PROCESSED_DIR / "candidate_universe.csv", index=False
    )

    rankings_df = rankings.reset_index()
    rankings_df.columns = ["ticker", "avg_dollar_volume"]
    rankings_df["rank"] = np.arange(1, len(rankings_df) + 1)
    rankings_df.to_csv(PROCESSED_DIR / "ranked_universe.csv", index=False)

    returns.to_csv(PROCESSED_DIR / "returns_matrix.csv")

    pd.DataFrame(
        {
            "ticker": top_200,  # default published list = top_200
            "rank": np.arange(1, len(top_200) + 1),
        }
    ).to_csv(PROCESSED_DIR / "asset_list.csv", index=False)

    report = {
        "data_source": DATA_SOURCE,
        "kaggle_dataset": KAGGLE_DATASET,
        "data_download_date": DATA_DOWNLOAD_DATE,
        "window_requested": {"start": WINDOW_START, "end": WINDOW_END},
        "window_effective_end": str(actual_end.date()),
        "trading_days": len(prices.index),
        "candidate_pool_size": len(candidate_universe),
        "stage1": s1.to_dict(),
        "stage2": s2.to_dict(),
        "survivors_count": len(survivors),
        "top_100_size": len(top_100),
        "top_200_size": len(top_200),
        "all_survivors_size": len(all_survivors),
        "no_missing_in_final_returns": bool(
            returns[survivors].isna().sum().sum() == 0
        ),
        "config": {
            "coverage_threshold": COVERAGE_THRESHOLD,
            "extreme_return_threshold": EXTREME_RETURN,
            "n_flat": N_FLAT,
            "constant_price_threshold": CONSTANT_PRICE,
        },
    }

    with open(PROCESSED_DIR / "preprocessing_report.json", "w") as f:
        json.dump(report, f, indent=2)

    print()
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Candidate pool size:    {len(candidate_universe)}")
    print(f"Stage 1 drops:          {len(s1.all_dropped())}")
    print(f"Stage 2 drops:          {len(s2.all_dropped())}")
    print(f"Total survivors:        {len(survivors)}")
    print(f"top_100 universe size:  {len(top_100)}")
    print(f"top_200 universe size:  {len(top_200)}")
    print(f"all_survivors size:     {len(all_survivors)}")
    print(f"Effective window end:   {actual_end.date()}")
    print(f"Trading days in window: {len(prices.index)}")
    print()
    print("Outputs written to data/processed/:")
    for p in sorted(PROCESSED_DIR.glob("*")):
        print(f"  {p.name}")


if __name__ == "__main__":
    main()
