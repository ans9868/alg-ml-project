"""Three-layer stock selection framework per AMLDS_Project_proposal-5 §4.3.

Layer 1: candidate pool (input list of tickers)
Layer 2: ranking metric (avg daily dollar volume)
Layer 3: elimination pipeline (Stage 1 data integrity, Stage 2 anomaly)
         then top_N selection.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------
# Stage 1: data integrity
# ---------------------------------------------------------------------
@dataclass
class Stage1Drops:
    nan_in_window: list[str] = field(default_factory=list)
    delisted_before_end: list[str] = field(default_factory=list)
    low_coverage: list[str] = field(default_factory=list)
    extreme_returns: list[str] = field(default_factory=list)

    def all_dropped(self) -> set[str]:
        return (
            set(self.nan_in_window)
            | set(self.delisted_before_end)
            | set(self.low_coverage)
            | set(self.extreme_returns)
        )

    def to_dict(self) -> dict:
        return {
            "nan_in_window": sorted(self.nan_in_window),
            "delisted_before_end": sorted(self.delisted_before_end),
            "low_coverage": sorted(self.low_coverage),
            "extreme_returns": sorted(self.extreme_returns),
            "total_dropped": len(self.all_dropped()),
        }


def run_stage1(
    prices: pd.DataFrame,
    returns: pd.DataFrame,
    window_end: pd.Timestamp,
    coverage_threshold: float = 0.95,
    extreme_return_threshold: float = 0.5,
) -> Stage1Drops:
    """Apply Stage 1 data integrity filter to a wide adjusted-close frame.

    Drops a ticker if:
      - any NaN in adjusted close inside the window
      - delisted before the window end (last non-NaN date < window_end)
      - fewer than coverage_threshold of expected trading days are present
      - any single-day |log return| > extreme_return_threshold (likely data error)
    """
    drops = Stage1Drops()
    expected_days = len(prices.index)

    for ticker in prices.columns:
        col = prices[ticker]
        nan_count = col.isna().sum()
        coverage = (expected_days - nan_count) / expected_days

        # Find last non-NaN date
        non_nan = col.dropna()
        if len(non_nan) == 0:
            drops.nan_in_window.append(ticker)
            continue
        last_date = non_nan.index[-1]

        # 1a: NaN anywhere in window
        if nan_count > 0:
            drops.nan_in_window.append(ticker)
            continue

        # 1b: delisted before window end (last_date is more than ~5 trading days before end)
        if last_date < window_end - pd.Timedelta(days=7):
            drops.delisted_before_end.append(ticker)
            continue

        # 1c: low coverage
        if coverage < coverage_threshold:
            drops.low_coverage.append(ticker)
            continue

        # 1d: extreme single-day returns (data errors)
        if ticker in returns.columns:
            r = returns[ticker]
            if (r.abs() > extreme_return_threshold).any():
                drops.extreme_returns.append(ticker)
                continue

    return drops


# ---------------------------------------------------------------------
# Stage 2: anomaly filter
# ---------------------------------------------------------------------
@dataclass
class Stage2Drops:
    consecutive_zero_returns: list[str] = field(default_factory=list)
    constant_price: list[str] = field(default_factory=list)

    def all_dropped(self) -> set[str]:
        return set(self.consecutive_zero_returns) | set(self.constant_price)

    def to_dict(self) -> dict:
        return {
            "consecutive_zero_returns": sorted(self.consecutive_zero_returns),
            "constant_price": sorted(self.constant_price),
            "total_dropped": len(self.all_dropped()),
        }


def _max_consecutive_true(s: pd.Series) -> int:
    """Length of the longest consecutive run of True in a boolean series."""
    if len(s) == 0:
        return 0
    # group consecutive runs
    groups = (s != s.shift()).cumsum()
    runs = s.groupby(groups).sum()
    return int(runs.max()) if len(runs) else 0


def run_stage2(
    prices: pd.DataFrame,
    returns: pd.DataFrame,
    n_flat: int = 10,
    constant_price_threshold: int = 5,
) -> Stage2Drops:
    """Apply Stage 2 anomaly filter.

    Drops a ticker if:
      - more than n_flat consecutive days of exactly zero return (halted/stale)
      - adjusted close exactly constant for more than constant_price_threshold
        consecutive days
    """
    drops = Stage2Drops()

    for ticker in prices.columns:
        # 2a: consecutive zero returns
        if ticker in returns.columns:
            r = returns[ticker]
            zero_mask = (r == 0)
            if _max_consecutive_true(zero_mask) > n_flat:
                drops.consecutive_zero_returns.append(ticker)
                continue

        # 2b: constant price stretches
        p = prices[ticker]
        constant_mask = (p == p.shift())
        if _max_consecutive_true(constant_mask) > constant_price_threshold:
            drops.constant_price.append(ticker)
            continue

    return drops


# ---------------------------------------------------------------------
# Layer 2: ranking
# ---------------------------------------------------------------------
def rank_by_dollar_volume(dollar_volume: pd.DataFrame) -> pd.Series:
    """Average daily dollar volume per ticker, descending."""
    avg = dollar_volume.mean(axis=0, skipna=True)
    return avg.sort_values(ascending=False)


# ---------------------------------------------------------------------
# Layer 3 Stage 3: selection
# ---------------------------------------------------------------------
def select_top_n(rankings: pd.Series, survivors: list[str], n: int) -> list[str]:
    """Return top n tickers from survivors, ordered by ranking."""
    survivor_rankings = rankings[rankings.index.isin(survivors)]
    return survivor_rankings.head(n).index.tolist()


# ---------------------------------------------------------------------
# Train/test split + standardization (proposal §4.7--4.8)
# ---------------------------------------------------------------------
def chronological_train_test_split(
    returns: pd.DataFrame, train_frac: float = 0.7
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Chronological split: first train_frac rows -> train, rest -> test.

    No randomization (financial time series have temporal structure).
    """
    n_train = int(len(returns) * train_frac)
    train = returns.iloc[:n_train].copy()
    test = returns.iloc[n_train:].copy()
    return train, test


def standardize_with_train_stats(
    train: pd.DataFrame, test: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Z-score per asset using train statistics only (no test leakage).

    Returns (train_z, test_z, mu, sigma).
    """
    mu = train.mean(axis=0)
    sigma = train.std(axis=0, ddof=1)
    # Guard against zero std (constant column would have been dropped at Stage 2)
    sigma = sigma.replace(0, np.nan)
    train_z = (train - mu) / sigma
    test_z = (test - mu) / sigma
    return train_z, test_z, mu, sigma
