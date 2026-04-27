"""Loads the pinned Kaggle S&P 500 dataset and pivots to wide-format matrices.

Source: kaggle datasets download andrewmvd/sp-500-stocks
File schema (long format):
    Date, Symbol, Adj Close, Close, High, Low, Open, Volume

This module produces wide-format matrices (rows = trading days, cols = tickers).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def load_kaggle_long(raw_dir: Path) -> pd.DataFrame:
    """Read sp500_stocks.csv as long-format with parsed dates."""
    path = Path(raw_dir) / "sp500_stocks.csv"
    df = pd.read_csv(path, parse_dates=["Date"])
    return df


def filter_date_range(df: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    """Keep rows with Date in [start, end] inclusive."""
    mask = (df["Date"] >= pd.Timestamp(start)) & (df["Date"] <= pd.Timestamp(end))
    return df.loc[mask].copy()


def pivot_wide(df: pd.DataFrame, value_col: str) -> pd.DataFrame:
    """Pivot long-format to wide: rows = Date, cols = Symbol."""
    return df.pivot(index="Date", columns="Symbol", values=value_col).sort_index()


def compute_log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Daily log returns. First row is dropped (NaN by construction)."""
    returns = np.log(prices / prices.shift(1))
    return returns.iloc[1:]


def compute_dollar_volume(prices: pd.DataFrame, volume: pd.DataFrame) -> pd.DataFrame:
    """Daily dollar volume = adjusted close * share volume."""
    return prices * volume


def load_company_metadata(raw_dir: Path) -> pd.DataFrame:
    """Read sp500_companies.csv (sector, market cap, weight, etc.)."""
    path = Path(raw_dir) / "sp500_companies.csv"
    return pd.read_csv(path)


# ---------------------------------------------------------------------
# yfinance loader (fallback per proposal §4.2; primary source for Phase 0
# because the Kaggle price data was found to be broken for many large
# tickers including AAPL, GOOGL, JPM, etc.)
# ---------------------------------------------------------------------
def load_yfinance_wide(raw_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load yfinance wide-format CSVs (rows = Date, cols = Symbol).

    Returns (prices, volume).
    """
    raw = Path(raw_dir)
    prices = pd.read_csv(raw / "yfinance_prices.csv", index_col=0, parse_dates=True)
    volume = pd.read_csv(raw / "yfinance_volume.csv", index_col=0, parse_dates=True)
    return prices.sort_index(), volume.sort_index()
