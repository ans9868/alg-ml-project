"""Download S&P 500 daily prices and volume via yfinance.

Uses the ticker list from the Kaggle company-metadata file (which IS correct
even though the Kaggle price data is broken).

Outputs:
    data/raw/yfinance_prices.csv   -- wide adjusted-close matrix (Date x Symbol)
    data/raw/yfinance_volume.csv   -- wide volume matrix
    data/raw/yfinance_failed.txt   -- tickers yfinance failed to fetch
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import pandas as pd
import yfinance as yf

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"

START = "2014-01-01"
END = "2025-01-01"  # yfinance end is exclusive; this gives us through 2024-12-31

BATCH_SIZE = 50
SLEEP_BETWEEN_BATCHES = 1.0


def chunks(seq, n):
    for i in range(0, len(seq), n):
        yield seq[i : i + n]


def main() -> None:
    # Get ticker list from Kaggle's company metadata
    companies = pd.read_csv(RAW_DIR / "sp500_companies.csv")
    tickers = sorted(companies["Symbol"].tolist())
    print(f"Will download {len(tickers)} tickers from yfinance, {BATCH_SIZE} at a time")
    print(f"Date range: {START} to {END}")
    print()

    all_adj_close = {}
    all_volume = {}
    failed = []

    for i, batch in enumerate(chunks(tickers, BATCH_SIZE), 1):
        print(f"[batch {i}] downloading {len(batch)} tickers: {batch[0]} .. {batch[-1]}")
        try:
            data = yf.download(
                batch,
                start=START,
                end=END,
                auto_adjust=False,
                progress=False,
                threads=True,
            )
            if data.empty:
                print(f"  WARN: batch returned empty")
                failed.extend(batch)
                continue

            # When more than one ticker, columns are MultiIndex (field, ticker)
            if isinstance(data.columns, pd.MultiIndex):
                adj_close = data["Adj Close"]
                volume = data["Volume"]
            else:
                # single ticker case
                adj_close = data[["Adj Close"]].rename(columns={"Adj Close": batch[0]})
                volume = data[["Volume"]].rename(columns={"Volume": batch[0]})

            for t in batch:
                if t in adj_close.columns:
                    series = adj_close[t]
                    if series.notna().sum() > 0:
                        all_adj_close[t] = series
                        all_volume[t] = volume[t]
                    else:
                        failed.append(t)
                else:
                    failed.append(t)

        except Exception as e:
            print(f"  ERROR: {e}")
            failed.extend(batch)

        time.sleep(SLEEP_BETWEEN_BATCHES)

    print()
    print(f"Downloaded successfully: {len(all_adj_close)}")
    print(f"Failed: {len(failed)}")
    if failed:
        print(f"  failed tickers: {failed[:20]}{'...' if len(failed) > 20 else ''}")

    # Save wide-format CSVs
    prices_df = pd.DataFrame(all_adj_close)
    volume_df = pd.DataFrame(all_volume)

    # Align indices
    prices_df = prices_df.sort_index()
    volume_df = volume_df.sort_index()

    print(f"Prices shape:  {prices_df.shape}  (T x N)")
    print(f"Volume shape:  {volume_df.shape}")
    print(f"Date range:    {prices_df.index.min().date()} .. {prices_df.index.max().date()}")

    prices_df.to_csv(RAW_DIR / "yfinance_prices.csv")
    volume_df.to_csv(RAW_DIR / "yfinance_volume.csv")

    if failed:
        with open(RAW_DIR / "yfinance_failed.txt", "w") as f:
            for t in failed:
                f.write(f"{t}\n")

    print()
    print(f"Saved to {RAW_DIR}/")


if __name__ == "__main__":
    main()
