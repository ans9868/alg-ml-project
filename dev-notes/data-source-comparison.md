# Data Source: yfinance vs Kaggle

This note documents why the project pivoted from the pinned Kaggle dataset to yfinance during Phase 0, and what the trade-offs were.

## TL;DR

The proposal pinned `andrewmvd/sp-500-stocks` (a community-maintained Kaggle dataset). On download and inspection, **the dataset has zero valid rows for ~127 large-cap tickers including AAPL, GOOGL, JPM, XOM, JNJ, KO, IBM, BAC, HD, BRK-B, WMT.** This is not survivorship bias; it is the dataset itself being broken for those tickers. The project pivoted to yfinance, which the proposal already listed as the fallback in §4.2.

## What the two sources actually are

A common misconception: yfinance and Kaggle are alternative *sources*. They are not. The Kaggle dataset is itself produced by a scraper that calls yfinance and saves the result to CSV. They have the same upstream (Yahoo Finance). The Kaggle dataset is just a cached snapshot.

```
                 ┌─────────────────┐
                 │  Yahoo Finance  │
                 │   (live API)    │
                 └────────┬────────┘
                          │
                ┌─────────┴──────────┐
                │                    │
     yfinance Python          Kaggle uploader's
       package call            update script
       (us, live)              (cached, ours via download)
```

## Side-by-side comparison

| Dimension | yfinance (Python package) | Kaggle dataset `andrewmvd/sp-500-stocks` |
|---|---|---|
| **Source of truth** | Yahoo Finance API (live calls) | Frozen CSV snapshot, last updated 2024-12-23 |
| **What's underneath** | Direct API calls | A scraper that calls yfinance, caches output |
| **Authentication** | None — `pip install yfinance` and go | Kaggle account + API token in `~/.kaggle/kaggle.json` |
| **Download size** | ~100 MB streaming over many API calls | One 19 MB zip → 96 MB unzipped |
| **Download time** | ~5 min for S&P 500 (rate-limited, batched) | ~5 sec |
| **Schema** | MultiIndex DataFrame `(field, ticker)` | Long-format CSV: `(Date, Symbol, Adj Close, Close, …, Volume)` |
| **Date range** | Anything you ask for | Fixed: 2010-01-04 → 2024-12-20 |
| **Coverage in our window** | 491/502 tickers | 160/502 with full data, **127 large-caps with zero data** |
| **Reproducibility** | Low — Yahoo retroactively re-adjusts splits/dividends | High in principle (frozen file), but only if the cached file is correct |
| **Update freshness** | Real-time | Periodic, but with silent failures |
| **Quality control** | None — Yahoo is authoritative | None — uploader's script either succeeds or quietly produces NaN |
| **Cost** | Free | Free |
| **License** | Yahoo Finance ToS (research use generally OK) | CC0 Public Domain |

## What "broken" looks like specifically

After downloading the Kaggle dataset and pivoting to wide format, we found:

```
Total tickers: 502
  with 0 NaN in 2014-2024:    160
  with 1-10 NaN in window:      0
  with >10 NaN in window:     342
```

The discontinuity is suspicious — there are no tickers with "a few" NaN values; either everything is present or everything is missing.

Investigating specific tickers:

| Ticker | Market Cap | Status in Kaggle CSV |
|---|---|---|
| AAPL | $3.8T | **0 valid rows** in entire 2010–2024 history |
| GOOGL | $2.4T | **0 valid rows** |
| JPM | $668B | **0 valid rows** |
| XOM | $465B | **0 valid rows** |
| JNJ | $347B | **0 valid rows** |
| KO | $269B | **0 valid rows** |
| IBM | $206B | **0 valid rows** |
| MSFT | $3.0T | full coverage ✓ |
| TSLA | $1.0T | full coverage ✓ |
| NVDA | $3.5T | full coverage ✓ |

127 of the dropped tickers have market cap > $50B. These are the largest, oldest, most-continuously-traded stocks on the planet. There is no plausible reason for them to be entirely absent from a "S&P 500 historical prices" dataset.

## Probable cause

The Kaggle uploader's pipeline almost certainly works like:

```python
for ticker in sp500_list:
    data = yf.download(ticker, start=START)
    df = pd.DataFrame(data)
    df.to_csv(f"data/{ticker}.csv")
```

When `yf.download` fails (rate-limited, transient HTTP error, ticker symbol issue) it returns an empty DataFrame, which then gets written as empty rows to CSV. There is no "if the result is empty, retry or alert" logic. After enough updates, accumulated failures pile up.

Why MSFT/TSLA/NVDA are fine and AAPL is not is presumably random — they all hit yfinance at different moments and some calls happened to succeed.

## What we did instead

The proposal already named yfinance as the fallback in §4.2:

> If the Kaggle dataset becomes unavailable or fails validation, the data pipeline must support a yfinance-based fallback that reproduces the same schema.

So the pivot was within the existing design envelope:

1. Kept the Kaggle company metadata (`sp500_companies.csv`) — that file *is* correct, it's only the price data that's broken.
2. Wrote `scripts/download_yfinance.py` — batched downloads of 50 tickers at a time, 1-second sleep between batches.
3. Saved `data/raw/yfinance_prices.csv` and `data/raw/yfinance_volume.csv` (wide format).
4. Updated `src/data_loader.py` with a `load_yfinance_wide()` function.
5. Updated `scripts/run_data_pipeline.py` to consume yfinance output.

## What yfinance gave us

```
Tickers downloaded: 491/502
Tickers failed:     11 (recent M&A or ticker changes — see below)
Date range:         2014-01-02 → 2024-12-31  (full year, vs Kaggle's Dec 20)
Trading days:       2768
```

Failed tickers (yfinance returned "possibly delisted; no timezone found" or similar):

| Ticker | Likely reason |
|---|---|
| ANSS | Ansys; merger with Synopsys pending/closed |
| DAY | Dayforce; recent IPO/rename |
| DFS | Discover Financial; merging with Capital One |
| FI | Fiserv; could be a transient yfinance error |
| HES | Hess; acquired by Chevron |
| IPG | Interpublic Group; could be a transient yfinance error |
| JNPR | Juniper Networks; HPE acquisition pending |
| K | Kellanova (formerly Kellogg's); ticker change |
| MMC | Marsh McLennan; could be a transient yfinance error |
| PARA | Paramount; recent merger |
| WBA | Walgreens Boots Alliance; could be a transient yfinance error |

5–6 of these are plausibly transient (worth re-attempting in a separate session). The others are real M&A or ticker-symbol changes; for our research purposes they are correctly excluded.

## Best of both: what we get with the pivot

The hybrid approach we ended up with delivers what the proposal wanted (reproducibility) without the broken-Kaggle problem:

- **yfinance's coverage** — no broken AAPL.
- **Kaggle-style reproducibility** — we save the yfinance result as a versioned CSV in our own repo. The committed CSV is the frozen artifact.
- **No third-party dependency at re-run time** — once the CSV is committed, anyone with the repo can re-run the pipeline without internet.
- **`preprocessing_report.json` records the download date and survivor counts** — full audit trail.

The only "live" step is the initial yfinance download. Any reviewer can re-run that step to cross-validate against our committed CSVs.

## Implications for the proposal

The frozen `.tex` (proposal v5) names Kaggle as primary and yfinance as fallback. The next revision (v6) should swap these:

- Make yfinance the primary documented source.
- Keep the Kaggle company-metadata file as the ticker-list driver.
- Document this finding in the limitations section so reviewers understand why we chose the more reproducibility-fragile source.

This is tracked as an open item in [`LAPTOP_PLAN.md`](../LAPTOP_PLAN.md).

## Lesson for future projects

When you depend on a community-maintained data source, **validate before you trust**. The Kaggle dataset description, download count (55k+), upvote count (502), and "daily updated" badge all looked legitimate. The actual quality issue only surfaced when we cross-checked specific tickers against well-known facts (AAPL has been traded for 40+ years; it cannot be missing from an S&P 500 dataset).

The validation cross-check we already had in the proposal (§4.2: "compare 5–10 random tickers against yfinance for a handful of dates") would have caught this in seconds. The cost of running it on day one is tiny; the cost of skipping it is wasted compute and rework.
