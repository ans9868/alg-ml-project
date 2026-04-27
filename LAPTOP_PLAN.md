# Phase 0 — Laptop Smoke Test

Goal: validate the data pipeline and one method end-to-end on a tiny universe **before** going to GCP. If anything surprises us here, we fix it locally.

Environment: `dev-env` venv (`alias dev-env='source ~/dev-env/bin/activate'`), Python 3.10.18.

Working dir: `~/projects/alg-ml-project` on branch `adel`.

---

## Steps

### Step 1 — Install dependencies
- [x] `pip install kaggle yfinance` (pandas, numpy, scikit-learn, scipy, matplotlib already present)
- [x] Verify Kaggle CLI authenticates (`~/.kaggle/kaggle.json` installed)

### Step 2 — Download Kaggle dataset
- [x] `kaggle datasets download andrewmvd/sp-500-stocks -p data/raw/`
- [x] Unzipped into `data/raw/` (3 files: `sp500_stocks.csv` 96MB, `sp500_companies.csv`, `sp500_index.csv`)
- [x] Recorded download date 2025-04-27 in `scripts/run_data_pipeline.py`

### Step 3 — Inspect CSV
- [x] Schema confirmed: long-format (`Date, Symbol, Adj Close, Close, High, Low, Open, Volume`)
- [x] Date range: 2010-01-04 → 2024-12-20 (Kaggle hasn't refreshed through year-end)
- [x] 502 distinct tickers
- [x] **MAJOR FINDING: Kaggle dataset is broken for many large tickers.** AAPL, GOOGL, JPM, XOM, JNJ, KO, IBM, BAC, HD, BRK-B, WMT — all have ZERO valid rows in entire 2010-2024 history. 127 of the 342 NaN-dropped tickers had market cap > $50B. Likely cause: failed yfinance API calls during the Kaggle creator's update script that were never backfilled.

### Step 3b — Pivot to yfinance (proposal-approved fallback)
- [x] `pip install yfinance` ✓
- [x] Downloaded 491/502 tickers via yfinance batched (50 at a time)
- [x] 11 yfinance failures (ANSS, DAY, DFS, FI, HES, IPG, JNPR, K, MMC, PARA, WBA — recent M&A / ticker changes)
- [x] yfinance window: 2014-01-02 → **2024-12-31** (full year ✓), 2768 trading days
- [x] Saved to `data/raw/yfinance_prices.csv`, `yfinance_volume.csv`

### Step 4 — Write `data_loader.py`
- [x] Long-format Kaggle loader (`load_kaggle_long`) — kept for fallback
- [x] yfinance wide-format loader (`load_yfinance_wide`) — primary path now
- [x] Date-window filter, wide-pivot, log-returns, dollar-volume helpers

### Step 5 — Run elimination Stages 1–2 ✓
- [x] Stage 1 data integrity filter (NaN, delisted, coverage, extreme-returns)
- [x] Stage 2 anomaly filter (consecutive zeros, constant price)
- [x] Compute ranking metric (avg daily dollar volume)
- [x] Write `preprocessing_report.json` and slice CSVs

#### Phase 0 amendment: extreme-return threshold raised 0.5 → 1.0

The 0.5 log-return threshold from proposal §4.3 Stage 1d caught real historical volatility (COVID crash, PCG bankruptcy, EPAM Russia exposure, GL short-seller report, SMCI Bloomberg story, etc.) — not data errors. A 0.5 log-return = -39%/+65% move, which IS rare but happens. Raised to 1.0 (= -63%/+172%, physically implausible without a data error). The next `.tex` revision should reflect this.

---

## Survivor count — empirical result

```
Data source:            yfinance (Kaggle broken; switched per proposal §4.2 fallback)
Candidate pool size:    491    (502 in metadata, 11 yfinance failed)
Stage 1 drops:          37     (37 NaN + 0 delisted + 0 low coverage + 0 extreme-return at threshold 1.0)
Stage 2 drops:          2      (consecutive zero returns: AMCR, SW)
Total survivors:        452
top_100 universe size:  100    (TSLA, AAPL, NVDA, AMZN, MSFT, ...)
top_200 universe size:  200
all_survivors size:     452
Effective window end:   2024-12-31  ✓
Trading days in window: 2768
No missing in final returns: ✓
```

---

## Steps still to do (deferred to Phase 0 part 2)

### Step 6 — Implement PCA end-to-end
- [ ] Fit `sklearn.decomposition.TruncatedSVD` on train set (chronological 70/30 split)
- [ ] Transform test set
- [ ] Save compressed test-set matrix

### Step 7 — Implement distance distortion metric
- [ ] Compute pairwise distances on raw test set (subsampled 50K pairs)
- [ ] Compute pairwise distances on PCA-compressed test set
- [ ] Report mean absolute distortion |ρ - 1|

### Step 8 — One-number test
- [ ] Run on `top_100` slice with k=20
- [ ] Confirm pipeline produces a single distortion number

### Step 9 — Commit (this commit covers Steps 1–5)
- [x] Commit working data pipeline + plan docs to `adel` branch

---

## What to do after Phase 0

Phase 0 (steps 1–5) is **on track**. 452 survivors gives us full top_100 and top_200 universes. AAPL, MSFT, NVDA, JPM, etc. all present and ranked correctly. No further data surprises expected.

Continue to **steps 6–8** (one method + one metric end-to-end), then move to **Phase 1: Core results** per [`PROJECT_PLAN.md`](PROJECT_PLAN.md).

---

## Open items to resolve before Phase 1

- [ ] Investigate the 11 yfinance failures — try once more with fresh session, otherwise document as excluded
- [ ] Re-fetch Kaggle dataset later in year and verify if AAPL/JPM/etc. issue is fixed (would let us pin Kaggle as primary per proposal)
- [ ] Discuss with Vishal: top_100, top_200, or all_survivors as the primary universe for the paper?
- [ ] Update `AMLDS_Project_proposal-5.tex` to:
  - reflect `extreme_return_threshold = 1.0` amendment
  - note the Kaggle quality finding and yfinance-as-primary pivot
