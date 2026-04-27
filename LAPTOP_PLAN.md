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

## Steps 6–8 — End-to-end smoke test ✓

### Step 6 — PCA end-to-end ✓
- [x] `src/preprocessing.py`: chronological 70/30 split, z-score with train-only statistics
- [x] `src/pca_baseline.py`: `PCAReducer` wrapping `TruncatedSVD` with fit/transform interface (proposal §14.3)
- [x] Fit on train (1936 days), transform test (831 days)

### Step 7 — Distance distortion metric ✓
- [x] `src/metrics.py`: `sample_pair_indices`, `pairwise_distances`, `distance_distortion`
- [x] Subsamples 50K pairs with fixed seed
- [x] Returns mean / median / 95p `|rho - 1|`, plus mean rho

### Step 8 — One-number test ✓
- [x] `scripts/run_phase0_smoke_test.py` ties it all together
- [x] top_100 universe, k=20, single PCA run
- [x] Result saved to `results/phase0_smoke_test.{csv,json}`

#### Smoke test result (top_100, k=20, seed=0, all 4 methods)

```
N (assets):                    100
k (compressed dim):            20
Trading days train/test:       1936 / 831
Train range:                   2014-01-03 .. 2021-09-10
Test  range:                   2021-09-13 .. 2024-12-31
PCA cumulative explained var:  0.7109     (strong low-rank structure)
Pairs sampled:                 50,000
```

| Method | mean \|ρ-1\| | median \|ρ-1\| | p95 \|ρ-1\| | mean ρ | nnz |
|---|---|---|---|---|---|
| raw | 0.0000 | 0.0000 | 0.0000 | 1.0000 | (identity) |
| **PCA** | 0.2580 | 0.2523 | 0.4443 | **0.7420** | data-adaptive |
| **dense JL** | 0.1242 | 0.1059 | 0.3020 | **1.0082** | 2000 |
| **sparse JL** s=3 | 0.1147 | 0.0969 | 0.2821 | **0.9756** | **300** |

**Interpretation (theory holds):**

- **PCA mean ρ = 0.74** — systematically under-estimates distances. PCA preserves variance in the top-k factor subspace and discards orthogonal directions. Distances between days that differ in the discarded directions get compressed.
- **Dense Gaussian JL mean ρ = 1.008** — distance preservation almost exactly. The Johnson–Lindenstrauss lemma in action.
- **Sparse JL mean ρ = 0.976** with only 300 nonzeros vs 2000 for dense — **6.7× fewer parameters**, comparable distortion. The sparsity is essentially free at this k.
- **PCA captures 71% of variance** — strong low-rank structure in financial returns, supporting the proposal's hypothesis.

⚠️ *Single seed result.* Per proposal §6.7 fairness rule, the real comparison averages 10 seeds. Sparse JL marginally beating dense JL on \|ρ-1\| here is noise — both should be near-equivalent in expectation.

### Step 9 — Commit ✓
- [x] Commit working scaffold to `adel` branch

---

## What to do after Phase 0

**Phase 0 complete.** All 8 steps green:
- Data pipeline (yfinance → 452 survivors) ✓
- PCA baseline ✓
- Distance distortion metric ✓
- End-to-end smoke test produces sensible numbers ✓

Move to **Phase 1: Core results** per [`PROJECT_PLAN.md`](PROJECT_PLAN.md). Phase 1 needs:

- `src/projections.py` — dense Gaussian JL + sparse JL with the fit/transform interface
- `src/experiment_config.py` — `ExperimentConfig` dataclass per proposal §9.3
- `src/experiment_runner.py` — local backend; spans (method × k × s × seed × slice) grid
- Remaining metrics — NN overlap, clustering ARI, anomaly recall, runtime
- COVID slice handling
- Synthetic factor-model data generator

This can stay on the laptop (single-node multiprocessing). Cloud only becomes interesting once we add Ray for the synthetic+expanded grid in Phase 2.

---

## Open items to resolve before Phase 1

- [ ] Investigate the 11 yfinance failures — try once more with fresh session, otherwise document as excluded
- [ ] Re-fetch Kaggle dataset later in year and verify if AAPL/JPM/etc. issue is fixed (would let us pin Kaggle as primary per proposal)
- [ ] Discuss with Vishal: top_100, top_200, or all_survivors as the primary universe for the paper?
- [ ] Update `AMLDS_Project_proposal-5.tex` to:
  - reflect `extreme_return_threshold = 1.0` amendment
  - note the Kaggle quality finding and yfinance-as-primary pivot
