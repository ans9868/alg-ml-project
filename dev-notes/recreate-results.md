# Recreate the Results

Step-by-step commands to reproduce every Phase 0 result from a fresh checkout. Tested on macOS 24.2 with Python 3.10.18.

## 0. Prerequisites

- Python 3.10+ (we use 3.10.18 in the `dev-env` venv).
- ~500 MB free disk for raw + processed data.
- Internet for the initial yfinance download (about 5 minutes).
- A Kaggle account with API token (only if you also want the Kaggle company-metadata file; not strictly required to reproduce numbers).

## 1. Clone and enter the repo

```bash
git clone https://github.com/ans9868/alg-ml-project.git
cd alg-ml-project
git checkout adel
```

## 2. Set up the Python environment

```bash
python3 -m venv ~/dev-env
source ~/dev-env/bin/activate
pip install --upgrade pip
pip install pandas numpy scikit-learn scipy matplotlib yfinance kaggle
```

If you already have a working venv, activate that instead.

## 3. (Optional) Get the Kaggle company-metadata file

This step is only needed if you want to regenerate `data/raw/sp500_companies.csv`. The file is small (~800 KB) and lists tickers + sector + market cap; it drives the yfinance download.

If you don't have Kaggle credentials, you can skip this — the repo includes a small fallback ticker list, or you can scrape Wikipedia's "List of S&P 500 companies" page.

```bash
mkdir -p ~/.kaggle
# Drop your kaggle.json (download from https://www.kaggle.com/settings -> "API" -> "Create New Token")
chmod 600 ~/.kaggle/kaggle.json

mkdir -p data/raw
kaggle datasets download andrewmvd/sp-500-stocks -p data/raw/ --unzip
```

After this you should see `data/raw/sp500_companies.csv`, `data/raw/sp500_index.csv`, `data/raw/sp500_stocks.csv`. The first is what we need; the latter two are not used (the price data in `sp500_stocks.csv` is broken — see [data-source-comparison.md](data-source-comparison.md)).

## 4. Download daily prices via yfinance

```bash
python scripts/download_yfinance.py
```

What this does:
- Reads the ticker list from `data/raw/sp500_companies.csv`.
- Calls yfinance in batches of 50 tickers (1-second sleep between batches to be polite).
- Saves wide-format `data/raw/yfinance_prices.csv` (rows = Date, cols = Symbol) and `yfinance_volume.csv`.
- Saves `data/raw/yfinance_failed.txt` listing tickers that yfinance couldn't fetch (recent M&A / ticker changes — typically ~11 of 502).

Expected runtime: about 5 minutes. Expected output: `Downloaded successfully: 491` (give or take a few depending on transient network conditions).

## 5. Run the data pipeline

```bash
python scripts/run_data_pipeline.py
```

What this does:
- Loads the yfinance wide CSVs.
- Filters to the 2014-01-01 → 2024-12-31 window.
- Computes log returns and dollar volume.
- Runs Stage 1 (data integrity) and Stage 2 (anomaly) elimination filters.
- Ranks survivors by average daily dollar volume.
- Writes:
  - `data/processed/candidate_universe.csv` (502 input tickers)
  - `data/processed/ranked_universe.csv` (survivors with rank + avg dollar volume)
  - `data/processed/asset_list.csv` (top_200 — the published default slice)
  - `data/processed/returns_matrix.csv` (full wide return matrix; gitignored — too large)
  - `data/processed/preprocessing_report.json` (full audit trail)

Expected output (from a clean run):

```
Candidate pool size:    491
Stage 1 drops:          37
Stage 2 drops:          2
Total survivors:        452
top_100 universe size:  100
top_200 universe size:  200
all_survivors size:     452
Effective window end:   2024-12-31
Trading days in window: 2768
```

If your numbers differ by a few tickers, that is normal — yfinance results are not perfectly reproducible (Yahoo retroactively re-adjusts splits/dividends). The committed `data/processed/*.csv` artifacts pin the exact run that produced our reported numbers.

## 6. Run the smoke test

```bash
python scripts/run_phase0_smoke_test.py
```

What this does:
- Loads `returns_matrix.csv`.
- Restricts to the `top_100` universe.
- Chronological 70/30 split.
- Z-score with train statistics only.
- Fits all four methods at `k = 20`, `seed = 0`:
  - raw (identity)
  - PCA (`TruncatedSVD`)
  - Dense Gaussian JL
  - Sparse JL with `s = 3`
- Computes pairwise distance distortion on 50,000 sampled test-day pairs.
- Writes `results/phase0_smoke_test.csv` and `.json`.

Expected output (committed in the repo):

```
method        mean|rho-1|   median   p95   mean rho
raw           0.0000        0.0000   0.0000  1.0000
pca           0.2580        0.2523   0.4443  0.7420
dense_jl      0.1242        0.1059   0.3020  1.0082
sparse_jl_s3  0.1147        0.0969   0.2821  0.9756
```

Different yfinance download timing → very slightly different numbers (small last-decimal-place noise). The qualitative ordering should be identical.

## 7. Compile the preliminary report

```bash
cd report
pdflatex preliminary_results.tex
pdflatex preliminary_results.tex   # second pass for refs
```

Output: `report/preliminary_results.pdf`.

## Knobs you can tweak

The smoke-test driver script has these constants near the top:

```python
UNIVERSE = "top_100"     # also: "top_200" (200 stocks) or "all_survivors" (452)
K = 20                   # also: 5, 10, 30, 50, 100
SPARSE_S = 3             # also: 1 or 5
TRAIN_FRAC = 0.7
N_PAIRS = 50_000
SEED = 0
```

For different universe sizes, change `UNIVERSE`. The pipeline already produces all three slices. Sweeping `K` and `SEED` is what Phase 1 does as a proper grid (currently the smoke test runs one configuration).

## Troubleshooting

**`ModuleNotFoundError: No module named 'src'`**
Run from the repo root, not from inside `scripts/`. The script appends `PROJECT_ROOT` to `sys.path`, but it expects to be invoked from one level above.

**yfinance returns mostly empty data**
You may have hit a rate limit. Wait 5 minutes and retry, or use a VPN if you're in a region Yahoo Finance blocks.

**Numbers differ from this document**
Most likely the yfinance download captured at a slightly different time. If they differ by more than a few percent, something is wrong — check the `survivors` count from `run_data_pipeline.py` first.

**`pdflatex` fails**
You probably don't have a TeX distribution. On macOS: `brew install --cask mactex` or use TinyTeX. The `.tex` files compile cleanly with TeX Live ≥ 2021.
