# Architecture

Module-by-module description of the codebase. Audience: engineers working on the project.

## Top-level layout

```
alg-ml-project/
├── AMLDS_Project_proposal-5.tex   # frozen design (39 pages)
├── AMLDS_Project_proposal-5.pdf
├── PROJECT_PLAN.md                # operational plan, phase tracking
├── LAPTOP_PLAN.md                 # Phase 0 checklist + results
├── README.md                      # one-screen project overview
├── .gitignore
│
├── configs/                       # YAML config files (planned for Phase 1)
│
├── data/
│   ├── raw/                       # gitignored; produced by download scripts
│   │   ├── sp500_companies.csv    # ticker list (Kaggle metadata)
│   │   ├── yfinance_prices.csv    # wide-format adjusted close
│   │   └── yfinance_volume.csv    # wide-format volume
│   └── processed/
│       ├── candidate_universe.csv # 491 input tickers
│       ├── ranked_universe.csv    # survivors with rank + avg dollar volume
│       ├── asset_list.csv         # top_200 published default slice
│       ├── returns_matrix.csv     # gitignored; full T×N log-return matrix
│       └── preprocessing_report.json  # full audit trail of the pipeline
│
├── src/                           # importable modules
│   ├── __init__.py
│   ├── data_loader.py             # CSV I/O for both Kaggle and yfinance
│   ├── preprocessing.py           # 3-layer selection + train/test/standardize
│   ├── projections.py             # DenseGaussianJL, SparseJL
│   ├── pca_baseline.py            # PCAReducer (TruncatedSVD wrapper)
│   └── metrics.py                 # distance_distortion + helpers
│
├── scripts/                       # entry points; keep imports thin
│   ├── download_yfinance.py       # batched download, ~5 min
│   ├── run_data_pipeline.py       # ticker filter -> survivors
│   └── run_phase0_smoke_test.py   # end-to-end one-config validation
│
├── results/                       # CSV / JSON outputs from experiments
│   ├── phase0_smoke_test.csv
│   └── phase0_smoke_test.json
│
├── report/                        # LaTeX deliverables
│   ├── preliminary_results.tex
│   └── preliminary_results.pdf
│
└── dev-notes/                     # this directory
```

## Module dependency graph

```
                       ┌─────────────────────┐
                       │   data_loader.py    │
                       │  (no imports from   │
                       │   our other modules)│
                       └──────────┬──────────┘
                                  │
                                  ▼
                       ┌─────────────────────┐
                       │  preprocessing.py   │
                       │  (Stage1, Stage2,   │
                       │   train/test split, │
                       │   standardization)  │
                       └──────────┬──────────┘
                                  │
       ┌──────────────────────────┼──────────────────────────┐
       ▼                          ▼                          ▼
┌─────────────────┐   ┌─────────────────┐    ┌─────────────────────┐
│ pca_baseline.py │   │ projections.py  │    │     metrics.py      │
│   PCAReducer    │   │ DenseGaussianJL │    │ distance_distortion │
│                 │   │   SparseJL      │    │                     │
└─────────────────┘   └─────────────────┘    └─────────────────────┘
       │                       │                          │
       └───────────────────────┼──────────────────────────┘
                               ▼
              ┌──────────────────────────────────┐
              │ scripts/run_phase0_smoke_test.py │
              └──────────────────────────────────┘
```

## src/data_loader.py

I/O only. No transformations beyond loading.

| Function | Purpose |
|---|---|
| `load_kaggle_long(raw_dir)` | Read `sp500_stocks.csv` as long-format. Kept for fallback / archival. |
| `filter_date_range(df, start, end)` | Keep rows in [start, end] inclusive (long-format). |
| `pivot_wide(df, value_col)` | Long → wide pivot (rows = Date, cols = Symbol). |
| `compute_log_returns(prices)` | Daily log returns; drops first row. |
| `compute_dollar_volume(prices, volume)` | Element-wise multiply. |
| `load_company_metadata(raw_dir)` | Read `sp500_companies.csv` (sector, market cap, weight). |
| `load_yfinance_wide(raw_dir)` | Read `yfinance_prices.csv` and `yfinance_volume.csv`; returns `(prices, volume)`. **This is the primary loader after Phase 0 found Kaggle prices broken.** |

## src/preprocessing.py

The three-layer stock selection framework (proposal §4.3) plus the train/test infrastructure (§4.7–4.8).

### Layer 1 / Layer 2

| Function | Purpose |
|---|---|
| `rank_by_dollar_volume(dollar_volume)` | Average daily dollar volume per ticker, descending. Returns a `pd.Series`. |

### Layer 3 elimination

| Class / Function | Purpose |
|---|---|
| `Stage1Drops` | Dataclass holding ticker lists for `nan_in_window`, `delisted_before_end`, `low_coverage`, `extreme_returns`. |
| `run_stage1(prices, returns, window_end, ...)` | Returns a `Stage1Drops`. |
| `Stage2Drops` | Dataclass for `consecutive_zero_returns`, `constant_price`. |
| `run_stage2(prices, returns, ...)` | Returns a `Stage2Drops`. |
| `select_top_n(rankings, survivors, n)` | Selection step; returns a list of tickers. |

### Train/test/standardize (proposal §4.7–4.8)

| Function | Purpose |
|---|---|
| `chronological_train_test_split(returns, train_frac=0.7)` | Returns `(train, test)`; first `train_frac` rows → train, rest → test. No randomization. |
| `standardize_with_train_stats(train, test)` | Z-score per asset using train statistics only. Returns `(train_z, test_z, mu, sigma)`. |

## src/pca_baseline.py

Single class.

```python
class PCAReducer:
    def __init__(self, k: int, random_state: int = 0): ...
    def fit(self, X_train: np.ndarray) -> "PCAReducer": ...
    def transform(self, X: np.ndarray) -> np.ndarray: ...
    def fit_transform(self, X_train: np.ndarray) -> np.ndarray: ...
    @property
    def explained_variance_ratio(self) -> np.ndarray: ...
```

Wraps `sklearn.decomposition.TruncatedSVD`. The interface (`fit / transform / fit_transform`) matches the proposal §14.3 contract so all method classes are interchangeable.

## src/projections.py

Two classes, same interface as `PCAReducer`.

```python
class DenseGaussianJL:
    def __init__(self, k: int, seed: int): ...
    def fit(self, X_train: np.ndarray) -> "DenseGaussianJL": ...
    def transform(self, X: np.ndarray) -> np.ndarray: ...
    def fit_transform(self, X_train: np.ndarray) -> np.ndarray: ...
    @property
    def nnz(self) -> int: ...   # always N*k for dense

class SparseJL:
    def __init__(self, k: int, s: int, seed: int): ...
    def fit(self, X_train: np.ndarray) -> "SparseJL": ...
    def transform(self, X: np.ndarray) -> np.ndarray: ...
    def fit_transform(self, X_train: np.ndarray) -> np.ndarray: ...
    @property
    def nnz(self) -> int: ...   # always N*s for sparse
```

Implementation notes:
- Both use `np.random.default_rng(seed)` for reproducibility.
- Dense: $A_{ij} \sim \mathcal{N}(0, 1/k)$ matches proposal §6.3.
- Sparse: each row of $S$ has exactly $s$ nonzero entries at random column indices, with random signs scaled by $1/\sqrt{s}$ (proposal §6.4).
- Both are data-independent — `fit` only uses `X_train.shape[1]` (which is N), not the values. The `X_train` argument is part of the interface for symmetry with `PCAReducer`, which actually uses the data.
- Constraint enforced: `s <= k` for sparse JL.

## src/metrics.py

Phase 0 implements only Metric 1 (proposal §7.1). Other metrics added in Phase 1.

| Function | Purpose |
|---|---|
| `sample_pair_indices(n, n_pairs, rng)` | Sample `n_pairs` unique (i, j) pairs with i < j from {0..n-1}. Falls back to `np.triu_indices` if n_pairs ≥ all pairs. |
| `pairwise_distances(Z, pair_idx)` | Euclidean distance for each row pair. Returns a 1-D array. |
| `distance_distortion(Z_raw, Z_compressed, n_pairs, seed)` | Returns dict with `mean_abs_distortion`, `median_abs_distortion`, `p95_abs_distortion`, `mean_rho`, `n_pairs`. |

## scripts/

Thin entry points. Keep business logic in `src/`.

| Script | Purpose | Runtime |
|---|---|---|
| `download_yfinance.py` | Batched yfinance download. Reads ticker list from Kaggle metadata, writes `data/raw/yfinance_*.csv`. | ~5 min |
| `run_data_pipeline.py` | End-to-end data prep: load → filter → returns → elimination → rank → save artifacts. | ~30 sec |
| `run_phase0_smoke_test.py` | Smoke test: load returns → top_100 → split → standardize → fit/transform 4 methods → distance distortion → write `results/phase0_smoke_test.{csv,json}`. | ~5 sec |

All three have a `main()` function and a single canonical invocation `python scripts/<script>.py` from the repo root.

## What's NOT here yet (Phase 1 work)

- `src/synthetic_data.py` — synthetic factor-model data generator (proposal §8).
- `src/experiment_config.py` — `ExperimentConfig` dataclass per proposal §9.3.
- `src/experiment_runner.py` — local backend that sweeps the (method × k × s × seed × slice) grid.
- `src/parallel_backend.py` — Ray backend (Phase 2).
- `src/experiments_real.py` / `experiments_synthetic.py` / `experiments_covid.py` / `experiments_regime.py` — orchestrators per slice.
- `src/plotting.py` and the 50-figure catalogue.
- Remaining metrics: NN overlap, clustering ARI, anomaly recall, runtime/memory.

These will arrive incrementally in Phase 1.

## Conventions

- **Type hints**: required on all public function signatures.
- **NumPy**: prefer arrays for hot paths (the projections/metrics layer); keep DataFrames for I/O and bookkeeping.
- **Random state**: every randomized component takes a `seed` argument; we use `np.random.default_rng(seed)` rather than the legacy `np.random` module.
- **Reproducibility**: all entry-point scripts use `seed=0` by default; Phase 1 will sweep seeds 0-9.
- **No globals**: configuration lives in script constants near the top, not in module-level state.
