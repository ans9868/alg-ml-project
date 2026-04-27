# alg-ml-project — Master Project Plan

This is the high-level plan for the AMLDS project on **random projections and low-rank structure in financial return data**, motivated by the SOSA 2018 paper *Simple Analyses of the Sparse Johnson–Lindenstrauss Transform* (Cohen, Jayram, Nelson).

The frozen design lives in [`AMLDS_Project_proposal-5.tex`](AMLDS_Project_proposal-5.tex) / [`AMLDS_Project_proposal-5.pdf`](AMLDS_Project_proposal-5.pdf). This file is the operational checklist.

---

## Research question

When does data-independent sparse JL compression preserve the geometry of factor-structured financial return data, and when does data-adaptive PCA dominate?

Methods compared (all reduce $\mathbb{R}^N \to \mathbb{R}^k$, except raw):
1. Raw uncompressed return vectors
2. Dense Gaussian Johnson–Lindenstrauss
3. Sparse Johnson–Lindenstrauss
4. PCA / truncated SVD

---

## Phased execution plan

| Phase | Where | Goal | Status |
|---|---|---|---|
| **0. Laptop smoke test** | local laptop | Pipeline runs end-to-end on tiny universe; design is empirically validated | **steps 1–5 done; data pipeline produces 452-survivor universe** |
| **1. Core results** | local or e2-standard-4 GCP | Core grid as proposed (top-200, k ∈ {5,10,20,30,50,100}, s ∈ {1,3,5}, 10 seeds, 3 COVID slices) | not started |
| **2. Expanded grid** | e2-standard-8/16 GCP | Larger universe, more seeds, synthetic experiments, regime analysis | not started |
| **3. Figure generation + report** | local | 40+ figures, 10 tables, final LaTeX report | not started |

### Phase 0 findings (empirical, Apr 2025)

1. **Kaggle dataset `andrewmvd/sp-500-stocks` is broken for many large tickers.** AAPL, GOOGL, JPM, XOM, JNJ, KO, IBM, BAC, HD, BRK-B, WMT all have zero valid rows in the entire 2010–2024 history. 127 of the dropped tickers had market cap >$50B. Cause: failed scraper update for those tickers, never backfilled.
2. **Switched to yfinance as primary data source.** Proposal §4.2 already listed it as the fallback. 491/502 tickers downloaded successfully; 11 failures due to recent M&A or ticker changes.
3. **Extreme-return threshold raised 0.5 → 1.0.** The 0.5 log-return threshold from the proposal caught real volatility events (COVID, bankruptcies, news shocks). 1.0 catches only physically-implausible data errors.
4. **Final survivor count: 452** out of 491 candidate tickers. Comfortably supports `top_100`, `top_200`, and `all_survivors` slices.

Phase 0 details live in [`LAPTOP_PLAN.md`](LAPTOP_PLAN.md).

---

## Core decisions (frozen)

| Decision | Choice |
|---|---|
| Asset universe | S&P 500 today's constituents (survivorship bias declared) |
| Data source | yfinance (primary, after Phase 0 found Kaggle broken for AAPL/JPM/etc.); Kaggle company metadata still used for ticker list |
| Selection framework | 3-layer: candidate pool → rank by avg dollar volume → eliminate → top_100 / top_200 / all_survivors |
| Date range | 2014-01-01 to 2024-12-31 |
| Returns | Daily log returns |
| Train/test split | Chronological 70/30 |
| Standardization | Z-score per asset using train statistics only |
| Methods | Raw, Dense Gaussian JL, Sparse JL, PCA |
| Core dimensions | k ∈ {5, 10, 20, 30, 50, 100} |
| Sparse JL sparsity | s ∈ {1, 3, 5}, primary s=3 |
| Random seeds | 10 (for randomized JL methods) |
| COVID stress window | 2020-02-19 to 2020-04-30 |
| Synthetic ranks | r ∈ {3, 5, 10, 20} |
| Synthetic noise | σ ∈ {0.1, 0.5, 1.0} |
| Parallelization | Local backend required; Ray optional for large grids |

---

## Open decisions (still to make)

| Decision | Default if not made | Owner |
|---|---|---|
| Dual-class shares (GOOG/GOOGL) — keep both or one? | Keep higher-volume only | Adel + Vishal |
| Test-set distance sampling — all pairs or subsample? | Subsample 50K pairs with fixed seed | Adel |
| Random seed schema | `seed = hash((method, k, s, base)) % 2**32` | Adel |
| Vishal's role / work split | TBD | Adel + Vishal |
| Raw data committed to git or not | Don't commit, document version pin | Adel |
| Time budget — submission date / scope-cut date | TBD | Adel + Vishal |

---

## Repo structure (target)

Per proposal §13 Implementation Output Contract:

```
alg-ml-project/
├── AMLDS_Project_proposal-5.tex      # frozen design
├── AMLDS_Project_proposal-5.pdf
├── PROJECT_PLAN.md                    # this file
├── LAPTOP_PLAN.md                     # Phase 0 checklist
├── README.md
├── pyproject.toml
├── .gitignore
│
├── configs/
│   ├── experiment_config.yaml         # Kaggle version pin, universe slice, seeds
│   └── figure_catalog.yaml
│
├── data/
│   ├── raw/                           # Kaggle CSV (.gitkeep, not committed)
│   └── processed/
│       ├── candidate_universe.csv
│       ├── ranked_universe.csv
│       ├── returns_matrix.csv
│       ├── standardized_returns_train.csv
│       ├── standardized_returns_test.csv
│       ├── asset_list.csv
│       ├── date_index.csv
│       └── preprocessing_report.json
│
├── src/
│   ├── data_loader.py
│   ├── preprocessing.py
│   ├── projections.py                 # raw, dense JL, sparse JL
│   ├── pca_baseline.py
│   ├── metrics.py
│   ├── synthetic_data.py
│   ├── experiment_config.py
│   ├── experiment_runner.py
│   ├── parallel_backend.py
│   ├── experiments_real.py
│   ├── experiments_synthetic.py
│   ├── experiments_covid.py
│   ├── experiments_regime.py
│   ├── plotting.py
│   └── utils.py
│
├── scripts/
│   ├── run_data_pipeline.py
│   ├── run_real_experiments.py
│   ├── run_synthetic_experiments.py
│   ├── run_covid_experiments.py
│   ├── run_regime_experiments.py
│   ├── generate_figures.py
│   └── run_all.py
│
├── notebooks/
│   ├── exploratory_analysis.ipynb
│   └── final_results.ipynb
│
├── results/                           # CSV outputs, one per experiment family
├── figures/                           # 50 PNGs per figure catalogue
└── report/
    ├── final_report.tex
    └── references.bib
```

---

## Definition of done

The project is "done" when [`AMLDS_Project_proposal-5.tex`](AMLDS_Project_proposal-5.tex) §18 (Definition of Success) is satisfied:

1. Reproducible data pipeline anchored to a pinned Kaggle dataset
2. Final clean return matrix with no missing values
3. Clean comparison of raw, Gaussian JL, sparse JL, PCA across core dimensions
4. Seed-averaged JL results
5. COVID including/excluding/only result sets
6. Synthetic factor-model experiments varying rank and noise
7. Ray-compatible parallel execution with local fallback
8. At least 40 generated figures
9. Tables and figures suitable for the final report
10. Clear conclusion: when sparse JL is competitive with PCA, when it is not
