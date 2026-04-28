# alg-ml-project

NYU MS Computer Science research project on **random projections and low-rank structure in financial return data**, motivated by the SOSA 2018 paper *Simple Analyses of the Sparse Johnson–Lindenstrauss Transform* (Cohen, Jayram, Nelson).

**Authors:** Vishal Jha, Adel Sahuc

## Status: Phase 0 complete

The data pipeline and method/metric architecture are validated end-to-end on a 100-stock smoke test. All four methods (raw, PCA, dense Gaussian JL, sparse JL) produce theory-consistent numbers on the first metric (pairwise distance distortion). Ready to scale to the full grid in Phase 1.

## Documents

| Document | Audience | Length |
|---|---|---|
| [`AMLDS_Project_proposal-5.pdf`](AMLDS_Project_proposal-5.pdf) | Team / professor / reviewer | 39 pages — frozen design |
| [`report/preliminary_results.pdf`](report/preliminary_results.pdf) | Anyone who wants to read Phase 0 results without diving into code | ~5 pages |
| [`PROJECT_PLAN.md`](PROJECT_PLAN.md) | Project lead | Operational checklist, phase tracking |
| [`LAPTOP_PLAN.md`](LAPTOP_PLAN.md) | Active developer | Phase 0 task list with empirical results |
| [`dev-notes/metrics-explained.md`](dev-notes/metrics-explained.md) | Trader / non-academic reader | All 5 metrics in plain English with math + impact |
| [`dev-notes/`](dev-notes/) | Engineers reproducing or extending the work | Full technical notes index |

## Quick start

To reproduce everything from a fresh checkout, follow [`dev-notes/recreate-results.md`](dev-notes/recreate-results.md). Short version:

```bash
git clone https://github.com/ans9868/alg-ml-project.git
cd alg-ml-project
git checkout adel

python3 -m venv ~/dev-env
source ~/dev-env/bin/activate
pip install pandas numpy scikit-learn scipy matplotlib yfinance kaggle

python scripts/download_yfinance.py        # ~5 min
python scripts/run_data_pipeline.py        # ~30 sec
python scripts/run_phase0_smoke_test.py    # ~5 sec
```

## Headline result (Phase 0 smoke test)

100 stocks (top by avg dollar volume), $k = 20$, single seed:

| Method | mean \|ρ-1\| | mean ρ | nonzeros |
|---|---|---|---|
| raw (identity) | 0.000 | 1.000 | — |
| PCA / truncated SVD | 0.258 | 0.742 | data-adaptive |
| Dense Gaussian JL | 0.124 | **1.008** | 2,000 |
| Sparse JL (s=3) | 0.115 | **0.976** | **300** |

Dense JL preserves distances on average ($\rho \approx 1$), per the JL lemma. Sparse JL achieves comparable distortion with **6.7× fewer parameters**. PCA captures 71% of variance but systematically under-estimates pairwise distances. The full Phase 1 sweep across $k$, sparsity, seeds, and COVID slices will turn this into the actual research narrative.

For the full preliminary write-up: [`report/preliminary_results.pdf`](report/preliminary_results.pdf).

## Repository structure

```
.
├── AMLDS_Project_proposal-5.{tex,pdf}    # frozen design
├── PROJECT_PLAN.md                       # operational plan
├── LAPTOP_PLAN.md                        # active checklist
├── README.md                             # this file
│
├── src/                                  # importable modules
├── scripts/                              # entry points
├── data/{raw,processed}/                 # data artifacts
├── results/                              # experiment outputs
├── report/                               # LaTeX deliverables
└── dev-notes/                            # technical docs
```

For a deeper module walk-through, see [`dev-notes/architecture.md`](dev-notes/architecture.md).

## License

Code: MIT. Report and proposal: CC-BY-4.0.

## Notes for graders / reviewers

- The frozen design is in `AMLDS_Project_proposal-5.tex`. Changes since v5 (data source pivot, threshold amendment) are documented in [`dev-notes/phase0-findings.md`](dev-notes/phase0-findings.md) and will roll into v6 once Phase 1 is underway.
- The reported numbers are pinned to a specific yfinance download. Re-running may produce slightly different values because Yahoo retroactively re-adjusts splits/dividends. The qualitative findings are robust to this noise.
