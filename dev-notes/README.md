# Dev Notes

Technical notes for engineers working on this project. Audience: developers and reviewers who want to understand or reproduce the implementation, not the team / professor reading the proposal or report.

## Index

| File | Purpose |
|---|---|
| [recreate-results.md](recreate-results.md) | Step-by-step commands to recreate every result from a fresh checkout. Start here if you want to run the project. |
| [data-source-comparison.md](data-source-comparison.md) | Detailed comparison of yfinance vs the Kaggle `andrewmvd/sp-500-stocks` dataset. Documents the Phase 0 finding that the Kaggle dataset is broken for ~127 large-cap tickers. |
| [phase0-findings.md](phase0-findings.md) | Empirical findings from the Phase 0 smoke test, including amendments to the frozen proposal. |
| [architecture.md](architecture.md) | Module-by-module description of the codebase. How `data_loader`, `preprocessing`, `projections`, `pca_baseline`, `metrics` fit together. |

## Where the high-level docs live

These dev notes are deliberately technical. For higher-level project docs:

- [`AMLDS_Project_proposal-5.tex/.pdf`](../AMLDS_Project_proposal-5.tex) — frozen design (39 pages)
- [`PROJECT_PLAN.md`](../PROJECT_PLAN.md) — operational plan and phase tracking
- [`LAPTOP_PLAN.md`](../LAPTOP_PLAN.md) — Phase 0 checklist with empirical results
- [`report/preliminary_results.tex/.pdf`](../report/preliminary_results.tex) — Phase 0 results write-up

## Conventions used in dev notes

- File paths are relative to the repo root unless noted otherwise.
- All commands assume the `dev-env` Python venv is active (`source ~/dev-env/bin/activate`).
- Code references use `path/to/file.py:line_no` so they remain stable as files grow.
- Empirical numbers in these notes are pinned to specific commit SHAs to keep them traceable when results change.
