# HPC Experiment Plan: Configs to Run on GCP

## Premise

We have GCP credits expiring and a finite project window. The right strategy is "save everything, run everything that's even tangentially interesting, sort it out in analysis." This document specifies what to actually run on the cloud.

Prerequisites: [`upgrade-better-saving.md`](upgrade-better-saving.md) and [`upgrade-ray-parallelization.md`](upgrade-ray-parallelization.md) completed and tested locally.

## Cloud setup

### VM choice

| Option | vCPU / RAM | Price | When to use |
|---|---|---|---|
| `e2-standard-8` | 8 / 32 GB | ~$0.27/hr | Initial testing, validation |
| `e2-standard-16` | 16 / 64 GB | ~$0.55/hr | Phase 1 main runs (recommended) |
| `e2-standard-32` | 32 / 128 GB | ~$1.10/hr | If we want speed and credits to spare |
| `c2-standard-30` | 30 / 120 GB | ~$1.40/hr | Compute-heavy if synthetic experiments expand |

**Default recommendation: `e2-standard-16`.** Phase 1 grid runs in ~30–60 minutes on this. No GPU needed (linear algebra on small matrices).

### Setup commands (once per VM session)

```bash
# Update + Python toolchain
sudo apt-get update && sudo apt-get install -y python3.11 python3.11-venv git
python3.11 -m venv ~/dev-env
source ~/dev-env/bin/activate
pip install --upgrade pip

# Project deps
pip install pandas numpy scikit-learn scipy matplotlib yfinance kaggle ray pyarrow

# Pull repo
git clone https://github.com/ans9868/alg-ml-project.git
cd alg-ml-project
git checkout adel
```

Snapshot the disk image after this is set up — saves time on subsequent VM boots.

### Auth

- Kaggle: copy `~/.kaggle/kaggle.json` from local laptop via `gcloud compute scp` (don't put it in git).
- Git push: SSH key on GCP, added to GitHub.
- GCS for results: optional, for archiving the artifacts directory after each run.

### Cost guardrails

```bash
# Set a billing alert at $50 in the GCP console
# Set an auto-stop on the VM after 4 hours of idle (instance scheduler)
```

Phase 1 cost estimate: under $5 of credits even if we run inefficiently.

## The experiment matrix (what to actually run)

### Real-data experiments

The full grid:

| Dimension | Values | Count |
|---|---|---|
| Universe | top_100, top_200, all_survivors | 3 |
| Slice | full_incl_covid, full_excl_covid, covid_only | 3 |
| Method | raw, dense_jl, sparse_jl, pca | 4 |
| k | 5, 10, 20, 30, 50, 100 | 6 |
| s (sparse_jl only) | 1, 3, 5 | 3 |
| Seed (randomized methods only) | 0..49 | 50 |

After applying constraints (s ≤ k, deterministic methods get one seed, raw is independent of k):

```
~ 30,000 individual experiment configs
```

Each config takes ~0.1–1 sec. On 16 vCPUs, total ~30–60 min.

### Synthetic experiments (proposal §8)

For each (rank r, noise σ) combination, generate one synthetic dataset and run all the same methods:

| Dimension | Values | Count |
|---|---|---|
| Synthetic rank r | 3, 5, 10, 20 | 4 |
| Noise σ | 0.1, 0.5, 1.0 | 3 |
| Method | raw, dense_jl, sparse_jl, pca | 4 |
| k | 5, 10, 20, 30, 50, 100 | 6 |
| s | 1, 3, 5 | 3 |
| Seed | 0..49 | 50 |

```
~ 12,000 synthetic configs
```

Wall time: ~10–20 minutes on 16 vCPU.

### Stress-window experiments (extension beyond proposal)

Run the same metric suite for additional stress windows:

| Window | Approx dates | Why |
|---|---|---|
| COVID stress | 2020-02-19 to 2020-04-30 | proposal §5 |
| 2018 Q4 selloff | 2018-10-01 to 2018-12-31 | volmageddon, fed pivot |
| 2022 rate-hike onset | 2022-01-01 to 2022-06-30 | regime change |
| Optional: 2015 China devaluation | 2015-08-01 to 2015-09-30 | sharp short-lived shock |

Adds ~4× the COVID-slice configs. Even so, total stays under an hour on 16 vCPU.

### Total budget

```
Real-data:        ~30k configs ≈ 30–60 min on 16 vCPU
Synthetic:        ~12k configs ≈ 10–20 min
Stress windows:   ~3× covid-only config slice, included above
Saving overhead:  +20% wall time roughly
————————————————————————————————————————————————————————
Total Phase 1:    ~1–2 hours wall clock on a single 16 vCPU VM
Cost:             ~$1.10–$2.20 of credits
```

## What metrics to compute

The 5 from proposal §7, plus the bonus 3 we discussed:

| # | Metric | Status | Cost per config |
|---|---|---|---|
| 1 | Pairwise distance distortion | implemented | ~50ms (50K subsampled pairs) |
| 2 | Nearest-neighbor overlap (m=5, m=10) | TODO | ~20ms |
| 3 | Clustering ARI (C=3, 5, 8) | TODO | ~50ms |
| 4 | Anomaly recall @ 5%, precision @ 5%, top-10 overlap | TODO | ~5ms |
| 5 | Runtime + nnz | TODO | trivial (instrumentation) |
| **Bonus** | Procrustes alignment distance | not in proposal | ~20ms |
| **Bonus** | Reconstruction error (PCA only) | not in proposal | ~10ms |
| **Bonus** | Cosine distance preservation | not in proposal | ~30ms |

All metrics together: ~200ms per config. Comfortable.

## Order of operations

Don't try to run the whole grid on day one. Stage it:

### Stage 1 — Validation (local laptop, 10 min)

Run a tiny subset of the grid locally with the new artifact-saving enabled:

```
universe = top_100
slice = full_incl_covid
methods = pca, dense_jl, sparse_jl
k = 20
s = 3
seeds = 0..2
```

Goal: confirm artifact saving + Ray local backend produce sensible output. Inspect a saved `compressed.npz` in a notebook to make sure it's loadable.

### Stage 2 — Provision GCP, dry run (5 min)

Spin up `e2-standard-16`, run the same Stage-1 subset there with `backend="ray"`. Confirm the cloud setup works and outputs match the laptop.

### Stage 3 — Real-data grid (top_100, 30 min)

Full grid for `top_100` × all 3 slices × all methods × all k × all s × 50 seeds. Save everything. Pull the resulting CSV down to the laptop for inspection.

### Stage 4 — Expand universe (top_200, all_survivors, 30 min each)

Same grid for the larger universes. Each adds about 30 min on 16 vCPU.

### Stage 5 — Stress windows (30 min)

Add the 2018 Q4, 2022 rate-hike, and (if time) 2015 windows.

### Stage 6 — Synthetic experiments (20 min)

Run the synthetic factor-model grid.

### Stage 7 — Pull artifacts back

```
gsutil rsync -r gs://your-bucket/results/phase1/ ~/projects/alg-ml-project/results/phase1/
```

Or just use `gcloud compute scp` if we don't bother with GCS.

### Stage 8 — Figures + analysis (laptop)

All figure generation runs locally from the saved CSV + npz files. No more cloud needed.

## What to look for in the results

The "did the experiment succeed scientifically" checklist:

- [ ] **Dense JL ρ ≈ 1** at all k, all universes, all slices (consistency check — JL lemma)
- [ ] **Sparse JL ρ ≈ Dense JL ρ** within seed-noise (SOSA paper's central claim)
- [ ] **PCA explained variance high at small k** (low-rank hypothesis)
- [ ] **PCA dominates on NN overlap and clustering ARI at small k** (proposal Expected Result 1)
- [ ] **Sparse JL becomes competitive on those metrics at large k** (proposal Expected Result 2)
- [ ] **Anomaly recall degrades during COVID-only slice for all methods, but degrades least for raw / most for sparse JL** (or whatever pattern actually shows up)
- [ ] **Synthetic: PCA advantage shrinks as σ increases** (Expected Result 3)
- [ ] **Synthetic: PCA advantage shrinks as k grows past true rank r** (Expected Result 2)

If any of these are violated, that's interesting — either the data is doing something we didn't predict, or there's a bug.

## What to do if results are surprising

Two flavors of "surprising":

**(a) "Scientifically interesting" surprise.** E.g., sparse JL beats PCA on anomaly recall during COVID. Investigate, write up. The full saved-artifact tree means we can drill into specific runs without re-fitting.

**(b) "Probably a bug" surprise.** E.g., dense JL with ρ=2.0, or PCA explained variance of 99% at k=5. Reload one offending config from disk, run in a notebook with the saved compressed matrix, find the bug.

Either way, the saved artifacts pay off.

## After Phase 1

If the results look good:

1. Generate the 50-figure catalogue (proposal §11) from saved CSVs.
2. Write the final report (proposal §17 structure).
3. Amend the `.tex` to v6: yfinance primary, threshold update, any other findings.

If the results raise new questions:

1. Decide whether they warrant a Phase 2 run with different configurations.
2. We have ~4 GB of saved artifacts that already let us answer many follow-up questions without re-running.

## Open items

- [ ] Decide whether to push raw data CSVs to GCS or include them in the VM image.
- [ ] Confirm `e2-standard-16` is the right VM size (or step up to 32-vCPU for headroom).
- [ ] Decide on a cost cap and a wall-clock cap before starting.
- [ ] Set up the GCS bucket for archived results (optional but recommended).
