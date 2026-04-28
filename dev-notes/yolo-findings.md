# YOLO Findings (Phase 1, full grid)

This is the analysis writeup of the first complete Phase 1 grid: 48,240 configs across 5 universes, 2 training protocols (chronological 70/30 + the §5.5 pre-COVID protocol), 4 methods, 6 k-values, 3 sparsity values, and 100 seeds for randomized methods. Run on a 32-vCPU GCP VM in ~34 minutes.

Source: `results/phase1/cloud/phase1_yolo_ray.csv`. Plots in `figures/{universe}/*.png` and `figures/cross_universe/*.png` (49 plots total — 9 per-universe plots × 5 universes + 4 cross-universe). The CSV is committed; per-config artifacts are gitignored but were copied back to `data/cloud_artifacts/runs/` for notebook use.

## TL;DR — three real findings

1. **JL methods are scale-invariant; PCA degrades with N.** Distance distortion at k=20 stays at ~0.125 for both dense and sparse JL across N ∈ {100, 200, 300, 400, 452}. PCA's distortion grows from 0.258 to 0.355 over the same range. This is exactly the SOSA paper's central claim playing out on real financial data.

2. **All methods lose ~50% of nearest-neighbor structure during COVID.** PCA preserves clustering structure (ARI stable at ~0.82) and PCA's anomaly recall drops by 18 points. JL methods lose NN/ARI but their anomaly recall is essentially unchanged.

3. **Sparse JL achieves dense JL's quality with 85% sparsity at k=20.** Identical seed-averaged distance distortion (0.122 vs 0.127) and matching nearest-neighbor overlap. The SOSA "be sparse without paying" claim holds empirically.

These are the three things to put on slides.

---

## 1. Universe scaling: JL beats PCA *more* as N grows

Table: distance distortion at k=20 across universe sizes (`chrono70_30/full`):

| N (assets) | dense_jl | sparse_jl(s=3) | **PCA** |
|---|---|---|---|
| 100 | 0.127 | 0.122 | **0.258** |
| 200 | 0.125 | 0.123 | 0.317 |
| 300 | 0.128 | 0.123 | 0.338 |
| 400 | 0.128 | 0.126 | 0.350 |
| **452** | **0.127** | **0.125** | **0.355** |

**Plot:** [`figures/cross_universe/universe_scaling_dd.png`](../figures/cross_universe/universe_scaling_dd.png).

**Why this matters:** The Johnson–Lindenstrauss lemma says the embedding dimension k needed for a target distortion ε depends on the *number of points* (here, T_test trading days) and ε, **not on the input dimension N**. So at fixed k, JL's epsilon is essentially constant in N. PCA, in contrast, maps to the top-k subspace of an N-dimensional input. As N grows, the orthogonal complement (the part PCA discards) gets relatively larger — so distance distortion grows.

This is one of the cleanest empirical confirmations of the SOSA paper's framing in our dataset. It's the kind of result that justifies sparse JL when you scale to thousands of assets.

---

## 2. COVID stress: how methods degrade under regime change

Table: how each metric changes during COVID-only vs post-COVID, both evaluated under the §5.5 pre-COVID training protocol (top_100, k=20). **Negative for distortion / positive for the others = stays good during COVID**.

| metric | dense_jl | sparse_jl(s=3) | **pca** |
|---|---|---|---|
| dd_mean_abs (lower better) | +0.002 | −0.002 | **−0.103** |
| nn_overlap_at_5 (higher better) | **−0.49** | **−0.49** | **−0.47** |
| ari_mean (higher better) | −0.25 | −0.24 | **−0.004** |
| anomaly_recall (higher better) | −0.009 | +0.018 | **−0.18** |

**Plots:** [`figures/top_100/covid_anomaly_recall.png`](../figures/top_100/covid_anomaly_recall.png), [`figures/top_100/covid_dd.png`](../figures/top_100/covid_dd.png), [`figures/top_100/covid_nn5.png`](../figures/top_100/covid_nn5.png), [`figures/top_100/covid_ari.png`](../figures/top_100/covid_ari.png). Same plots are also produced for top_200 / top_300 / top_400 / all_survivors under their respective subdirectories.

Three findings here:

**(a) All methods lose ~50% of nearest-neighbor structure during COVID.** Trading days that look similar in normal regimes cease to be neighbors during the crash. This is *not* a method defect — it's a real property of the data. The cross-sectional return geometry changes during a crisis.

**(b) PCA preserves clustering structure during COVID.** ARI stays at ~0.82 across both regimes, while JL drops by 0.24–0.25 ARI. Variance preservation pays off here: PCA captures the same low-rank factor structure during stress as in calm regimes, even when the raw distances scale up.

**(c) PCA's anomaly recall plummets 18 points; JL methods barely move.** This is the single most trader-relevant finding. If you use PCA-compressed states to detect "unusual market days," you'll catch ~67% of true anomalies during COVID vs ~85% in calm regimes. JL methods stay around 73% in both. So sparse JL is more **robust** as an anomaly detector across regimes, even though PCA wins on average.

The reason: PCA's top-k components fit on pre-COVID data don't fully span the COVID-day variance directions. Anomalous days in COVID get projected into a stale low-rank subspace, losing their "unusualness" signal. JL's data-independent random projection doesn't have this problem.

**(d) PCA's distance distortion is *lower* during COVID** (mean |ρ-1| drops from 0.257 to 0.154). This is initially counterintuitive but makes sense: during high-vol periods raw distances are larger, the relative error in compressed distances shrinks. JL methods don't show this effect because they preserve distances on average regardless of regime.

---

## 3. Sparse JL achieves dense JL's quality with 6.7× fewer parameters

At k=20 and s=3, sparse JL has 300 nonzeros (`N×s`); dense JL has 2,000 nonzeros (`N×k`). That's a 6.7× reduction. Quality?

| metric | dense JL | sparse JL (s=3) |
|---|---|---|
| dd_mean_abs | 0.127 | 0.122 |
| nn_overlap_at_5 | 0.197 | 0.204 |
| ari_mean | 0.342 | 0.350 |
| anomaly_recall_5pct | 0.710 | 0.738 |

Within seed-noise, identical. Sparse JL even nudges ahead on anomaly recall (probably noise across 100 seeds, but interesting).

**Plot:** [`figures/top_100/sparsity_tradeoff.png`](../figures/top_100/sparsity_tradeoff.png) shows the full sparsity-vs-distortion curve across (k, s) for top_100; same plot exists for each of the other universes.

**Why this matters in practice:** Sparse JL's projection matrix can be stored as sparse data and applied via sparse matrix-vector multiplication. For a streaming pipeline that compresses every day's market state in real time, this is a 6.7× speedup with no measurable quality loss.

---

## 4. The full headline panel (top_100, chrono/full, mean ± std across seeds)

| method | dd_mean_abs ↓ | nn_overlap_at_5 ↑ | ari_mean ↑ | anomaly_recall ↑ | sparsity_ratio ↑ |
|---|---|---|---|---|---|
| **raw** | 0.000 | 1.000 | 1.000 | 1.000 | (n/a) |
| **pca** | 0.258 | **0.374** | **0.818** | **0.810** | 1.000 |
| **dense_jl** | 0.127 | 0.197 | 0.342 | 0.710 | 0.000 |
| **sparse_jl (s=3)** | **0.122** | 0.204 | 0.350 | 0.738 | **0.850** |

**Plot:** [`figures/top_100/metric_panel_chrono.png`](../figures/top_100/metric_panel_chrono.png) (4-panel sweep across k). Same plot for the preCOVID protocol is at [`figures/top_100/metric_panel_precovid_all_test.png`](../figures/top_100/metric_panel_precovid_all_test.png).

The "winner" depends entirely on what you're optimizing for:

| If you care about... | Use... | Why |
|---|---|---|
| Pairwise distances | **sparse JL** | Lowest mean \|ρ-1\| and 85% sparser than dense JL |
| "Find similar past days" lookups | **PCA** | Almost double the NN overlap |
| Regime classification (clustering) | **PCA** | ARI 0.82 vs 0.35 |
| Catching crash days in calm markets | PCA | Best anomaly recall |
| Catching crash days under stress (COVID) | **JL methods** | Don't lose 18 points like PCA does |
| Memory / streaming compute budget | **sparse JL** | 6.7× fewer parameters |

This is the "trader's decision tree" from `dev-notes/metrics-explained.md`, now backed by real numbers.

---

## 5. Limitations of these results

- **Single dataset:** S&P 500 today's constituents, 2014–2024, top_100 to all_survivors. Results may not generalize to other markets, instruments, or eras.
- **Survivorship bias:** today's S&P 500 backfilled. Documented in proposal §15 and `dev-notes/data-source-comparison.md`. Companies acquired/bankrupt mid-period are excluded.
- **One stress window:** COVID (Feb–Apr 2020). Adding 2018 Q4, 2022 rate hikes, 2015 China devaluation would generalize the "stress robustness" claim but those weren't in this run. Tracked in `dev-notes/hpc-experiments.md` for a Phase 2 run.
- **PCA's COVID anomaly drop could be specific to fitting on pre-COVID data only.** A robust check: refit PCA on rolling windows during the test period; we didn't run that protocol.
- **Synthetic factor-model experiment (proposal §8) not yet run.** That's the controlled experiment that would let us isolate "JL wins because of high N" from "JL wins because of factor noise". Easy add for next cloud run.

---

## 6. What to do with this

**For the final report (proposal §17 structure):**
- Sections "Real-data results" and "COVID-stress results" can lift directly from this doc.
- The 49 plots in `figures/{universe}/*.png` and `figures/cross_universe/*.png` cover most of the proposal §11 figure-catalog Phase-1 demand. Synthetic plots (figs 43–48 in §11) require running the synthetic experiment.
- The `best_method_table` from analyze_phase1.py becomes the §13.1 "Final recommendation table".

**Proposal v6 amendments needed:**
- Make yfinance the primary data source (not the broken Kaggle dataset)
- Update Stage 1d extreme-return threshold from 0.5 to 1.0
- Add the §5.5 pre-COVID protocol details
- Add the new universe slices (top_300, top_400) we ran
- Document the bonus metrics if we added any (we didn't this round; reserved for next run)

**Next experiments worth running** (with leftover GCP credits):
1. Synthetic factor-model grid (proposal §8) — known-truth control experiment.
2. Additional stress windows: 2018 Q4, 2022 rate hikes.
3. Bigger seed count (100 → 500) for the headline COVID-anomaly-recall finding to confirm the 18-point PCA drop with tight CIs.
4. Walk-forward / rolling-PCA evaluation: does PCA's anomaly drop persist if PCA is refit periodically?

---

## 7. How to reproduce these numbers

```bash
# On the laptop with the YOLO CSV pulled down:
source ~/dev-env/bin/activate
cd ~/projects/alg-ml-project
python scripts/analyze_phase1.py results/phase1/cloud/phase1_yolo_ray.csv

# Outputs:
#   figures/{universe}/*.png   (45 per-universe plots)
#   figures/cross_universe/*.png  (4 cross-universe plots)
#   results/phase1/cloud/yolo_*_summary.csv  (3 summary tables)
```

The full per-config artifact tree is in `data/cloud_artifacts/runs/` (5.3 GB, not committed). Each run dir has `compressed.npz`, `projection.npz`, `metrics.json`, `run_meta.json`, so any new metric can be added later without re-fitting via `scripts/recompute_metrics.py`.

If the CSV gets lost: re-run `python scripts/run_phase1.py --backend ray --subset yolo --num-cpus 32` on a 32-vCPU machine. Wall clock ~34 min, ~$0.62 of credits.
