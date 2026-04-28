# CrashSketch: Findings from the Path B experiment

This note records the findings from implementing and evaluating CrashSketch — a sparse, sign-only, quantizable Johnson–Lindenstrauss variant inspired by the TurboQuant family (PolarQuant + QJL, Google Research 2025).

Run on the laptop, 8 vCPU, 648 configs, ~3 minutes wall clock. Top_100 only, k=20, both protocols (chrono70_30 + preCOVID), all 4 working slices, 20 seeds for randomized methods.

## Setup

The CrashSketch class composes three independent ideas:

1. **Sparse sign-only JL projection** (the SOSA 2018 baseline)
2. **Random orthogonal rotation** before projection (the PolarQuant idea)
3. **Output quantization** (the QJL idea)

We tested four output modes:

- `float`: no quantization; just sparse sign-only JL with optional rotation.
- `int8`: project to floats, then z-score per coordinate using *training-set* statistics, scale to int8 range, clip and round, then dequantize back to float for downstream metric computation. Storage saving: 8× per coordinate.
- `1bit`: project, then store only `sign(Z)` (relative to zero, no training stats). Distribution-free. Storage saving: 64× per coordinate. Limitation: ‖Z_recon‖₂ = 1 by construction, which makes the anomaly_recall metric degenerate.
- `normsign`: project, then store `(sign(Z), ‖Z‖₂)` — k bits + 1 float per vector. Distribution-free. Storage saving: 24×. Reconstruction: `Z_recon = sign(Z) · ‖Z‖₂ / √k`.

All four were tested with rotation on (`_R`) and rotation off (`_noR`).

## The story in three findings

### Finding 1: random rotation alone (float output) does not help

| Method | dd_mean_abs | NN5 | ARI | anomaly_recall |
|---|---|---|---|---|
| sparse_jl | 0.124 | 0.199 | 0.317 | 0.723 |
| crashsketch_noR (float) | 0.121 | 0.198 | 0.320 | 0.727 |
| crashsketch_R (float) | 0.122 | 0.198 | 0.344 | 0.726 |

(top_100, chrono70_30, full slice, 20 seeds.)

These three are statistically indistinguishable. **Rotation alone, with floating-point output, contributes no measurable benefit.** This is theoretically expected: a random orthogonal rotation R is an isometry, so X·R·S has the same expected geometry as X·S. The PolarQuant intuition for rotation only kicks in *under quantization* — making the per-coordinate distribution uniform-ish so that uniform quantization is MSE-optimal.

### Finding 2: int8 quantization with training stats *breaks* regime-robustness

This is the most interesting empirical finding from this experiment. With float-precision JL methods, anomaly recall is essentially flat across regimes:

```
Anomaly recall            covid_only   post_covid    stress_delta
crashsketch_R (float)         0.750       0.751         +0.001
sparse_jl                     0.717       0.752         +0.036
```

But adding int8 quantization with training-set stats drops COVID-only anomaly recall by 17 points:

```
crashsketch_R_int8            0.483       0.657         +0.173
crashsketch_noR_int8          0.483       0.648         +0.164
pca                           0.667       0.847         +0.181
```

**int8 CrashSketch behaves like PCA under stress.** The two are regime-fragile for the same fundamental reason: both learn parameters from training data. PCA learns the variance subspace; int8 quantization learns the per-coordinate scale. When the test distribution shifts (COVID), training-fit parameters fail. For int8, projected COVID values lie outside the training distribution range and get clipped, destroying the "this day was unusual" signal.

This generalizes the project's earlier finding that PCA is regime-fragile: **any compression that learns parameters from training data is regime-fragile**, not just methods that explicitly fit a low-rank subspace.

### Finding 3: norm + sign decomposition recovers regime-robustness with 24× compression

Pure 1-bit (sign relative to zero) is distribution-free and regime-robust by construction, but it makes the anomaly_recall metric degenerate (‖Z_1bit‖₂ is constant). It also loses too much magnitude info for NN preservation.

The fix: store sign and norm separately. Both are distribution-free.

| Variant | bytes/vec | dd_mean_abs | NN5 | anomaly_recall calm | anomaly_recall stress |
|---|---|---|---|---|---|
| float | 160 | 0.123 | 0.198 | 0.751 | 0.750 |
| int8 | 20 | 0.160 | 0.169 | 0.657 | 0.483 |
| 1bit | 2.5 | 0.922 | 0.043 | 0.017 | 0.000 |
| **normsign** | **6.5** | **0.135** | **0.094** | **0.751** | **0.750** |

normsign is the only quantized variant that:
- Achieves **regime-robust anomaly recall** (stress delta 0.001 — identical to float).
- Achieves **24× storage compression** vs float.
- Has **non-degenerate NN preservation** (0.094 vs 0.043 for pure 1-bit).

The trade-off vs float: NN5 drops from 0.198 to 0.094 (about half). For applications where anomaly detection is the primary metric and NN search is secondary, this is acceptable.

## Why normsign works

The mathematical observation is simple: ‖sign(Z) · ‖Z‖₂ / √k‖₂² = (k · ‖Z‖₂² / k) = ‖Z‖₂². So the L₂ norm is preserved *exactly*. Since the anomaly score is `a_i = ‖X_i‖₂` and JL preserves L₂ norms in expectation, the anomaly score on the reconstructed vector matches the score on the float reconstruction.

The cost is per-coordinate magnitude information: each entry of the reconstruction has the same absolute value `‖Z‖₂ / √k`, regardless of which entries were originally large. This is what hurts NN preservation (which depends on relative magnitudes between coordinates).

## Comparison summary at top_100, k=20, COVID-only slice

```
Method                       dd      NN5     ARI    anomaly  storage
raw                         0.000   1.000   1.000   1.000    100% (uncompressed)
pca                         0.154   0.847   0.747   0.667    100% (no compression)
dense_jl                    0.131   0.687   0.604   0.717    160 bytes (full)
sparse_jl                   0.124   0.672   0.585   0.717    24 bytes (sparse, full)
crashsketch_R (float)       0.122   0.667   0.609   0.750     ~24 bytes
crashsketch_R_int8          0.488   0.473   0.364   0.483     20 bytes
crashsketch_R_1bit          0.92    0.30    0.31    0.00       2.5 bytes
crashsketch_R_normsign      0.123   0.524   0.521   0.750       6.5 bytes
```

- **Best anomaly recall**: float JL family (CrashSketch float / normsign).
- **Best NN/ARI under stress**: PCA.
- **Best storage**: 1-bit, but it loses anomaly entirely.
- **Best balance**: CrashSketch_R_normsign — anomaly = float quality, NN/ARI between 1-bit and float, storage 24× smaller than float.

## Storage budget for top_100 production deployment

For a real-time financial risk system that compresses every market state and stores billions of historical states:

```
Float64 dense JL:        2,000 bytes per (k=20 vector × 100 dim)
Float64 sparse JL:         300 bytes (sparse representation)
crashsketch_R_normsign:    302 bytes total: 300 sparse + 2 sign-bit-per-coord ÷ 8 bytes
                              Actually: sparse projection is shared across all
                              vectors, so per-vector storage is just the
                              compressed Z: 6.5 bytes vs 160 bytes for float Z.
```

So: per *day* compressed, normsign costs 6.5 bytes; float costs 160 bytes; 24× saving. Across a billion historical days, that's 6.5 GB instead of 160 GB.

## Implications for the broader project

This experiment closes the loop on a question the YOLO Phase 1 results posed: **why is float JL regime-robust while PCA is not?** The answer was always "because PCA fits training-data-dependent parameters." The CrashSketch path_b run *generalizes* this: any compression that fits training-data-dependent parameters is regime-fragile (we now have int8 as a second instance, in addition to PCA). Conversely, distribution-free compressions are regime-robust (1-bit, normsign), at the cost of either degenerate metrics (1-bit) or some loss in non-anomaly metrics (normsign).

This is a more conceptually clean story than "JL beats PCA on a single empirical observation." It now has a *mechanism*: regime robustness comes from being parameter-free.

## Limitations

1. Single dataset (S&P 500, 2014–2024). The mechanism claim should generalize but isn't tested elsewhere.
2. Single stress event (COVID). The 2018 Q4, 2022 rate-hike, 2008 GFC windows are not yet tested.
3. Single k (20) on the laptop. Cloud run with full grid would tighten and validate.
4. normsign is one specific decomposition; many variants are possible (e.g., different magnitude bins, separately quantized norm, etc.).
5. We have not formally proven the regime-robustness of normsign (we have empirical observation only). A theorem stating "distribution-free compressions preserve anomaly recall under arbitrary distribution shift" would be a real theoretical contribution.

## Files

- `src/crashsketch.py` — the four-variant CrashSketch class
- `scripts/run_phase1.py` — `--subset path_b` runs this comparison
- `results/phase1/phase1_path_b_ray.csv` — full result table

## What to do next

Two options:

**(a) Cloud run.** Take CrashSketch_R_normsign to the YOLO grid (5 universes, 100 seeds, both protocols, full k sweep). The mechanism claim becomes a multi-universe, multi-N empirical pattern. Cost: ~$0.62, ~30 min wall.

**(b) Write up the mechanism claim.** Frame the project around "regime-robustness as a property of distribution-free compressions" with a small theory section sketching the argument and the path_b empirics as evidence. This shifts the narrative from "we benchmarked JL on financial data" to "we discovered a structural property of compression methods relevant to stress detection." More publishable.

Probably both: do (a) first to get the broader empirical pattern, then write up (b) leveraging the YOLO + path_b combined data.
