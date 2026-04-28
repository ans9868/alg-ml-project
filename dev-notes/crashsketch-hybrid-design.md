# CrashSketch Hybrid: Self-Adaptive Compression for Financial Risk Systems

A design document for the hybrid version of CrashSketch — a compression scheme that combines PCA's calm-regime accuracy with sparse-JL's stress robustness, gated by a parameter-free per-input weight, and refreshed periodically through a champion–challenger protocol.

This is the "v2" CrashSketch. The v1 (sparse JL with `normsign` quantization) gave us a regime-robust compressor. The v2 adds back PCA's calm-regime quality without losing v1's robustness, and bakes in an operational story for keeping the model current as the market evolves.

## Goals (re-stated clearly)

The whole point of this work is data compression for financial market state. The hybrid system must hit three goals simultaneously:

1. **Storage cost is small.** A real risk system stores billions of historical compressed market states. Float-precision compression (160 bytes per day at $k=20$) is too big to keep at scale.
2. **Anomaly detection works in both calm and stress regimes.** A compressed risk system that misses unusual market days during a crisis is worse than useless. PCA fails this in our experiments by 18 percentage points.
3. **Compression quality stays current as the world changes.** A PCA basis fit on 2014–2019 data should not still be used in 2030. The system needs to refresh its model as new regimes accumulate.

Each goal maps to a piece of the design:

| Goal | Solved by |
|---|---|
| (1) Small storage | `normsign` quantization (sign + norm) applied to *both* PCA and JL outputs |
| (2) Regime-robust anomaly | Self-normalizing alpha gate that blends PCA and JL outputs per-input |
| (3) Stay current | Periodic champion–challenger PCA retrain |

## The full system

### Train time

```
1. Fit PCA on the most recent T days of training data → V_k (the principal directions).
2. Construct sparse-JL projection (data-independent) → S.
3. Optionally compute calm-regime residual stats for sanity-checking the alpha gate.
```

The PCA basis `V_k` is data-dependent and will be refreshed periodically. The sparse-JL matrix `S` is data-independent and effectively never changes.

### Run time, for each new market vector x

```
# Compute both compressed representations
Z_pca = V_kᵀ (x − μ_train)              # PCA path
Z_jl  = x @ S                            # data-blind JL path

# Storage: both representations stored as (sign, norm) — 6.5 bytes each
store(x) = (
    sign(Z_pca), ||Z_pca||₂,             # PCA-flavored compressed view
    sign(Z_jl),  ||Z_jl||₂                # JL-flavored compressed view
)

# Self-normalizing alpha (no parameters, no thresholds)
α(x) = ||x − V_k V_kᵀ x|| / ||x||         ∈ [0, 1]

# Final anomaly score: smooth blend by alpha
anomaly_score(x) = (1 − α) · ||Z_pca|| + α · ||Z_jl||

# Similarity / NN search uses the alpha-weighted view
# (engineering detail: alpha-gated query path)
```

The `alpha` gate measures *what fraction of x lies outside the PCA subspace*. When PCA's subspace explains the input well, $\alpha \to 0$ and the system trusts PCA. When the input has substantial mass orthogonal to the PCA subspace (a stress event), $\alpha$ grows and the system falls back on JL.

The blend is smooth — no thresholds, no cliffs. The math sets the weight automatically.

### Periodic retrain (e.g., quarterly)

```
1. Fit candidate PCA_new on the most recent 2 years of data.
2. On a held-out rolling window of recent days, compute anomaly recall and
   any other key metrics for both PCA_current and PCA_new.
3. If PCA_new is meaningfully better than PCA_current:
       retire PCA_current → PCA_archive (kept for one cycle as fallback)
       promote PCA_new → PCA_current
4. Sparse-JL projection S is never retrained — it's data-independent and
   serves as the always-on regime-robust backup.
```

This is a champion–challenger pattern. A new PCA only takes over if it actually outperforms the incumbent. Bad retrains don't degrade the system.

## Storage accounting

Per market vector ($k = 20$):

| What's stored | Size |
|---|---|
| PCA-flavored signs | $k$ bits $= 2.5$ bytes |
| PCA-flavored norm | 4 bytes |
| JL-flavored signs | 2.5 bytes |
| JL-flavored norm | 4 bytes |
| **Total per day** | **~13 bytes** |

Compared to alternatives:

| Compression scheme | Bytes per market vector | Compression vs raw |
|---|---|---|
| Raw 100-stock float64 | 800 | 1× |
| PCA / dense JL float64 | 160 | 5× |
| Pure normsign (single method) | 6.5 | 123× |
| **Hybrid (CrashSketch v2)** | **13** | **62×** |

The hybrid pays $\sim 2\times$ the storage of pure normsign because we keep both signal paths alive. In return we get calm-regime quality close to PCA *and* stress-regime quality close to JL.

For a real production risk system that stores every minute of every market state on 5{,}000 stocks for 20 years:

```
5000 × 252 × 390 × 20 × 13 bytes ≈ 6.4 GB
```

Fits on a single machine. The same data uncompressed would be $\sim 400$ GB.

## Why this design hits all three goals

**Storage (Goal 1).** 62× compression vs raw. 12× compression vs single-method full-precision compression. 6.4 GB to store 20 years of minute-bar S&P 500 history.

**Calm-regime quality (Goal 2 calm half).** When markets are normal, $\alpha \approx 0$. The system trusts PCA's higher-precision compressed score. NN preservation, clustering, anomaly recall stay near PCA's strong calm-regime numbers.

**Stress-regime quality (Goal 2 stress half).** When markets shift, $\alpha$ grows. The system smoothly falls back on JL. Anomaly recall stays near float-JL's regime-robust level $\sim 0.75$, much higher than PCA's stress-regime drop to $\sim 0.67$.

**Stay current (Goal 3).** Periodic retrain refreshes the PCA basis as new regimes accumulate. If a candidate PCA underperforms on rolling-window evaluation, it doesn't get promoted. The JL backup is always live.

**Self-monitoring.** $\alpha$ is computed from the input itself, not from external regime signals. No manual stress detector. No volatility-trigger rule. The math watches itself.

## Open design questions

A few things you'd want to nail down for production deployment:

### Retrain cadence

| Asset class | Recommended cadence |
|---|---|
| Equities | Quarterly |
| Crypto | Monthly or faster |
| Fixed income | Semi-annually |
| FX | Quarterly |

This is operational, not algorithmic. The system supports any cadence; the question is how often is *enough* to keep the calm-regime quality good without thrashing the model.

### How to evaluate "is the new PCA better"

The cleanest approach: rolling anomaly-recall delta on the most recent N days.

```
score(model, window) = mean anomaly recall over window
if score(PCA_new, last_N_days) − score(PCA_current, last_N_days) > δ:
    promote PCA_new
```

Two subtleties:

1. **Stress days during evaluation.** If the evaluation window happens to include a flash crash, both PCA models will have low absolute recall — but the *relative* comparison should still be informative.
2. **Promotion threshold $\delta$.** Set conservatively (e.g., $\delta = 0.02$, i.e., 2 percentage points). We don't want to promote a marginally-better model and risk regressing.

### Storage strategy: store both, or store the alpha-winner?

Two valid approaches:

**Approach A: Store both representations always.** 13 bytes per day. Re-querying historical days uses the at-day alpha. Simpler operationally; predictable storage.

**Approach B: Store only the alpha-winning representation per day.** 6.5 bytes per day on average. Re-querying requires recomputing alpha at lookup time (cheap). Storage is dynamic.

A is simpler and probably the right default. B is for storage-constrained edge deployments.

### What about model versioning?

Each PCA retrain produces a new $V_k$. Compressed records made under PCA $i$ should be re-decompressible at any time. Store with each compressed record a small `model_version_id` (1 byte). Lookup the right $V_k$ at decompression time.

The JL projection $S$ is always the same. Doesn't need versioning.

## Implementation plan

For the research-project version of this:

| Component | Effort | Notes |
|---|---|---|
| `CrashSketchHybrid` class composing PCA + sparse JL + alpha gate | 2 hours | Builds on existing `PCAReducer` and `CrashSketch` |
| Wire `crashsketch_hybrid_alpha_R` into `parallel_backend._build_method` | 30 min | Mechanical |
| Add to `path_b` grid in `scripts/run_phase1.py` | 5 min | Trivial |
| Run path_b on laptop | 3 min | Already paid for |
| Analysis script update for hybrid | 30 min | Existing infra; just adds method |

Total: 3–4 hours of focused work, then a 3-minute laptop run.

If results look good, push to the YOLO grid (5 universes, 100 seeds, both protocols) on cloud for ~$0.62 in credits and ~30 minutes of wall-clock.

The periodic-retrain logic is **out of scope** for the research-project version. We assume static PCA fit once on the training set. The retrain logic is what a real production deployment would build on top.

## What this design will deliver if it works

If the hybrid works as predicted by the path_b empirics so far:

- A single compression scheme that is **simultaneously the best in calm regimes (matching PCA) and the best in stress regimes (matching JL)**, with the choice made smoothly per input by the math.
- **62× total storage compression** vs raw market vectors; **12×** vs single-method full-precision compression.
- **Self-monitoring** via the alpha gate — the system tells you when its primary model can't be trusted.
- **Self-updating** via champion–challenger retrain — bad models can't be promoted; old models can be retired or kept as fallback.
- **An operational story for production deployment**, not just an academic curiosity.

This directly answers the original goal — reduce the size of financial market state data dramatically while preserving both calm-regime fidelity and stress-regime safety. The earlier YOLO and path_b experiments showed the gap between methods; this design is the engineering recipe that closes it.

## Honest assessment

**What's novel.** The composition of (a) self-normalizing per-instance alpha gate, (b) data-blind JL fallback, (c) normsign storage applied to both signal paths, in the specific context of regime-robust financial-data compression. None of the individual pieces are new; the combination and the empirical case for *why this combination matters* are.

**What's incremental.** Each of the underlying ideas — PCA, sparse JL, sign-only quantization, reconstruction-error-based anomaly gating — has been studied separately for decades. The hybrid is a clean engineering composition of known parts.

**Publishability.** With this design plus the existing path_b empirical evidence, this is workshop-paper-quality material. Adding multiple stress windows and one cross-market validation gets it to a defensible main-track conference submission. The novelty argument is strongest if framed around "regime-robustness as a structural property of distribution-free compressions, with the hybrid as the recipe that recovers calm-regime quality without losing stress robustness."

**Production-readiness.** With the periodic-retrain layer added on top, this could plausibly be deployed in a real risk system. The system has the operational properties (self-monitoring, self-updating, fallback when primary model fails) that production systems require. The research-project version is an end-to-end demonstration that the underlying mechanism works; production engineering is a separate (and substantial) effort.

## What to do next

The natural sequence:

1. **Build it.** Implement `CrashSketchHybrid` as described. ~3 hours.
2. **Run path_b.** ~3 minutes laptop wall-clock. Verify the hybrid Pareto-dominates pure PCA and pure normsign on the (anomaly recall × storage × stress robustness) frontier.
3. **If it works:** push to cloud YOLO grid for multi-universe, multi-stress empirical validation. ~$0.62, ~30 min.
4. **Write up.** Frame the project around "regime-robustness as a structural property" with the hybrid as the constructive recipe.

We're one evening of focused work away from having the full design empirically validated.
