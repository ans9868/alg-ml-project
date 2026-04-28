# Metrics Explained

Each of the 5 metrics from proposal §7, written for a non-academic audience (think: young technologist on a trading desk). For every metric: the math, what it really means in plain English, and why a practitioner would care.

> Status legend: ✅ implemented in Phase 0 — 🚧 Phase 1 work.

---

## 1. Pairwise Distance Distortion ✅

**The math.**
For two trading days $i$ and $j$:

$$
d_{ij} = \|X_i - X_j\|_2 \quad\text{(raw distance)}
$$

$$
\hat{d}_{ij} = \|Z_i - Z_j\|_2 \quad\text{(compressed distance)}
$$

$$
\rho_{ij} = \frac{\hat{d}_{ij}}{d_{ij}} \quad\text{(distortion ratio; perfect = 1)}
$$

We report the mean of $|\rho - 1|$, the median, the 95th percentile, and the signed mean $\rho$.

**In plain English.**
> Pick any two trading days. Measure how "different" they look across all 100 stocks. Now compress those days down to 20 numbers each, and measure again. If the second number is close to the first, the compression preserved the geometry between days.

**Why a trader cares.**
This is the foundation. Almost every "find similar past days" model depends on distance. Risk overlays that say "today looks like 2015-08-24, here is what happened next" — that is a distance computation. If $|\rho - 1| = 0.30$, your "lookalike" recommendations are 30% off, which means your top-10 list of similar days is mostly garbage.

**Phase 0 result on top_100, k=20:**
- Dense JL: mean $\rho = 1.008$ (excellent — JL lemma working as advertised)
- Sparse JL ($s = 3$): mean $\rho = 0.976$ (excellent, with 6.7× fewer parameters)
- PCA: mean $\rho = 0.742$ (systematic 26% under-estimation, by design — PCA preserves variance, not distances)

---

## 2. Nearest-Neighbor Preservation 🚧

**The math.**
For each test day $i$, find its $m$ nearest neighbors in raw space, $\mathcal{N}^{\text{raw}}_m(i)$, and in compressed space, $\mathcal{N}^{\text{comp}}_m(i)$.

$$
\text{NN-Overlap}_m(i) = \frac{|\mathcal{N}^{\text{raw}}_m(i) \cap \mathcal{N}^{\text{comp}}_m(i)|}{m}
$$

Average across all test days. Report at $m = 5$ and $m = 10$.

**In plain English.**
> For today, find the 5 most similar past trading days using the full 100-stock vector. Now do the same using only the compressed 20-number vector. How many of those 5 days are the same?
>
> 5 out of 5 = perfect. 1 out of 5 = your compressed system is mostly finding *different* "similar days."

**Why a trader cares.**
This is what *all* kNN-style systems care about — regime classification, scenario lookups, vol forecasting based on historical analogues. Where distance distortion (#1) tells you whether distances are stretched on average, this tells you whether the *order* (which days are closer to which) survives.

**A method can score badly on distance distortion but perfectly here** — if all distances shrink by the same factor, ordering is preserved. PCA is likely to win on this metric even though it loses on distance distortion. That nuance matters when picking a method for your particular use case.

---

## 3. Clustering Stability (Adjusted Rand Index) 🚧

**The math.**
Run KMeans on raw test data, getting labels $y^{\text{raw}}$. Run KMeans on compressed test data, getting $y^{\text{comp}}$. Compare via Adjusted Rand Index:

$$
\text{ARI} = \frac{\text{RI} - \mathbb{E}[\text{RI}]}{\max(\text{RI}) - \mathbb{E}[\text{RI}]} \in [-1, 1]
$$

where RI is the Rand Index — the fraction of pair decisions ("are these two days in the same cluster?") that agree between the two labelings, with the expected agreement under random labeling subtracted off.

- ARI = 1: clusterings are identical
- ARI = 0: agreement is no better than random
- ARI < 0: worse than random

We test at cluster counts $C \in \{3, 5, 8\}$.

**In plain English.**
> Group historical days into "market regimes" (say 3 = calm / medium / volatile). Do it twice — once on raw 100-stock data, once on compressed 20-number data. Are the same days grouped together both times?
>
> Note: cluster *labels* don't matter (whether a day is "cluster 1" or "cluster 3"). What matters is whether days that are grouped together in one are grouped together in the other.

**Why a trader cares.**
Tons of strategies are conditional on regime: vol-targeting, dispersion trading, mean-reversion-vs-momentum switches. If your regime detector breaks under compression, your entire conditional logic disintegrates. This is the metric that says "compression is safe to use as a preprocessing step for my regime classifier."

**Why ARI specifically and not naive accuracy:** clustering algorithms produce labels in arbitrary order. A direct comparison of label strings would give 0% match between perfectly identical clusterings just because the labels rotated. ARI is invariant to label permutation — it asks the right question.

---

## 4. Anomaly Recall (Unusual Market Day Preservation) 🚧

**The math.**
Anomaly score in raw space:

$$
a_i = \|X_i\|_2
$$

A day is "unusual" if its anomaly score is in the top 5%:

$$
\mathcal{A}_{\text{raw}} = \{i : a_i \text{ in top 5\%}\}
$$

Compute the same in compressed space:

$$
\hat{a}_i = \|Z_i\|_2, \quad \mathcal{A}_{\text{comp}} = \{i : \hat{a}_i \text{ in top 5\%}\}
$$

Then:

$$
\text{Recall@5\%} = \frac{|\mathcal{A}_{\text{raw}} \cap \mathcal{A}_{\text{comp}}|}{|\mathcal{A}_{\text{raw}}|}
$$

We also report precision@5%, top-10 unusual-day overlap, and anomaly-score correlation.

**In plain English.**
> The 5% most extreme historical days — COVID crash, flash crash, fed shocks, whatever made markets go nuts that day. Does the compressed system still flag them as extreme?
>
> Recall = 0.85 means we catch 85% of the truly unusual days. Recall = 0.40 means we miss most of them.

**Why a trader cares.**
This is *the* metric that matters for risk and kill-switches. Compression is fine in calm markets — every day looks roughly the same. The whole point of detecting unusual days is to flip the system into "be careful" mode when something weird happens. If your compressed anomaly score doesn't fire on March 16, 2020 (S&P −12% on a Monday), your risk system is asleep at exactly the moment it shouldn't be.

The proposal explicitly slices this metric to **COVID-only**, because that's the most demanding stress test in the 2014–2024 window.

---

## 5. Runtime and Sparsity 🚧

**The math.**
Not a "metric" in the statistical sense — these are infrastructure measurements:

- **Projection construction time:** wall-clock seconds to build the projection matrix (or fit PCA on training data)
- **Transform runtime:** seconds to compute $Z = X \cdot M$
- **Nonzero count (nnz):** how many entries of the projection matrix are nonzero
  - Dense Gaussian JL: $N \cdot k$ (every entry is nonzero)
  - Sparse JL: $N \cdot s$ (typically much less; with $N=100, k=20, s=3$ this is $300$ vs $2{,}000$, a 6.7× reduction)
  - PCA: $N \cdot k$ (the top-$k$ singular vectors are dense)
- **Memory footprint:** bytes used by the projection matrix
- **End-to-end time:** total wall clock for one compression pass

**In plain English.**
> How fast and small is each method? At inference time (when you're actually using the model in production), how long does it take to compress one trading day's vector? How much memory does the projection matrix take up?

**Why a trader cares.**
- **HFT / market-making desks** measure latency in microseconds. A 10× faster method that is only 5% less accurate may be the right production call.
- **Streaming/realtime pipelines** care about memory pressure. Sparse JL with 300 nonzeros vs dense JL with 2000 is a 6.7× reduction in cache pressure — sometimes that is the entire reason to pick sparse.
- **Backtest infrastructure** that runs the same compression millions of times across simulations cares about cumulative wall time.

This is the metric that says "even if sparse JL is technically less accurate than PCA, sparse JL might be the right *production* choice because it's 50× cheaper to deploy."

---

## How they fit together (the trader's decision tree)

| If you care about… | Look at metric | Method most likely to win |
|---|---|---|
| Are similar days still similar? | Distance distortion (#1) | Dense JL or sparse JL |
| Will my "find lookalike days" model still work? | Nearest-neighbor preservation (#2) | Probably PCA |
| Will my regime classifier still work? | Clustering stability (#3) | Probably PCA |
| Will my risk overlay still detect unusual days? | Anomaly recall (#4), especially COVID-only | Open question — likely depends on $k$ |
| Can I deploy this in latency-sensitive production? | Runtime & sparsity (#5) | Sparse JL |

A method can win one and lose another. The whole point of the Phase 1 grid is to map which method wins which metric, at which compression ratio, and under which market conditions. That's the deliverable that turns "compression worked" into "use sparse JL for X, PCA for Y, dense JL only when…" — actionable guidance rather than a single number.

---

## Cross-references

- Source of truth for the math: [`AMLDS_Project_proposal-5.tex`](../AMLDS_Project_proposal-5.tex) §7 (and §10 for the regime-conditional version).
- Phase 0 implementation of metric #1: [`src/metrics.py`](../src/metrics.py).
- Phase 0 results: [`results/phase0_smoke_test.csv`](../results/phase0_smoke_test.csv) and [`report/preliminary_results.pdf`](../report/preliminary_results.pdf).
