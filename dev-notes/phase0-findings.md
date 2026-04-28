# Phase 0 Empirical Findings

Phase 0 is the laptop smoke test: validate the data pipeline and the method/metric architecture before scaling to the full grid in Phase 1. This note records what we learned that wasn't predicted by the proposal — the kind of things that matter for the next `.tex` revision.

## Finding 1: The Kaggle dataset is broken for ~127 large-cap tickers

**Severity:** blocking.
**Action taken:** pivoted to yfinance (already named as the fallback in proposal §4.2).
**Detail:** see [data-source-comparison.md](data-source-comparison.md).

127 of the largest, oldest stocks on the planet (AAPL, GOOGL, JPM, XOM, JNJ, KO, IBM, BAC, HD, BRK-B, WMT, …) have **zero valid rows** in the entire `andrewmvd/sp-500-stocks` Kaggle dataset. The probable cause is silent failures in the uploader's yfinance scraper that were never backfilled.

**Implication for the .tex (v6 amendment):** swap yfinance from "fallback" to "primary"; document the Kaggle finding in Limitations.

## Finding 2: Stage 1 extreme-return threshold of 0.5 was too tight

**Severity:** wrong design choice; would have biased the universe.
**Action taken:** raised threshold from 0.5 to 1.0 in `scripts/run_data_pipeline.py`.

The proposal §4.3 Stage 1d originally specified:

> Drop a ticker if any single-day return $|r| > 0.5$ that does not correspond to a known corporate action (treated as a data error).

In practice, 10 tickers were caught by this filter at $|r| > 0.5$:

| Ticker | Date | Reason |
|---|---|---|
| OXY | 2020-03-09 | COVID + Saudi-Russia oil price war (real -50%+ day) |
| APA | 2020-03-09 | same |
| FANG | 2020-03-09 | same |
| TRGP | 2020-03-09 | same |
| PCG | 2019-01-14, 2019-01-24 | Pacific Gas & Electric bankruptcy filing (real) |
| EPAM | 2022-02-28 | Russia exposure on Ukraine invasion (real) |
| GL | 2024-04-11 | Globe Life short-seller report (real) |
| DXCM | 2024-07-26 | Dexcom earnings disaster (real) |
| SMCI | 2018-10-04 | Bloomberg "spy chip" article (real) |
| BLDR | 2015-04-13 | secondary offering / merger announcement (real) |

These are **real historical events**, not data errors. A 0.5 log return is a -39%/+65% move, which is rare but happens on big-news days.

We raised the threshold to 1.0 (= -63%/+172% move), which catches only physically-implausible data errors (e.g., a botched split adjustment producing a 1000% one-day return).

**Implication for the .tex:** §4.3 Stage 1d should be updated to use the 1.0 threshold and the rationale.

## Finding 3: yfinance fails on 11 of 502 tickers

**Severity:** minor.
**Action taken:** documented in `data/raw/yfinance_failed.txt`; pipeline drops them gracefully.

11 tickers from the `sp500_companies.csv` list could not be downloaded (yfinance returns "possibly delisted; no timezone found" or similar):

```
ANSS, DAY, DFS, FI, HES, IPG, JNPR, K, MMC, PARA, WBA
```

Inspection: ANSS, DFS, HES, JNPR, PARA are real M&A or pending-acquisition cases (correctly excluded from a survivor universe). DAY, K are recent ticker renames. FI, IPG, MMC, WBA are large continuously-traded companies that may have been transient yfinance errors — worth re-trying in a separate session.

**Implication for the .tex:** mention the yfinance failure rate (~2%) in the data-cleaning section.

## Finding 4: 452 survivors after filtering

**Severity:** informational.

Final survivor count from the elimination pipeline:

```
Candidate pool:        491   (502 from companies metadata, 11 failed yfinance)
Stage 1 drops:          37   (37 NaN, 0 delisted, 0 low coverage, 0 extreme returns at threshold 1.0)
Stage 2 drops:           2   (AMCR and SW: >10 consecutive zero-return days)
Survivors:             452
```

This comfortably supports the three slices in the proposal:
- `top_100` ✓
- `top_200` ✓
- `all_survivors` = 452 (vs the original "top 200 if pipeline is robust" wording in the proposal — we now have more flexibility)

**Implication for the .tex:** confirm that all_survivors at $N \approx 450$ is a viable third slice for sensitivity analysis.

## Finding 5: Strong low-rank structure in the data

**Severity:** informational; supports the proposal's main hypothesis.

From the smoke test on `top_100`: PCA captures **71.1% of variance** in 20 components out of 100. This is a strong indicator that financial returns lie close to a low-rank factor space — the central modeling assumption of proposal §3.2.

**Implication:** the proposal's hypothesis is empirically plausible. PCA is likely to dominate sparse JL at small $k$ (as predicted in proposal §12.1 Expected Result 1).

## Finding 6: All four methods produce sensible, theory-consistent numbers

**Severity:** informational; architecture validation.

Single-seed smoke test ($N=100$, $k=20$, seed=0):

| Method | mean \|ρ-1\| | mean ρ | nnz |
|---|---|---|---|
| raw | 0.0000 | 1.0000 | (identity) |
| PCA | 0.2580 | 0.7420 | data-adaptive |
| Dense JL | 0.1242 | 1.0082 | 2,000 |
| Sparse JL (s=3) | 0.1147 | 0.9756 | 300 |

Theory predictions, all confirmed:
- **Dense JL: mean ρ ≈ 1** — JL lemma works.
- **Sparse JL: mean ρ ≈ 1**, comparable distortion to dense JL with **6.7× fewer parameters** — the SOSA paper's central claim.
- **PCA: mean ρ < 1** — preserves variance, not pairwise distances; under-estimates by 26%.

The architecture is producing correct outputs.

**Implication for the .tex:** none yet — these are single-seed numbers. Phase 1 will produce real results.

## Open items (for the next .tex revision)

1. Swap data source: yfinance → primary, Kaggle → fallback (or removed).
2. Update Stage 1d extreme-return threshold: 0.5 → 1.0.
3. Note the 11 yfinance ticker failures in the data section.
4. Update the §2 configuration table accordingly.
5. Consider whether to keep `all_survivors` as a published slice now that we know it's $\sim 450$.

These will be batched into proposal v6 once Phase 1 is underway.
