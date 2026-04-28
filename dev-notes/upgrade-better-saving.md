# Upgrade Plan: Comprehensive Artifact Saving

## Goal

Persist every intermediate from every experiment run so that:

1. New metrics can be computed *without* re-fitting projections.
2. Specific runs can be re-loaded for inspection in notebooks.
3. Audit trail is complete — every figure/table in the final report can be traced to specific saved arrays.

This is the cheap-disk-and-cloud-credits version of "save everything you might want later." Total estimated disk use across the full Phase 1 grid: ~5–10 GB.

## Current state (Phase 0)

What we save today:

| Artifact | Path | Persisted? |
|---|---|---|
| Summary metrics row | `results/phase0_smoke_test.csv` | ✅ |
| Preprocessing report | `data/processed/preprocessing_report.json` | ✅ |
| Asset list | `data/processed/asset_list.csv` | ✅ |
| Compressed Z matrices (one per method) | — | ❌ |
| Projection matrices (A for dense JL, S for sparse JL, V_k for PCA) | — | ❌ |
| Standardization params (μ, σ vectors) | — | ❌ |
| Train/test row indices | — | ❌ |
| Pair sample indices used in distance distortion | — | ❌ |
| Random seeds (used vs. requested) | implicit in script constants | ⚠️ |

## Target state

Add the following artifacts per experiment run:

### Per-config artifacts

For each `(universe, slice, method, k, s, seed)` config:

```
results/phase1/runs/{universe}/{slice}/{method}/k{k}_s{s}_seed{seed}/
├── compressed.npz           # Z_train and Z_test compressed arrays
├── projection.npz           # A / S / V_k (whatever is method-specific)
├── metrics.json             # all metrics computed for this config
└── run_meta.json            # config dict, timestamps, runtime, git sha
```

Notes on schema:
- `compressed.npz`: contains `Z_train`, `Z_test`, plus `train_dates` and `test_dates` (DatetimeIndex serialized as int64 epoch).
- `projection.npz`: method-specific. PCA → `V_k`, `mean_train`, `singular_values`. Dense JL → `A` (full N × k matrix). Sparse JL → `S` saved as scipy sparse `.npz` (csr_matrix dump for storage efficiency).
- `metrics.json`: nested by metric name; each metric stores its summary stats and any auxiliary arrays.
- `run_meta.json`: full `ExperimentConfig` as a dict, plus `timestamp_utc`, `runtime_seconds`, `git_sha`, `python_version`.

### Per-data-prep artifacts (one set per universe)

```
data/processed/{universe}/
├── X_train_standardized.npz     # the standardized train matrix
├── X_test_standardized.npz      # the standardized test matrix
├── standardization.npz          # mu, sigma vectors
├── split_indices.json           # which row indices are train, which are test
└── asset_list.csv               # final ticker list
```

This means the standardized matrices are saved once per universe and shared across all method/k/seed combinations — saves disk and ensures consistency.

### Pair-sample artifacts (one per slice)

```
results/phase1/pair_samples/{slice}/{n_pairs}_seed{pair_seed}.npz
```

A single (n_pairs, 2) integer array of test-row indices. Shared across all methods so distance distortion comparisons are apples-to-apples.

## Implementation steps

### Step 1 — Add artifact-saving helpers (`src/utils.py`)

```python
def save_npz_dict(path, **arrays): ...          # wrapper around np.savez_compressed
def load_npz_dict(path) -> dict: ...
def save_json(path, obj): ...
def load_json(path) -> dict: ...
def get_git_sha() -> str: ...                   # via subprocess; "dirty" if uncommitted
def make_run_dir(base, config) -> Path: ...     # creates and returns the per-config dir
```

### Step 2 — Add `save_artifacts` to method classes (`src/pca_baseline.py`, `src/projections.py`)

Each method gains:

```python
def save(self, path: Path) -> None:
    """Save the fit state (projection matrix + metadata) to path."""
```

So `pca.save(...)` writes `V_k`, `mean_train`, `singular_values` into `projection.npz`. `dense_jl.save(...)` writes `A`. `sparse_jl.save(...)` writes the scipy sparse `S` plus `s` and `seed` for traceability.

Plus a class-level `load(path) -> Reducer` factory so `PCAReducer.load(p)` reconstructs an already-fit reducer for use on new data.

### Step 3 — Update `scripts/run_phase0_smoke_test.py` to persist everything

Add after each `fit_transform` call:

```python
run_dir = make_run_dir(RESULTS_DIR / "phase0/runs", config)
save_npz_dict(run_dir / "compressed.npz", Z_test=Z_test, ...)
method.save(run_dir / "projection.npz")
save_json(run_dir / "run_meta.json", config_to_dict(config))
save_json(run_dir / "metrics.json", metric_results)
```

### Step 4 — Add a re-evaluator script

`scripts/recompute_metrics.py` — given a path to a phase1 run dir, load `compressed.npz` and re-run all metrics. This is the payoff: adding a new metric never requires re-fitting.

### Step 5 — Update `dev-notes/architecture.md` and `recreate-results.md`

Document the new file layout so future-you (or Vishal, or a reviewer) can find anything in the artifact tree.

## File-format choices

| Artifact type | Format | Why |
|---|---|---|
| Compressed matrices | `.npz` (compressed) | Numeric, dense, small; loads in 1 line; ~3× smaller than CSV |
| Sparse projection (Sparse JL's S) | `scipy.sparse.save_npz` | Native sparse format; preserves structure |
| Pair sample indices | `.npz` | Integer array; tiny |
| Run metadata | `.json` | Human-readable, diff-able, easy to grep |
| Metric results | `.json` | Same; easy to aggregate across runs |

We do *not* use Parquet here — Parquet is great for tabular long-format data but overkill for fixed-shape numeric matrices. NPZ is simpler and faster.

## Disk budget

For Phase 1 worst case:

```
3 universes × 3 slices × 4 methods × 6 k × 3 s × 50 seeds = ~32K configs
                       (constraint: s ≤ k, deterministic methods are 1-seed)

Each config:
  compressed.npz (Z_train + Z_test, k=100 max):
    (1936 + 831) × 100 × 8 bytes = ~2.2 MB uncompressed; ~0.7 MB compressed
  projection.npz: <0.1 MB
  metrics.json: <0.01 MB
  run_meta.json: <0.01 MB
  ≈ 0.8 MB per config

Total: 32K × 0.8 MB ≈ 25 GB worst-case.
Realistic (with constraint pruning): ~10–15 GB.
```

15 GB is fine on a laptop and trivial on cloud storage. The 50-seed assumption is the dominant factor; if we drop to the proposal's 10 seeds, the total is ~3 GB.

## Estimated effort

| Step | Hours |
|---|---|
| 1. utils.py helpers | 0.5 |
| 2. method `save()` / `load()` | 1.0 |
| 3. wire into smoke test | 0.5 |
| 4. recompute_metrics.py | 0.5 |
| 5. doc updates | 0.5 |
| **Total** | **~3 hours** |

## Acceptance criteria

This upgrade is "done" when:

1. Running `scripts/run_phase0_smoke_test.py` produces a per-method directory tree under `results/phase0/runs/...` with all artifacts described above.
2. Running `scripts/recompute_metrics.py results/phase0/runs/.../pca__k20__seed0/` reproduces the same metrics row from the saved compressed matrices, **without** re-fitting PCA.
3. `dev-notes/architecture.md` lists the new file layout.
4. `dev-notes/recreate-results.md` shows how to load saved artifacts in a Jupyter notebook.

## What this enables (the payoff)

- Add a new metric next month? Run `recompute_metrics.py` over the saved tree. Zero re-fitting.
- Notebook exploration? `Z = np.load(...)['Z_test']` and you're off.
- Reviewer asks "exactly which numbers produced figure 17?" — point at the path.
- Re-apply a PCA run to fresh data later? `PCAReducer.load(...)` + `.transform(new_X)`.
