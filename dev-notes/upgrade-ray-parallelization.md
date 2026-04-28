# Upgrade Plan: Ray-Based Parallel Execution

## Goal

Move from "one-config-at-a-time sequential" to "embarrassingly-parallel grid execution" across method × k × s × seed × slice. Run on local multiprocessing for development; switch to Ray on GCP for the full grid.

## Why Ray (over multiprocessing or joblib)

The proposal §9.2 already calls this out, but in practical terms:

- **Shared object store via `ray.put`** — the standardized data matrix (a few MB) is referenced by all workers, not copied per process. Multiprocessing requires re-pickling per worker.
- **Cluster-friendly** — same code runs on a single 16-vCPU VM or on a multi-node cluster. We'll likely stay on a single node, but Ray makes that decision reversible.
- **Better introspection** — Ray Dashboard shows worker state, failures, and timing without us building it ourselves.
- **Cleaner failure model** — one worker crashing on a bad config doesn't take down the rest.

For a grid this size (low thousands of configs, each sub-second), `joblib` would also work. Ray is the right call because it matches the proposal's design and makes the upgrade path to multi-node painless if we ever want it.

## Current state vs. target state

### Current (Phase 0)

`scripts/run_phase0_smoke_test.py` is a linear script with hardcoded constants. One method run per script invocation. Scales by hand.

### Target (Phase 1)

```python
# pseudocode
configs = generate_experiment_grid(
    universes=["top_100", "top_200", "all_survivors"],
    slices=["full_incl_covid", "full_excl_covid", "covid_only"],
    methods=["raw", "dense_jl", "sparse_jl", "pca"],
    k_values=[5, 10, 20, 30, 50, 100],
    s_values=[1, 3, 5],
    seeds=range(50),                       # 50, not 10 — we have credits
)

results_df = run_experiment_grid(configs, backend="ray", num_cpus=16)
```

`results_df` is a long-format DataFrame: one row per (config, metric). Saves to `results/phase1/all_metrics.csv`.

## Design

### `src/experiment_config.py`

```python
@dataclass(frozen=True)
class ExperimentConfig:
    universe: str           # "top_100" | "top_200" | "all_survivors"
    slice: str              # "full_incl_covid" | "full_excl_covid" | "covid_only" | regimes...
    method: str             # "raw" | "dense_jl" | "sparse_jl" | "pca"
    k: int
    s: int | None           # only for sparse_jl; None otherwise
    seed: int               # only for randomized methods; 0 for deterministic
    metric_bundle: str = "all_phase1"   # which metrics to compute
```

`frozen=True` so configs are hashable and can be used as dict keys / cached.

A helper `generate_experiment_grid(...)` produces the list, applying constraints:
- For PCA and raw: only one seed (deterministic), so `seed=0`.
- For sparse JL: only `s ≤ k` combinations.
- Optionally filter to "smoke test" subsets for quick local runs.

### `src/parallel_backend.py`

```python
def run_experiment_grid(
    configs: list[ExperimentConfig],
    prepared_data: dict[str, dict],   # keyed by universe -> {X_train, X_test, ...}
    backend: str = "local",
    num_cpus: int | None = None,
    save_artifacts: bool = True,
) -> pd.DataFrame:
    if backend == "local":
        return _run_local(configs, prepared_data, save_artifacts)
    elif backend == "ray":
        return _run_ray(configs, prepared_data, num_cpus, save_artifacts)
    else:
        raise ValueError(...)


def _run_single_config(config, data, save_artifacts) -> dict:
    """Worker function: runs one config, returns one row dict."""
    method = build_method(config)               # PCA, DenseJL, SparseJL, or raw
    method.fit(data["X_train"])
    Z_test = method.transform(data["X_test"])

    metrics = compute_all_metrics(
        Z_raw=data["X_test"], Z_compressed=Z_test, slice_indices=data["slice_idx"]
    )

    if save_artifacts:
        run_dir = make_run_dir(config)
        save_compressed(run_dir, Z_test)
        method.save(run_dir / "projection.npz")
        save_json(run_dir / "metrics.json", metrics)
        save_json(run_dir / "run_meta.json", config_to_dict(config))

    return {**config_to_dict(config), **flatten_metrics(metrics)}
```

### Ray backend

```python
import ray

def _run_ray(configs, prepared_data, num_cpus, save_artifacts):
    ray.init(num_cpus=num_cpus, ignore_reinit_error=True)
    # Put each universe's prepared data into the object store ONCE
    data_refs = {u: ray.put(d) for u, d in prepared_data.items()}

    @ray.remote
    def remote_run(config_dict, data_ref, save_artifacts):
        config = ExperimentConfig(**config_dict)
        data = ray.get(data_ref)        # zero-copy on same node
        return _run_single_config(config, data, save_artifacts)

    futures = [
        remote_run.remote(
            asdict(c), data_refs[c.universe], save_artifacts
        )
        for c in configs
    ]
    results = ray.get(futures)
    return pd.DataFrame(results)
```

Note the `ray.put` for prepared data — with 50 seeds × multiple slices, we'd pay a serious cost re-pickling X_train and X_test if we passed them by value.

### Local backend (debug)

Same `_run_single_config` worker, but called sequentially in a loop. Used when debugging a specific failing config.

```python
def _run_local(configs, prepared_data, save_artifacts):
    rows = []
    for c in configs:
        rows.append(_run_single_config(c, prepared_data[c.universe], save_artifacts))
    return pd.DataFrame(rows)
```

We don't bother with `multiprocessing.Pool` — Ray is a better local backend for testing the parallel path.

## Implementation steps

### Step 1 — `src/experiment_config.py`
- `ExperimentConfig` dataclass
- `generate_experiment_grid(...)` with constraint pruning
- `config_to_dict` / `dict_to_config` utilities

### Step 2 — `src/parallel_backend.py`
- `_run_single_config` worker
- `_run_local` and `_run_ray` backends
- `run_experiment_grid` dispatcher

### Step 3 — Wire into a new script
`scripts/run_phase1.py` — generates the full grid, calls `run_experiment_grid`, saves the result DataFrame.

### Step 4 — Test on local before cloud
- Run on a tiny subset (e.g., top_100, k ∈ {20}, 3 seeds, 1 method) with `backend="local"`
- Then same subset with `backend="ray"` to confirm Ray path works
- Then expand to full grid

### Step 5 — Document
Update `dev-notes/architecture.md` with the runner / backend modules and `dev-notes/recreate-results.md` with the new entry point.

## Reproducibility considerations

This is where the proposal §9.7 risks bite:

- **Random seeds** must live in every config and result row. Done via the `seed` field on `ExperimentConfig`.
- **Worker order** doesn't affect results because each worker uses only its own config-derived seed. Order-of-completion is non-deterministic; order-of-content is deterministic.
- **Race conditions**: workers never write to shared paths. Each worker writes to its own `run_dir` keyed by config hash. Central process aggregates the DataFrame and writes the long-form CSV.
- **Environment drift**: pin `numpy`, `scipy`, `sklearn`, `ray` exact versions in `pyproject.toml` / `requirements.txt`.

## Estimated effort

| Step | Hours |
|---|---|
| 1. ExperimentConfig + grid generator | 1.0 |
| 2. parallel_backend.py (local + Ray) | 2.0 |
| 3. scripts/run_phase1.py | 0.5 |
| 4. local + Ray testing | 1.0 |
| 5. doc updates | 0.5 |
| **Total** | **~5 hours** |

## Acceptance criteria

This upgrade is "done" when:

1. `python scripts/run_phase1.py --backend local --subset smoke` runs the full smoke-grid sequentially and produces `results/phase1/all_metrics.csv` plus per-config artifacts.
2. Same command with `--backend ray` produces an identical CSV (modulo row order, which we sort consistently before writing).
3. Ray dashboard shows workers running in parallel.
4. `dev-notes/architecture.md` documents the runner and backend modules.

## Compute model on GCP

For Phase 1 with 50 seeds and the planned config matrix (see [`hpc-experiments.md`](hpc-experiments.md)):

- **Single config runtime:** 0.1–1 sec depending on $k$ and metric set
- **Total configs:** ~30,000 after constraint pruning
- **On 16 vCPU (e2-standard-16):** ~30 minutes wall clock
- **On 32 vCPU (e2-standard-32):** ~15 minutes wall clock

So Phase 1 fits comfortably in a single VM session. We'd only need multi-node Ray if we scale much further (e.g., 500 seeds, or much bigger universes).

## What this enables

Once Ray is wired, you can:

- Add new configs to the grid by editing one function
- Re-run any subset by filtering `configs`
- Move to a bigger VM by changing `num_cpus`
- Eventually move to a multi-node cluster with no code changes
- Plot speedup curves to put in the runtime/sparsity figure (proposal §7.5)
