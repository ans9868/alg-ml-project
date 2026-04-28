"""ExperimentConfig dataclass and grid generator (proposal §9.3)."""
from __future__ import annotations

from dataclasses import dataclass, field
from itertools import product
from typing import Iterable, Optional


@dataclass(frozen=True)
class ExperimentConfig:
    universe: str            # "top_100" | "top_200" | "all_survivors" | ...
    slice: str               # "full_incl_covid" | "full_excl_covid" | "covid_only"
    method: str              # "raw" | "dense_jl" | "sparse_jl" | "pca"
    k: int
    s: Optional[int] = None  # only meaningful for sparse_jl; None otherwise
    seed: int = 0            # only meaningful for randomized methods
    train_frac: float = 0.7
    n_pairs: int = 50_000

    def short_name(self) -> str:
        s_part = f"s{self.s}" if self.s is not None else "sNA"
        return f"{self.method}__k{self.k}__{s_part}__seed{self.seed}"


def generate_experiment_grid(
    universes: Iterable[str],
    slices: Iterable[str],
    methods: Iterable[str],
    k_values: Iterable[int],
    s_values: Iterable[int] = (1, 3, 5),
    seeds: Iterable[int] = range(10),
    train_frac: float = 0.7,
    n_pairs: int = 50_000,
) -> list[ExperimentConfig]:
    """Generate the full grid with constraints applied:

    - PCA and raw are deterministic -> only seed=seeds[0]
    - Sparse JL: only valid (k, s) where s <= k
    - Raw is independent of k (we treat each k as a separate config so the
      grid is rectangular, but raw doesn't actually compress)
    """
    seeds = list(seeds)
    s_values = list(s_values)
    k_values = list(k_values)
    methods = list(methods)
    universes = list(universes)
    slices = list(slices)

    configs: list[ExperimentConfig] = []
    for universe, slice_, method, k in product(universes, slices, methods, k_values):
        if method == "sparse_jl":
            for s, seed in product(s_values, seeds):
                if s > k:
                    continue
                configs.append(
                    ExperimentConfig(
                        universe=universe, slice=slice_, method=method,
                        k=k, s=s, seed=seed,
                        train_frac=train_frac, n_pairs=n_pairs,
                    )
                )
        elif method == "dense_jl":
            for seed in seeds:
                configs.append(
                    ExperimentConfig(
                        universe=universe, slice=slice_, method=method,
                        k=k, s=None, seed=seed,
                        train_frac=train_frac, n_pairs=n_pairs,
                    )
                )
        elif method in ("pca", "raw"):
            # deterministic -> single seed
            configs.append(
                ExperimentConfig(
                    universe=universe, slice=slice_, method=method,
                    k=k, s=None, seed=seeds[0],
                    train_frac=train_frac, n_pairs=n_pairs,
                )
            )
        else:
            raise ValueError(f"Unknown method {method!r}")

    return configs
