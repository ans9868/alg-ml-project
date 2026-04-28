"""ExperimentConfig dataclass and grid generator (proposal §9.3)."""
from __future__ import annotations

from dataclasses import dataclass, field
from itertools import product
from typing import Iterable, Optional


@dataclass(frozen=True)
class ExperimentConfig:
    universe: str            # "top_100" | "top_200" | "top_300" | "top_400" | "all_survivors" | ...
    protocol: str            # "chrono70_30" or "preCOVID" (proposal §5.5)
    slice: str               # interpretation depends on protocol
    method: str              # "raw" | "dense_jl" | "sparse_jl" | "pca"
    k: int
    s: Optional[int] = None  # only meaningful for sparse_jl; None otherwise
    seed: int = 0            # only meaningful for randomized methods
    train_frac: float = 0.7  # only used for chrono70_30
    n_pairs: int = 50_000

    def short_name(self) -> str:
        s_part = f"s{self.s}" if self.s is not None else "sNA"
        return f"{self.protocol}__{self.slice}__{self.method}__k{self.k}__{s_part}__seed{self.seed}"


def generate_experiment_grid(
    universes: Iterable[str],
    protocol_slices: Iterable[tuple[str, Iterable[str]]],   # [(protocol, [slices...])]
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
    - Each (protocol, slice) pair produces its own configs

    Args:
        protocol_slices: e.g.
            [("chrono70_30", ["full"]),
             ("preCOVID", ["covid_only", "post_covid", "all_test"])]
    """
    seeds = list(seeds)
    s_values = list(s_values)
    k_values = list(k_values)
    methods = list(methods)
    universes = list(universes)
    protocol_slices = [(p, list(ss)) for p, ss in protocol_slices]

    # Methods that take an `s` (sparsity) parameter and have multiple seeds
    SPARSE_RANDOMIZED = {"sparse_jl",
                         "crashsketch_R", "crashsketch_noR",
                         "crashsketch_R_int8", "crashsketch_noR_int8",
                         "crashsketch_R_1bit", "crashsketch_noR_1bit",
                         "crashsketch_R_normsign", "crashsketch_noR_normsign"}
    DENSE_RANDOMIZED = {"dense_jl"}
    DETERMINISTIC = {"pca", "raw"}

    configs: list[ExperimentConfig] = []
    for universe in universes:
        for protocol, slices in protocol_slices:
            for slice_, method, k in product(slices, methods, k_values):
                if method in SPARSE_RANDOMIZED:
                    for s, seed in product(s_values, seeds):
                        if s > k:
                            continue
                        configs.append(ExperimentConfig(
                            universe=universe, protocol=protocol, slice=slice_,
                            method=method, k=k, s=s, seed=seed,
                            train_frac=train_frac, n_pairs=n_pairs,
                        ))
                elif method in DENSE_RANDOMIZED:
                    for seed in seeds:
                        configs.append(ExperimentConfig(
                            universe=universe, protocol=protocol, slice=slice_,
                            method=method, k=k, s=None, seed=seed,
                            train_frac=train_frac, n_pairs=n_pairs,
                        ))
                elif method in DETERMINISTIC:
                    configs.append(ExperimentConfig(
                        universe=universe, protocol=protocol, slice=slice_,
                        method=method, k=k, s=None, seed=seeds[0],
                        train_frac=train_frac, n_pairs=n_pairs,
                    ))
                else:
                    raise ValueError(f"Unknown method {method!r}")

    return configs
