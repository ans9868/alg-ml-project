"""CrashSketch: sparse sign-only Johnson-Lindenstrauss projection with
optional random orthogonal rotation preprocessing.

Composes three ideas, two of which are configurable:
    1. (Optional) random orthogonal rotation R applied to inputs before
       projection. Idea borrowed from PolarQuant (Google Research, 2025):
       random rotation makes the per-coordinate distribution roughly
       Gaussian regardless of the input's heavy-tail behavior.
    2. Sparse JL projection matrix S in {-1/sqrt(s), 0, +1/sqrt(s)}^{N x k},
       per Cohen-Jayram-Nelson SOSA 2018.
    3. (Reserved for future variants) output quantization to int8 or 1-bit.
       For preliminary experiments we stick with float64 output so the
       metric implementations (Euclidean) work directly.

The preliminary `CrashSketch` class implements (1) + (2). A
`CrashSketchInt8` and `CrashSketch1Bit` could follow once these results
look good.

Interface matches PCAReducer / DenseGaussianJL / SparseJL: fit(X_train),
transform(X), fit_transform(X_train), nnz, save(path), load(path).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from src.utils import load_npz_dict, save_npz_dict


class CrashSketch:
    """Sparse sign-only JL with optional random rotation and output quantization.

    Args:
        k:            target compressed dimension
        s:            number of nonzeros per input coordinate
        seed:         RNG seed for rotation and projection
        use_rotation: if True, pre-rotate inputs by a random orthogonal matrix
                      before projection. Inspired by PolarQuant: rotation makes
                      the per-coordinate distribution roughly Gaussian, which
                      should make uniform quantization MSE-optimal.
        quantize:     output quantization mode:
                        'float' = no quantization (returns float64)
                        'int8'  = z-score per coordinate, 8-bit quantize,
                                  dequantize-on-output so downstream metrics
                                  remain Euclidean-comparable
                      Future modes: '1bit'.

    Output: numpy array of shape (n_samples, k); always returned as float
    (after quantize→dequantize round-trip in non-float modes) so that the
    existing metric framework (Euclidean distance, KMeans, ‖·‖₂ anomaly
    score) works unchanged.
    """

    def __init__(self, k: int, s: int = 3, seed: int = 0,
                 use_rotation: bool = True, quantize: str = "float"):
        if s > k:
            raise ValueError(f"CrashSketch requires s <= k, got s={s}, k={k}")
        if quantize not in ("float", "int8", "1bit", "normsign"):
            raise ValueError(f"unknown quantize mode {quantize!r}; "
                             f"supported: 'float', 'int8', '1bit', 'normsign'")
        self.k = k
        self.s = s
        self.seed = seed
        self.use_rotation = use_rotation
        self.quantize = quantize
        self.R: np.ndarray | None = None
        self.S: np.ndarray | None = None
        # Per-coordinate stats for int8 quantization, set in fit():
        self.q_mu: np.ndarray | None = None
        self.q_sigma: np.ndarray | None = None
        # int8 scaling: maps |z|=2 (in z-score units) to int8 boundary.
        self._q_scale: float = 64.0

    # -----------------------------------------------------------------
    # Fit / transform
    # -----------------------------------------------------------------
    def fit(self, X_train: np.ndarray) -> "CrashSketch":
        N = X_train.shape[1]

        # Use two distinct RNG streams so that varying use_rotation does not
        # change the projection matrix — keeps comparisons clean.
        rng_R = np.random.default_rng(self.seed)
        rng_S = np.random.default_rng(self.seed + 100_003)  # offset prime

        if self.use_rotation:
            # Random orthogonal matrix via QR of standard Gaussian.
            A = rng_R.standard_normal((N, N))
            Q, _ = np.linalg.qr(A)
            # Adjust signs so columns are uniformly distributed on the
            # unit sphere (standard trick from numerical linear algebra).
            d = np.sign(np.diag(Q))
            d[d == 0] = 1.0
            self.R = Q * d
        else:
            self.R = None

        # Sparse sign-only projection matrix
        S = np.zeros((N, self.k), dtype=np.float64)
        scale = 1.0 / np.sqrt(self.s)
        for j in range(N):
            cols = rng_S.choice(self.k, size=self.s, replace=False)
            signs = rng_S.choice([-1.0, 1.0], size=self.s)
            S[j, cols] = signs * scale
        self.S = S

        # Compute training-set per-coordinate stats for quantization
        if self.quantize != "float":
            X_in = X_train @ self.R if self.R is not None else X_train
            Z_train = X_in @ self.S
            self.q_mu = Z_train.mean(axis=0)
            self.q_sigma = Z_train.std(axis=0) + 1e-9

        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.S is None:
            raise RuntimeError("CrashSketch.fit must be called before transform.")
        X_in = X @ self.R if self.R is not None else X
        Z = X_in @ self.S

        if self.quantize == "float":
            return Z

        if self.quantize == "int8":
            # z-score, scale, clip, quantize, then dequantize so downstream
            # metrics see Euclidean-comparable floats. Note: this uses training
            # statistics (q_mu, q_sigma) and so is regime-fragile under stress.
            Z_norm = (Z - self.q_mu) / self.q_sigma
            Z_q = np.clip(np.round(Z_norm * self._q_scale), -128, 127).astype(np.int8)
            Z_dq = (Z_q.astype(np.float64) / self._q_scale) * self.q_sigma + self.q_mu
            return Z_dq

        if self.quantize == "1bit":
            # Distribution-free: sign relative to ZERO, no training stats.
            # By the QJL theorem, sign-only sparse JL preserves angular distance
            # in expectation. We return a {-1/sqrt(k), +1/sqrt(k)}^k vector so
            # the downstream Euclidean-distance metric framework still runs;
            # note that ||Z||_2 = 1 by construction, so anomaly_recall via the
            # ||Z||_2 score is degenerate for this mode (a known limitation).
            Z_sign = np.sign(Z)
            Z_sign[Z_sign == 0] = 1.0   # consistent tie-break
            return Z_sign / np.sqrt(self.k)

        if self.quantize == "normsign":
            # Norm + sign decomposition. Store sign(Z) (1 bit/coord) plus
            # ||Z||_2 (1 float). Distribution-free (no training stats).
            # Reconstruction: Z_recon[i] = sign(Z[i]) * ||Z||_2 / sqrt(k).
            # Note ||Z_recon||_2 = ||Z||_2 exactly, so anomaly recall via the
            # ||Z||_2 score is preserved unchanged. NN/ARI/distance use the
            # equalized-magnitude reconstruction, so suffer some loss vs float
            # but should still beat pure 1-bit substantially.
            norms = np.linalg.norm(Z, axis=1, keepdims=True)
            signs = np.sign(Z)
            signs[signs == 0] = 1.0
            return signs * (norms / np.sqrt(self.k))

        raise ValueError(f"unknown quantize {self.quantize!r}")

    def fit_transform(self, X_train: np.ndarray) -> np.ndarray:
        return self.fit(X_train).transform(X_train)

    # -----------------------------------------------------------------
    # Reporting
    # -----------------------------------------------------------------
    @property
    def nnz(self) -> int:
        """Number of nonzero entries in the *projection* matrix S only.
        The rotation R is dense by construction; we don't count it because
        it's a precompute that operates in the original input space.
        """
        return int(np.count_nonzero(self.S)) if self.S is not None else 0

    @property
    def total_nnz_with_rotation(self) -> int:
        """Total nonzeros across R and S (for accurate storage accounting).
        R is essentially full N*N. Only useful when comparing total size
        against dense JL or PCA which also have N*k storage.
        """
        s_nnz = self.nnz
        r_nnz = int(self.R.size) if self.R is not None else 0
        return s_nnz + r_nnz

    # -----------------------------------------------------------------
    # Persistence
    # -----------------------------------------------------------------
    def save(self, path: Path) -> None:
        if self.S is None:
            raise RuntimeError("Cannot save an unfit CrashSketch.")
        d: dict[str, np.ndarray] = {
            "method": np.array(["crashsketch"]),
            "k": np.array([self.k]),
            "s": np.array([self.s]),
            "seed": np.array([self.seed]),
            "use_rotation": np.array([int(self.use_rotation)]),
            "quantize": np.array([self.quantize]),
            "S": self.S,
        }
        if self.R is not None:
            d["R"] = self.R
        if self.q_mu is not None:
            d["q_mu"] = self.q_mu
            d["q_sigma"] = self.q_sigma
        save_npz_dict(path, **d)

    @classmethod
    def load(cls, path: Path) -> "CrashSketch":
        d = load_npz_dict(path)
        obj = cls(
            k=int(d["k"][0]),
            s=int(d["s"][0]),
            seed=int(d["seed"][0]),
            use_rotation=bool(int(d["use_rotation"][0])),
            quantize=str(d["quantize"][0]) if "quantize" in d else "float",
        )
        obj.S = d["S"]
        if "R" in d:
            obj.R = d["R"]
        if "q_mu" in d:
            obj.q_mu = d["q_mu"]
            obj.q_sigma = d["q_sigma"]
        return obj
