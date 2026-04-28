"""Shared helpers: I/O, git, run-directory layout."""
from __future__ import annotations

import json
import subprocess
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------
# Filesystem helpers
# ---------------------------------------------------------------------
def save_npz_dict(path: Path, **arrays: np.ndarray) -> None:
    """Save a dict of arrays to a compressed .npz file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)


def load_npz_dict(path: Path) -> dict[str, np.ndarray]:
    """Load a .npz file as a dict of arrays."""
    npz = np.load(path, allow_pickle=False)
    return {k: npz[k] for k in npz.files}


def save_json(path: Path, obj: Any) -> None:
    """Save a JSON-serializable object."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=_json_default)


def load_json(path: Path) -> Any:
    with open(path) as f:
        return json.load(f)


def _json_default(obj: Any) -> Any:
    """Fallback for things json.dump can't natively handle."""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.ndarray,)):
        return obj.tolist()
    if isinstance(obj, (pd.Timestamp,)):
        return obj.isoformat()
    if is_dataclass(obj):
        return asdict(obj)
    raise TypeError(f"Cannot JSON-serialize {type(obj)}")


# ---------------------------------------------------------------------
# Git
# ---------------------------------------------------------------------
def get_git_sha(repo_dir: Path | None = None) -> str:
    """Return the current commit SHA, suffixed with '-dirty' if there are uncommitted changes."""
    repo = str(repo_dir) if repo_dir else None
    try:
        sha = (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=repo, stderr=subprocess.DEVNULL
            )
            .decode()
            .strip()
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"
    try:
        dirty = subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=repo, stderr=subprocess.DEVNULL
        ).decode()
        if dirty.strip():
            sha += "-dirty"
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass
    return sha


# ---------------------------------------------------------------------
# Run directories
# ---------------------------------------------------------------------
def run_dir_for_config(base: Path, config: Any) -> Path:
    """Build the canonical per-config run directory path.

    Layout: {base}/{universe}/{slice}/{method}/k{k}_s{s}_seed{seed}/
    `s` is rendered as `sNA` if not applicable to the method.
    """
    cfg_dict = config_to_dict(config)
    s_part = f"s{cfg_dict.get('s', 'NA')}" if cfg_dict.get("s") is not None else "sNA"
    seed = cfg_dict.get("seed", 0)
    sub = (
        Path(base)
        / cfg_dict["universe"]
        / cfg_dict["slice"]
        / cfg_dict["method"]
        / f"k{cfg_dict['k']}_{s_part}_seed{seed}"
    )
    sub.mkdir(parents=True, exist_ok=True)
    return sub


def config_to_dict(config: Any) -> dict:
    """Coerce a dataclass / dict config into a plain dict."""
    if is_dataclass(config):
        return asdict(config)
    if isinstance(config, dict):
        return dict(config)
    raise TypeError(f"Cannot convert {type(config)} to dict")


def now_utc_iso() -> str:
    return datetime.now(tz=timezone.utc).isoformat()
