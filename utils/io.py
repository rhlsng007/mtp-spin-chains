"""
Saving and loading of simulation results.

Every result is stored as two files with the same base name:

    results/<name>.npz    numerical arrays (compressed)
    results/<name>.json   metadata: parameters, config snapshot, timestamp,
                          git commit, and package versions

This guarantees that every number in the thesis can be traced back to the
exact code version and parameters that produced it.
"""

import json
import platform
import subprocess
from datetime import datetime
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Callable

import numpy as np

from config import RESULTS_DIR, CFG

TRACKED_PACKAGES = ("numpy", "scipy", "pennylane", "torch", "scikit-learn", "matplotlib")


# ---------------------------------------------------------------------------
# Provenance helpers
# ---------------------------------------------------------------------------
def _git_commit() -> str:
    """Current git commit hash, or 'unknown' outside a repository."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, check=True,
            cwd=Path(__file__).resolve().parent.parent,
        )
        return out.stdout.strip()
    except Exception:
        return "unknown"


def _package_versions() -> dict:
    versions = {"python": platform.python_version()}
    for pkg in TRACKED_PACKAGES:
        try:
            versions[pkg] = importlib_metadata.version(pkg)
        except importlib_metadata.PackageNotFoundError:
            versions[pkg] = "not installed"
    return versions


def _to_json_safe(obj):
    """Convert numpy types and tuples so json.dump can handle them."""
    if isinstance(obj, dict):
        return {str(k): _to_json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_json_safe(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, Path):
        return str(obj)
    return obj


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def result_paths(name: str, subdir: str = "") -> tuple:
    folder = RESULTS_DIR / subdir
    folder.mkdir(parents=True, exist_ok=True)
    return folder / f"{name}.npz", folder / f"{name}.json"


def result_exists(name: str, subdir: str = "") -> bool:
    npz_path, json_path = result_paths(name, subdir)
    return npz_path.exists() and json_path.exists()


def save_results(name: str, arrays: dict, params: dict = None, subdir: str = "") -> Path:
    """
    Save a dictionary of arrays plus metadata.

    Parameters
    ----------
    name   : base file name, e.g. "ed_tfim_N8"
    arrays : {"h": h_values, "energy": energies, ...}
    params : experiment-specific parameters, e.g. {"N": 8, "model": "tfim"}
    subdir : optional subfolder inside results/, e.g. "phase1"
    """
    npz_path, json_path = result_paths(name, subdir)

    np.savez_compressed(npz_path, **{k: np.asarray(v) for k, v in arrays.items()})

    meta = {
        "name": name,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "git_commit": _git_commit(),
        "versions": _package_versions(),
        "params": params or {},
        "array_keys": sorted(arrays.keys()),
        "config": CFG.to_dict(),
    }
    with open(json_path, "w") as f:
        json.dump(_to_json_safe(meta), f, indent=2)

    return npz_path


def load_results(name: str, subdir: str = "") -> tuple:
    """Return (arrays_dict, metadata_dict)."""
    npz_path, json_path = result_paths(name, subdir)
    if not npz_path.exists():
        raise FileNotFoundError(f"No saved result named '{name}' in {npz_path.parent}")

    with np.load(npz_path, allow_pickle=False) as data:
        arrays = {k: data[k] for k in data.files}
    with open(json_path) as f:
        meta = json.load(f)
    return arrays, meta


def load_or_compute(name: str, compute_fn: Callable[[], dict], params: dict = None,
                    subdir: str = "", recompute: bool = False) -> dict:
    """
    Load a cached result if it exists, otherwise compute and save it.

    compute_fn must take no arguments and return a dict of arrays.
    Set recompute=True to force a fresh run (e.g. after fixing a bug).
    """
    if result_exists(name, subdir) and not recompute:
        arrays, _ = load_results(name, subdir)
        print(f"[cache] loaded '{name}'")
        return arrays

    print(f"[compute] running '{name}' ...")
    arrays = compute_fn()
    save_results(name, arrays, params=params, subdir=subdir)
    print(f"[saved] '{name}'")
    return arrays
