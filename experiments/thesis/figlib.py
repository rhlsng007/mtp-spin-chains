"""
Framework shared by every thesis figure script (Phase 8, T8.1 to T8.3).

Each thesis figure is one script experiments/thesis/figNN_<slug>.py that

1. declares a FigureSpec (number, slug, chapter, phases, caption);
2. defines build(ctx) -> Figure, which loads saved results through ctx.load(),
   records the parameters it uses (ctx.param) and its observations (ctx.note),
   and validates the data against exact results (ctx.check_*);
3. ends with  raise SystemExit(run_figure(SPEC, build)).

run_figure() applies the thesis style, runs build(), audits the style
(printed width, font sizes, fixed series colours), saves
figures/thesis/<name>.pdf and .png, and writes a sidecar
results/phase8/figures/<name>.json with the inputs (file, the commit that
produced it, a content hash), the parameters, the checks and the notes. The
sidecars are what make_thesis_figures.py turns into figures/thesis/MANIFEST.md,
and what the Results notes (T8.3) are written from.

Exit codes: 0 all checks passed; 1 a check failed (with CFG.thesis.strict);
an exception inside build() propagates (exit code 1 with a traceback).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import os
import platform
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")      # figures are written, never shown
logging.getLogger("fontTools").setLevel(logging.ERROR)   # cmr10 has a zero 'created' date

import numpy as np

from config import CFG, FIGURES_DIR, RESULTS_DIR
from utils.io import _git_commit
from utils.plotting import audit_style, save_thesis_figure, thesis_style

PROJECT_ROOT = Path(__file__).resolve().parents[2]


# ---------------------------------------------------------------------------
# Specification and context
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class FigureSpec:
    number: int
    slug: str
    chapter: str
    phases: tuple
    title: str
    caption: str
    width: str = "full"                  # "full" (6.3 in) or "half" (3.1 in)

    @property
    def key(self) -> str:
        return f"fig{self.number:02d}"

    @property
    def name(self) -> str:
        return f"{self.key}_{self.slug}"


@dataclass
class Check:
    label: str
    passed: bool
    value: object = None
    target: object = None
    tol: object = None
    detail: str = ""

    def line(self) -> str:
        tag = "PASS" if self.passed else "FAIL"
        parts = [f"[{tag}] {self.label}"]
        if self.value is not None:
            parts.append(f"value {_fmt(self.value)}")
        if self.target is not None:
            parts.append(f"target {_fmt(self.target)}")
        if self.tol is not None:
            parts.append(self.tol if isinstance(self.tol, str) else f"tol {_fmt(self.tol)}")
        if self.detail:
            parts.append(self.detail)
        return "; ".join(parts)


def _fmt(x) -> str:
    """Compact, deterministic number formatting for logs and the manifest."""
    if isinstance(x, (bool, np.bool_)):
        return str(bool(x))
    if isinstance(x, (int, np.integer)):
        return str(int(x))
    if isinstance(x, (float, np.floating)):
        x = float(x)
        if x == 0 or not math.isfinite(x):
            return f"{x:g}"
        return f"{x:.6g}" if 1e-3 <= abs(x) < 1e5 else f"{x:.3e}"
    if isinstance(x, (list, tuple, np.ndarray)):
        return "[" + ", ".join(_fmt(v) for v in np.ravel(np.asarray(x, dtype=object))) + "]"
    return str(x)


def _json_safe(obj):
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return _json_safe(obj.tolist())
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, float) and not math.isfinite(obj):
        return str(obj)
    if isinstance(obj, Path):
        return obj.as_posix()
    return obj


def file_digest(path: Path, n_hex: int = 12) -> str:
    """Short SHA-256 of a file (identifies the exact data a figure was drawn from)."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()[:n_hex]


class Context:
    """Passed to build(ctx): loads inputs and records parameters, notes and checks."""

    def __init__(self, spec: FigureSpec, results_dir: Path = RESULTS_DIR, verbose: bool = True):
        self.spec = spec
        self.results_dir = Path(results_dir)
        self.verbose = verbose
        self.inputs: list[dict] = []
        self.params: dict = {}
        self.notes: list[str] = []
        self.checks: list[Check] = []

    # -- inputs -------------------------------------------------------------
    def load(self, name: str, subdir: str, require: tuple = ()) -> dict:
        """
        Arrays of results/<subdir>/<name>.npz; records the file, the commit that
        produced it (from the .json metadata) and a content hash. ``require``
        lists keys that must be present (a clear error lists the available ones).
        """
        npz = self.results_dir / subdir / f"{name}.npz"
        meta_path = npz.with_suffix(".json")
        if not npz.exists():
            raise FileNotFoundError(
                f"{self.spec.name}: missing input {npz.relative_to(self.results_dir.parent)}; "
                f"run the Phase {subdir[-1] if subdir[-1:].isdigit() else '?'} experiments first")
        with np.load(npz, allow_pickle=False) as data:
            arrays = {k: data[k] for k in data.files}
        meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
        missing = [k for k in require if k not in arrays]
        if missing:
            raise KeyError(f"{subdir}/{name}: missing keys {missing}; available: {sorted(arrays)}")
        self.inputs.append({
            "file": f"{subdir}/{name}",
            "commit": meta.get("git_commit", "unknown"),
            "timestamp": meta.get("timestamp", ""),
            "sha256": file_digest(npz),
        })
        return arrays

    def load_json(self, name: str, subdir: str) -> dict:
        """A plain JSON result (e.g. phaseN_report_data.json), recorded as an input."""
        path = self.results_dir / subdir / name
        if not path.exists():
            raise FileNotFoundError(f"{self.spec.name}: missing input {subdir}/{name}")
        self.inputs.append({"file": f"{subdir}/{name}", "commit": "", "timestamp": "",
                            "sha256": file_digest(path)})
        return json.loads(path.read_text(encoding="utf-8"))

    # -- records -------------------------------------------------------------
    def param(self, **kw) -> None:
        self.params.update(kw)

    def note(self, text: str) -> None:
        self.notes.append(text)
        if self.verbose:
            print(f"[note] {text}")

    def _add(self, check: Check) -> bool:
        self.checks.append(check)
        if self.verbose:
            print(check.line())
        return check.passed

    def check_close(self, label: str, value, target, atol: float = 0.0, rtol: float = 0.0,
                    detail: str = "") -> bool:
        """|value - target| <= atol + rtol |target| (elementwise; max reported)."""
        v, t = np.asarray(value, dtype=float), np.asarray(target, dtype=float)
        err = np.abs(v - t)
        ok = bool(np.all(np.isfinite(v)) and np.all(err <= atol + rtol * np.abs(t)))
        worst = float(np.max(err)) if err.size else 0.0
        tol = " + ".join(f"{n} {_fmt(x)}" for n, x in (("atol", atol), ("rtol", rtol)) if x) or "exact"
        if v.size == 1:
            detail = "; ".join(x for x in (f"deviation {_fmt(worst)}", detail) if x)
            return self._add(Check(label, ok, float(v), float(t) if t.size == 1 else t,
                                   tol, detail))
        detail = "; ".join(x for x in (f"max deviation {_fmt(worst)} over {v.size} values", detail) if x)
        return self._add(Check(label, ok, None, None, tol, detail))

    def check_max(self, label: str, value, bound, detail: str = "") -> bool:
        """max(value) <= bound."""
        v = float(np.max(np.asarray(value, dtype=float)))
        return self._add(Check(label, bool(math.isfinite(v) and v <= bound), v, f"<= {_fmt(bound)}",
                               None, detail))

    def check_min(self, label: str, value, bound, detail: str = "") -> bool:
        """min(value) >= bound."""
        v = float(np.min(np.asarray(value, dtype=float)))
        return self._add(Check(label, bool(math.isfinite(v) and v >= bound), v, f">= {_fmt(bound)}",
                               None, detail))

    def check_true(self, label: str, condition, detail: str = "") -> bool:
        return self._add(Check(label, bool(condition), None, None, None, detail))

    @property
    def n_failed(self) -> int:
        return sum(not c.passed for c in self.checks)


# ---------------------------------------------------------------------------
# Running a figure script
# ---------------------------------------------------------------------------
def sidecar_dir(results_dir: Path = RESULTS_DIR) -> Path:
    return Path(results_dir) / CFG.thesis.results_subdir / "figures"


def figure_dir(figures_dir: Path = FIGURES_DIR) -> Path:
    return Path(figures_dir) / CFG.thesis.figure_subdir


def run_figure(spec: FigureSpec, build, argv=None, *, figures_dir: Path = FIGURES_DIR,
               results_dir: Path = RESULTS_DIR) -> int:
    """Build, audit, save and document one thesis figure; returns the exit code."""
    ap = argparse.ArgumentParser(description=f"{spec.name}: {spec.title}")
    ap.add_argument("--no-save", action="store_true", help="run the checks only, write nothing")
    ap.add_argument("--quiet", action="store_true", help="print only the summary line")
    ap.add_argument("--latex", action="store_true", help="render text with a LaTeX installation")
    args = ap.parse_args(argv)

    t0 = time.perf_counter()
    ctx = Context(spec, results_dir=results_dir, verbose=not args.quiet)
    import matplotlib.pyplot as plt

    with thesis_style(use_latex=args.latex):
        fig = build(ctx)
        for rule, ok, detail in audit_style(fig, spec.width):
            ctx._add(Check(f"style: {rule}", ok, None, None, None, detail))
        paths = []
        if not args.no_save:
            paths = save_thesis_figure(fig, spec.name, figure_dir(figures_dir),
                                       CFG.thesis.formats, CFG.thesis.png_dpi)
    plt.close(fig)

    n_pass = len(ctx.checks) - ctx.n_failed
    record = {
        "name": spec.name,
        "number": spec.number,
        "slug": spec.slug,
        "chapter": spec.chapter,
        "phases": list(spec.phases),
        "title": spec.title,
        "caption": spec.caption,
        "width": spec.width,
        "command": f"python experiments\\thesis\\{spec.name}.py",
        "outputs": [p.relative_to(Path(figures_dir).parent).as_posix() for p in paths],
        "inputs": ctx.inputs,
        "params": ctx.params,
        "notes": ctx.notes,
        "checks": [c.__dict__ | {"line": c.line()} for c in ctx.checks],
        "n_passed": n_pass,
        "n_failed": ctx.n_failed,
        "code_commit": _git_commit(),
        "python": platform.python_version(),
        "seconds": round(time.perf_counter() - t0, 2),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    if not args.no_save:
        out = sidecar_dir(results_dir)
        out.mkdir(parents=True, exist_ok=True)
        (out / f"{spec.name}.json").write_text(json.dumps(_json_safe(record), indent=2), encoding="utf-8")

    print(f"{spec.name}: {n_pass} checks passed, {ctx.n_failed} failed, "
          f"{len(ctx.notes)} notes ({record['seconds']:.1f} s)")
    return 1 if (ctx.n_failed and CFG.thesis.strict) else 0
