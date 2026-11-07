"""
Inventory and provenance check of every saved result (Phase 8).

    python scripts\\phase8_inventory.py

Walks results/phase1 to results/phase7 and writes

* results/phase8/inventory.md: one entry per file family (file names that
  differ only in numbers, e.g. vqe_hva_N{}_p{}), with the values those numbers
  take, the commits that produced the files, and the array keys, shapes and
  dtypes of one representative file; plain JSON results (report data, dataset
  index) with their key structure and values; summaries and tables.
  This is the reference the thesis figure scripts are written against.
* results/phase8/inventory.json: the same per file.

Array shapes are read from the .npy headers inside each .npz, so no array is
loaded (only arrays with at most 4 elements, whose values are shown).

Provenance checks (exit code 1 if any fails):
  P1 file count of every phase equals the count recorded at the Phase 7 backup
  P2 every .npz has its .json metadata with a recorded git commit
  P3 the keys listed in the metadata equal the keys stored in the .npz
  P4 every recorded commit is an ancestor of HEAD (the code that produced the
     result is part of the current history)
"""

import json
import re
import subprocess
import sys
import zipfile
from collections import OrderedDict, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from config import CFG, RESULTS_DIR  # noqa: E402
from utils.io import strip_dirty  # noqa: E402

# File counts per phase at the Phase 7 backup (THESIS_CONTEXT.md, Section 4:
# mtp_results_phase1-7_backup_2026-11-06.zip, 865 files)
EXPECTED_FILES = {1: 85, 2: 182, 3: 62, 4: 182, 5: 158, 6: 78, 7: 118}
NUMBER = re.compile(r"\d+(?:\.\d+)?")
SHOW_VALUES_MAX = 4
INLINE_LIST = re.compile(r"\[\s*((?:[-+\w.\"]+,?\s*)*?)\s*\]")   # short lists on one line


# ---------------------------------------------------------------------------
# Reading
# ---------------------------------------------------------------------------
def npz_headers(path: Path) -> "OrderedDict[str, tuple]":
    """{key: (shape, dtype string)} from the .npy headers inside an .npz."""
    out = OrderedDict()
    with zipfile.ZipFile(path) as zf:
        for info in zf.infolist():
            if not info.filename.endswith(".npy"):
                continue
            with zf.open(info) as f:
                version = np.lib.format.read_magic(f)
                if version == (1, 0):
                    shape, _, dtype = np.lib.format.read_array_header_1_0(f)
                else:
                    shape, _, dtype = np.lib.format.read_array_header_2_0(f)
            out[info.filename[:-4]] = (tuple(int(s) for s in shape), dtype.str)
    return out


def small_values(path: Path, headers: dict) -> dict:
    """Values of arrays with at most SHOW_VALUES_MAX elements."""
    keys = [k for k, (shape, dt) in headers.items()
            if int(np.prod(shape)) <= SHOW_VALUES_MAX and not dt.startswith("|O")]
    if not keys:
        return {}
    with np.load(path, allow_pickle=False) as data:
        return {k: data[k].tolist() for k in keys}


def describe_json(obj, depth: int = 0, max_depth: int = 5, max_keys: int = 60):
    """Compact structure of a JSON value, keeping numbers and short strings."""
    if isinstance(obj, dict):
        if depth >= max_depth:
            return f"{{dict with {len(obj)} keys}}"
        items = list(obj.items())
        out = OrderedDict((k, describe_json(v, depth + 1, max_depth, max_keys))
                          for k, v in items[:max_keys])
        if len(items) > max_keys:
            out["..."] = f"{len(items) - max_keys} more keys"
        return out
    if isinstance(obj, list):
        if all(isinstance(v, (int, float, str)) or v is None for v in obj):
            obj = [describe_json(v) if isinstance(v, str) else v for v in obj]
            return obj if len(obj) <= 12 else obj[:6] + [f"... {len(obj)} values"]
        if not obj:
            return []
        return {"list": len(obj), "first": describe_json(obj[0], depth + 1, max_depth, max_keys)}
    if isinstance(obj, str):
        return obj if len(obj) <= 80 else obj[:77] + "..."
    return obj


def family_of(rel: Path) -> tuple:
    stem = rel.stem
    return (rel.parent.as_posix() + "/" + NUMBER.sub("{}", stem)), tuple(NUMBER.findall(stem))


def natural_key(p: Path):
    return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", p.as_posix())]


# ---------------------------------------------------------------------------
# Git
# ---------------------------------------------------------------------------
def git(*args: str) -> tuple:
    try:
        p = subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True)
        return p.returncode, p.stdout.strip()
    except OSError:
        return 1, ""


def is_ancestor(commit: str, cache: dict) -> bool:
    c = strip_dirty(commit)
    if c not in cache:
        cache[c] = git("merge-base", "--is-ancestor", c, "HEAD")[0] == 0
    return cache[c]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> int:
    out_dir = RESULTS_DIR / CFG.thesis.results_subdir
    out_dir.mkdir(parents=True, exist_ok=True)
    head = git("rev-parse", "--short", "HEAD")[1] or "unknown"
    branch = git("rev-parse", "--abbrev-ref", "HEAD")[1]
    dirty = bool(git("status", "--porcelain", "--untracked-files=no")[1])
    tags = git("tag", "--list", "phase*")[1].split()

    files, problems, ancestor_cache = [], [], {}
    counts, sizes = {}, {}
    for ph in sorted(EXPECTED_FILES):
        folder = RESULTS_DIR / f"phase{ph}"
        all_files = sorted((p for p in folder.rglob("*") if p.is_file()), key=natural_key) \
            if folder.exists() else []
        counts[ph] = len(all_files)
        sizes[ph] = sum(p.stat().st_size for p in all_files)
        for p in all_files:
            rel = p.relative_to(RESULTS_DIR)
            entry = {"path": rel.as_posix(), "bytes": p.stat().st_size, "phase": ph}
            if p.suffix == ".npz":
                headers = npz_headers(p)
                entry["kind"] = "npz"
                entry["arrays"] = {k: [list(s), d] for k, (s, d) in headers.items()}
                entry["values"] = small_values(p, headers)
                meta_path = p.with_suffix(".json")
                if not meta_path.exists():
                    problems.append(f"P2 {rel}: no .json metadata")
                    entry["commit"] = "missing"
                else:
                    meta = json.loads(meta_path.read_text(encoding="utf-8"))
                    entry["commit"] = meta.get("git_commit", "missing")
                    entry["timestamp"] = meta.get("timestamp", "")
                    entry["params"] = meta.get("params", {})
                    listed = meta.get("array_keys")
                    if listed is not None and sorted(listed) != sorted(headers):
                        problems.append(f"P3 {rel}: metadata keys differ from stored keys")
                    if entry["commit"] in ("missing", "unknown", ""):
                        problems.append(f"P2 {rel}: no git commit recorded")
                    elif not is_ancestor(entry["commit"], ancestor_cache):
                        problems.append(f"P4 {rel}: commit {entry['commit']} is not an ancestor of HEAD")
            elif p.suffix == ".json" and p.with_suffix(".npz").exists():
                continue                                    # metadata, covered above
            elif p.suffix == ".json":
                entry["kind"] = "json"
                try:
                    entry["structure"] = describe_json(json.loads(p.read_text(encoding="utf-8")))
                except json.JSONDecodeError as exc:
                    entry["structure"] = f"unreadable: {exc}"
            elif p.suffix in (".txt", ".csv", ".md"):
                text = p.read_text(encoding="utf-8", errors="replace").splitlines()
                entry["kind"] = p.suffix[1:]
                entry["lines"] = len(text)
                if p.suffix == ".csv" and text:
                    entry["header"] = text[0]
            else:
                entry["kind"] = p.suffix[1:] or "file"
            files.append(entry)

    for ph, n in counts.items():
        if n != EXPECTED_FILES[ph]:
            problems.append(f"P1 phase{ph}: {n} files, expected {EXPECTED_FILES[ph]}")

    # -- families ----------------------------------------------------------
    fams = OrderedDict()
    for e in files:
        if e["kind"] != "npz":
            continue
        fam, nums = family_of(Path(e["path"]))
        f = fams.setdefault(fam, {"files": [], "numbers": defaultdict(set), "commits": set()})
        f["files"].append(e)
        f["commits"].add(e["commit"])
        for i, v in enumerate(nums):
            f["numbers"][i].add(v)

    # -- markdown -----------------------------------------------------------
    md = ["# Results inventory", "",
          f"HEAD {head}{'-dirty' if dirty else ''} on {branch}; tags: {', '.join(tags) or 'none'}", "",
          "| Phase | Files | Expected | MB |", "| --- | --- | --- | --- |"]
    for ph in counts:
        md.append(f"| {ph} | {counts[ph]} | {EXPECTED_FILES[ph]} | {sizes[ph] / 2**20:.1f} |")
    md += ["", f"Provenance problems: {len(problems)}"] + [f"- {p}" for p in problems[:50]]
    if len(problems) > 50:
        md.append(f"- ... {len(problems) - 50} more (see inventory.json)")
    md += ["", "## Array files (one entry per family)"]
    for fam, f in fams.items():
        rep = f["files"][0]
        nums = "; ".join("{" + ", ".join(sorted(vals, key=lambda x: float(x))) + "}"
                         for _, vals in sorted(f["numbers"].items()))
        md += ["", f"### {fam}  ({len(f['files'])} files)",
               f"numbers: {nums or '-'}; commits: {', '.join(sorted(f['commits']))}",
               f"representative: {rep['path']}"]
        if rep.get("params"):
            md.append("params: " + json.dumps(rep["params"], sort_keys=True)[:400])
        arr = ", ".join(f"{k} {d.lstrip('<|>=')}{list(s)}" for k, (s, d) in rep["arrays"].items())
        md.append("arrays: " + arr)
        if rep["values"]:
            md.append("values: " + json.dumps(rep["values"])[:600])
    md += ["", "## JSON results"]
    for e in files:
        if e["kind"] == "json":
            text = json.dumps(e["structure"], indent=1)
            text = INLINE_LIST.sub(lambda m: "[" + " ".join(m.group(1).split()) + "]", text)
            md += ["", f"### {e['path']}", "```", text[:20000], "```"]
    md += ["", "## Text and tables"]
    for e in files:
        if e["kind"] in ("txt", "csv", "md"):
            extra = f"; header: {e['header']}" if "header" in e else ""
            md.append(f"- {e['path']}: {e['lines']} lines{extra}")
    others = [e for e in files if e["kind"] not in ("npz", "json", "txt", "csv", "md")]
    if others:
        md += ["", "## Other files"] + [f"- {e['path']} ({e['bytes']} bytes)" for e in others]

    (out_dir / "inventory.md").write_text("\n".join(md) + "\n", encoding="utf-8", newline="\n")
    (out_dir / "inventory.json").write_text(json.dumps(
        {"head": head, "dirty": dirty, "branch": branch, "tags": tags, "counts": counts,
         "problems": problems, "files": files}, indent=1))

    total = sum(counts.values())
    print(f"HEAD {head}{'-dirty' if dirty else ''} on {branch}")
    for ph in counts:
        flag = "" if counts[ph] == EXPECTED_FILES[ph] else f"  <- expected {EXPECTED_FILES[ph]}"
        print(f"phase{ph}: {counts[ph]:4d} files, {sizes[ph] / 2**20:7.1f} MB{flag}")
    print(f"total {total} files, {len(fams)} array families, "
          f"{len({e['commit'] for e in files if e['kind'] == 'npz'})} distinct commits")
    for p in problems[:20]:
        print("  -", p)
    print(f"provenance checks: {'all passed' if not problems else f'{len(problems)} problems'}")
    print(f"written: {(out_dir / 'inventory.md').relative_to(ROOT)}, inventory.json")
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
