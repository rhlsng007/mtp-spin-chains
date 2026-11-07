"""
Phase 8 tests: thesis style, figure framework, manifest, inventory, provenance.

All tests use synthetic data in temporary folders; none needs saved results.
"""

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402
import scipy.sparse as sp  # noqa: E402

from config import CFG  # noqa: E402
from experiments.make_thesis_figures import _group_inputs, write_manifest  # noqa: E402
from experiments.thesis.fig01_tfim_ed_baseline import crossing, free_fermion_levels  # noqa: E402
from experiments.thesis.figlib import Context, FigureSpec, _fmt, run_figure  # noqa: E402
from scripts.phase8_inventory import describe_json, family_of, npz_headers  # noqa: E402
from utils.io import format_commit, strip_dirty  # noqa: E402
from utils.plotting import (ALLOWED_FONT_SIZES, SERIES_COLORS, THESIS_WIDTHS, audit_style,  # noqa: E402
                            save_thesis_figure, size_color, thesis_figure, thesis_style,
                            zero_strength_positions)

SPEC = FigureSpec(number=99, slug="toy", chapter="Test", phases=(0,), title="toy",
                  caption="toy figure")


# ---------------------------------------------------------------------------
# Style (T8.2)
# ---------------------------------------------------------------------------
def test_config_has_thesis_block():
    assert CFG.thesis.figure_subdir == "thesis" and CFG.thesis.results_subdir == "phase8"
    assert "thesis" in CFG.to_dict()


def test_size_colors_fixed_and_distinct():
    cols = [size_color(n) for n in (4, 6, 8, 10, 12, 14)]
    assert len(set(cols)) == 6
    assert size_color(8) == size_color(8)
    assert size_color(64) == size_color(14)          # larger N share the N = 14 colour


def test_thesis_figure_printed_width_and_deterministic_pdf(tmp_path):
    with thesis_style():
        fig, ax = thesis_figure()
        ax.plot([0, 1], [0, 1], label="ED", color=SERIES_COLORS["ED"])
        ax.set_xlabel("$h/J$")
        p1 = save_thesis_figure(fig, "a", tmp_path, formats=("pdf",))[0]
        p2 = save_thesis_figure(fig, "b", tmp_path, formats=("pdf",))[0]
        plt.close(fig)
    b1, b2 = p1.read_bytes(), p2.read_bytes()
    assert b1 == b2                                  # no creation date: byte-identical
    assert b"CreationDate" not in b1
    box = re.search(rb"/MediaBox \[ 0 0 ([\d.]+) ([\d.]+) \]", b1)
    assert abs(float(box.group(1)) - 72 * THESIS_WIDTHS["full"]) < 1e-6


def test_audit_style_passes_and_detects_violations():
    with thesis_style():
        fig, ax = thesis_figure(width="half")
        ax.plot([0, 1], [1, 0], color=SERIES_COLORS["HVA"], label="HVA p = 4")
        ax.set_ylabel("y")
        good = {rule: ok for rule, ok, _ in audit_style(fig, "half")}
        ax.text(0.5, 0.5, "too big", fontsize=12)
        ax.plot([0, 1], [0, 1], color="red", label="HEA")
        bad = {rule: ok for rule, ok, _ in audit_style(fig, "full")}
        plt.close(fig)
    assert all(good.values())
    assert not bad["figure width"] and not bad["font sizes"] and not bad["series colours"]
    assert 12.0 not in ALLOWED_FONT_SIZES


def test_zero_strength_positions():
    pos, x0 = zero_strength_positions([0.0, 1e-4, 1e-3, 1e-2])
    assert x0 == pytest.approx(1e-5) and pos[0] == pytest.approx(1e-5)
    assert np.allclose(pos[1:], [1e-4, 1e-3, 1e-2])


# ---------------------------------------------------------------------------
# Figure framework (T8.1, T8.3)
# ---------------------------------------------------------------------------
def test_checks_logic(tmp_path):
    ctx = Context(SPEC, results_dir=tmp_path, verbose=False)
    assert ctx.check_close("a", 1.0 + 1e-10, 1.0, atol=1e-9)
    assert not ctx.check_close("b", 1.1, 1.0, rtol=0.05)
    assert ctx.check_close("c", [1.0, 2.0], [1.0, 2.0 + 1e-12], atol=1e-9)
    assert not ctx.check_close("d", [1.0, np.nan], [1.0, 1.0], atol=1.0)
    assert ctx.check_max("e", [0.1, 0.2], 0.2)
    assert not ctx.check_max("f", [np.nan], 1.0)
    assert ctx.check_min("g", [0.3, 0.5], 0.3)
    assert not ctx.check_true("h", False, "detail")
    assert ctx.n_failed == 4
    assert "rtol" in ctx.checks[1].line() and ctx.checks[1].line().startswith("[FAIL]")


def _toy_results(root: Path) -> None:
    folder = root / "phase0"
    folder.mkdir(parents=True)
    np.savez_compressed(folder / "toy.npz", x=np.linspace(0, 1, 5), y=np.linspace(0, 1, 5) ** 2)
    (folder / "toy.json").write_text(json.dumps({"git_commit": "abc1234", "timestamp": "t",
                                                 "array_keys": ["x", "y"]}))


def _toy_build(target):
    def build(ctx):
        d = ctx.load("toy", "phase0", require=("x", "y"))
        ctx.param(points=len(d["x"]))
        ctx.note("toy note")
        ctx.check_close("y(1) = 1", d["y"][-1], target, atol=1e-12)
        fig, ax = thesis_figure()
        ax.plot(d["x"], d["y"], color=SERIES_COLORS["ED"], label="ED")
        return fig
    return build


def test_run_figure_sidecar_and_exit_codes(tmp_path):
    res, figs = tmp_path / "results", tmp_path / "figures"
    _toy_results(res)
    code = run_figure(SPEC, _toy_build(1.0), argv=["--quiet"], figures_dir=figs, results_dir=res)
    assert code == 0
    assert (figs / CFG.thesis.figure_subdir / "fig99_toy.pdf").exists()
    side = json.loads((res / CFG.thesis.results_subdir / "figures" / "fig99_toy.json").read_text())
    assert side["inputs"][0]["file"] == "phase0/toy" and side["inputs"][0]["commit"] == "abc1234"
    assert len(side["inputs"][0]["sha256"]) == 12
    assert side["params"] == {"points": 5} and side["notes"] == ["toy note"]
    assert side["n_failed"] == 0 and side["n_passed"] == 4      # 1 data check + 3 style checks
    code = run_figure(SPEC, _toy_build(2.0), argv=["--quiet"], figures_dir=figs, results_dir=res)
    assert code == (1 if CFG.thesis.strict else 0)


def test_missing_key_message(tmp_path):
    _toy_results(tmp_path)
    ctx = Context(SPEC, results_dir=tmp_path, verbose=False)
    with pytest.raises(KeyError, match="available"):
        ctx.load("toy", "phase0", require=("z",))
    with pytest.raises(FileNotFoundError):
        ctx.load("nothing", "phase0")


def test_manifest_deterministic(tmp_path):
    rec = {"name": "fig99_toy", "chapter": "Test", "phases": [0], "title": "toy",
           "caption": "cap", "command": "python x", "params": {"N": 8},
           "inputs": [{"file": "phase1/a_N4", "commit": "c1"}, {"file": "phase1/a_N6", "commit": "c1"},
                      {"file": "phase2/b", "commit": "c2"}],
           "checks": [{"line": "[PASS] x"}], "notes": ["n"], "n_passed": 1, "n_failed": 0,
           "timestamp": "2026-01-01T00:00:00", "seconds": 1.0}
    write_manifest({"fig99_toy": rec}, ["fig98_other"], tmp_path / "m1.md")
    rec["timestamp"], rec["seconds"] = "2027-01-01T00:00:00", 9.0
    write_manifest({"fig99_toy": rec}, ["fig98_other"], tmp_path / "m2.md")
    t1 = (tmp_path / "m1.md").read_text()
    assert t1 == (tmp_path / "m2.md").read_text()
    assert "2026" not in t1 and "fig98_other | (not run)" in t1
    assert _group_inputs(rec["inputs"])[0].startswith("phase1: 2 files")


def test_fmt_deterministic():
    assert _fmt(3.14159265) == "3.14159" and _fmt(1e-9) == "1.000e-09" and _fmt(True) == "True"


# ---------------------------------------------------------------------------
# Inventory and provenance
# ---------------------------------------------------------------------------
def test_npz_headers_match_arrays(tmp_path):
    arrays = {"a": np.zeros((3, 4)), "b": np.arange(5, dtype=np.int32),
              "c": np.array("tfim"), "d": np.zeros((2, 2), dtype=complex)}
    path = tmp_path / "x.npz"
    np.savez_compressed(path, **arrays)
    head = npz_headers(path)
    for k, v in arrays.items():
        assert head[k][0] == v.shape and np.dtype(head[k][1]) == v.dtype


def test_family_pattern():
    fam, nums = family_of(Path("phase4/noisy_train_hva_N8_h0.50_s1.npz"))
    assert fam == "phase4/noisy_train_hva_N{}_h{}_s{}" and nums == ("8", "0.50", "1")
    assert family_of(Path("phase1/ed_tfim_pbc_N14.npz"))[0] == "phase1/ed_tfim_pbc_N{}"


def test_describe_json_keeps_values():
    d = describe_json({"a": [1.0, 2.0], "b": {"c": "x"}, "e": list(range(30))})
    assert d["a"] == [1.0, 2.0] and d["b"]["c"] == "x" and len(d["e"]) == 7


def test_commit_dirty_flag():
    assert format_commit("abc1234", False) == "abc1234"
    assert format_commit("abc1234", True) == "abc1234-dirty"
    assert strip_dirty("abc1234-dirty") == "abc1234" and strip_dirty("abc1234") == "abc1234"


# ---------------------------------------------------------------------------
# Figure 1 helpers against exact results
# ---------------------------------------------------------------------------
def _dense_tfim(n, h, J=1.0):
    X = sp.csr_matrix([[0.0, 1.0], [1.0, 0.0]])
    Z = sp.csr_matrix([[1.0, 0.0], [0.0, -1.0]])

    def site(o, i):
        m = sp.identity(1, format="csr")
        for k in range(n):
            m = sp.kron(m, o if k == i else sp.identity(2), format="csr")
        return m
    H = sum(-J * site(Z, i) @ site(Z, i + 1) for i in range(n - 1))
    H = H + sum(-h * site(X, i) for i in range(n))
    return np.linalg.eigvalsh(H.toarray())


@pytest.mark.parametrize("h", [0.3, 1.0, 1.7])
def test_free_fermion_levels_equal_dense_ed(h):
    w = _dense_tfim(6, h)
    e0, gap = free_fermion_levels(6, h, 1.0)
    assert abs(e0 - w[0]) < 1e-12 and abs(gap - (w[1] - w[0])) < 1e-12


def test_closed_form_gap_at_critical_point():
    for n in (4, 6, 8, 14):
        assert abs(free_fermion_levels(n, 1.0, 1.0)[1] - 4 * np.sin(np.pi / (2 * (2 * n + 1)))) < 1e-13


def test_crossing_interpolation():
    x = np.linspace(0, 2, 21)
    assert crossing(x, x, 1.234 + 0 * x, (0, 2)) == pytest.approx(1.234, abs=1e-12)
    assert np.isnan(crossing(x, x, 5 + 0 * x, (0, 2)))


# ---------------------------------------------------------------------------
# Figures 2 to 7: analysis helpers against exact results
# ---------------------------------------------------------------------------
from experiments.thesis.figlib import cc_fit, chord_log, exp_decay_fit, field_index  # noqa: E402
from experiments.thesis.fig04_vqe_error_vs_field import sector_gaps  # noqa: E402


@pytest.mark.parametrize("boundary,scale", [("open", 6.0), ("periodic", 3.0)])
def test_cc_fit_recovers_exact_central_charge(boundary, scale):
    n, c, s0 = 14, 0.5, 0.3
    profile = c / scale * chord_log(n, boundary) + s0
    got, err, const = cc_fit(profile, n, boundary, ell_min=2)
    assert got == pytest.approx(c, abs=1e-12) and const == pytest.approx(s0, abs=1e-12)
    assert err == pytest.approx(0.0, abs=1e-10)
    assert np.isnan(cc_fit(profile[:3], 4, boundary, ell_min=2)[0])      # one point: no fit
    assert np.isnan(cc_fit(chord_log(6, boundary), 6, boundary, 2)[1])   # two points: no error


def test_exp_decay_fit_exact():
    n = np.arange(4, 16, 2)
    b, err, A = exp_decay_fit(n, 0.3 * np.exp(-np.log(2) * n))
    assert b == pytest.approx(np.log(2), abs=1e-12) and A == pytest.approx(0.3, rel=1e-10)
    assert err < 1e-10


def test_check_rounded_and_field_index(tmp_path):
    ctx = Context(SPEC, results_dir=tmp_path, verbose=False)
    assert ctx.check_rounded("a", [0.56441, 3.0e-15, 0.5], [0.5644, 3.161e-15, None])
    assert not ctx.check_rounded("b", [0.5650], [0.5644])
    assert not ctx.check_rounded("c", [np.nan], [0.5])
    assert field_index(np.round(np.linspace(0.1, 2.0, 39), 6), 1.0) == 18
    with pytest.raises(ValueError):
        field_index([0.1, 0.2], 0.15)


def test_sector_gaps_follow_labels():
    se = np.zeros((2, 2, 3))
    se[:, 0, 0], se[:, 1, 0] = [-2.0, -3.0], [-1.0, -2.5]
    e_even, e_odd = sector_gaps({"sector_labels": np.array([[-1], [1]]), "sector_energies": se})
    assert np.allclose(e_even, [-1.0, -2.5]) and np.allclose(e_odd, [-2.0, -3.0])
