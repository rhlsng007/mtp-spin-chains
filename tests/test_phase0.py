"""Unit tests for Phase 0 utilities. Run with:  pytest -q"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pytest

from config import CFG, make_grid


def test_grid_has_no_float_drift():
    g = make_grid(0.1, 2.0, 0.05)
    assert len(g) == 39
    assert 0.3 in g  # would fail with np.arange because of 0.30000000000000004


def test_noise_grid_starts_at_zero_and_is_increasing():
    s = CFG.noise.strengths()
    assert s[0] == 0.0
    assert np.all(np.diff(s) > 0)
    assert np.isclose(s[-1], CFG.noise.p_max)


@pytest.mark.parametrize("n, noisy, expected", [(8, False, 8), (8, True, 4), (4, True, 2)])
def test_depth_ranges(n, noisy, expected):
    assert CFG.system.depths(n, noisy=noisy)[-1] == expected


def test_io_roundtrip(tmp_path, monkeypatch):
    import utils.io as io
    monkeypatch.setattr(io, "RESULTS_DIR", tmp_path)
    io.save_results("x", {"a": np.arange(3)}, params={"N": 6})
    arrays, meta = io.load_results("x")
    assert np.array_equal(arrays["a"], np.arange(3))
    assert meta["params"]["N"] == 6


def test_load_or_compute_uses_cache(tmp_path, monkeypatch):
    import utils.io as io
    monkeypatch.setattr(io, "RESULTS_DIR", tmp_path)
    calls = []

    def compute():
        calls.append(1)
        return {"y": np.ones(2)}

    io.load_or_compute("cached", compute)
    io.load_or_compute("cached", compute)
    assert len(calls) == 1


def test_seeding_reproducible():
    from utils.seeding import set_seed, child_seeds
    assert np.allclose(set_seed(7).random(4), set_seed(7).random(4))
    assert len(set(child_seeds(7, 10))) == 10
