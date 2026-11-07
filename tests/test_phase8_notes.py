"""
Phase 8 (T8.3): structure of the Results notes in notes/.

Every phase note has the five sections, and every thesis-figure script it cites
exists, so the notes cannot silently point at a renamed or removed figure.
"""

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
NOTES = ROOT / "notes"
PHASE_NOTES = sorted(NOTES.glob("phase[1-8]_*.md"))
SECTIONS = ("**Status.**", "## Parameters", "## Key observations", "## Failures", "## Thesis figures")


def test_one_note_per_phase():
    phases = sorted(int(p.name[5]) for p in PHASE_NOTES)
    assert phases == list(range(1, 9))
    index = (NOTES / "README.md").read_text(encoding="utf-8")
    assert all(p.name in index for p in PHASE_NOTES)


@pytest.mark.parametrize("path", PHASE_NOTES, ids=lambda p: p.stem)
def test_note_sections_and_figure_scripts(path):
    text = path.read_text(encoding="utf-8")
    for section in SECTIONS:
        assert section in text, f"{path.name}: missing {section!r}"
    for script in re.findall(r"(fig\d\d_[a-z0-9_]+)\.py", text):
        assert (ROOT / "experiments" / "thesis" / f"{script}.py").exists(), f"{path.name}: {script}.py"
