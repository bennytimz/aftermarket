"""The write-up stays in sync with ABSTRACT.md, SUBMISSION.md and the template rules (no TBDs, under 500 words)."""

import importlib.util
import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def _abstract_body() -> str:
    # ABSTRACT.md: "# Abstract", blank, bold title line, blank, then the body the README repeats
    return (REPO / "ABSTRACT.md").read_text(encoding="utf-8").split("\n", 1)[1].strip().split("\n", 2)[2]


def test_readme_abstract_matches_abstract_md():
    assert _abstract_body() in (REPO / "README.md").read_text(encoding="utf-8")


def test_abstract_under_500_words_and_submission_in_sync():
    spec = importlib.util.spec_from_file_location("make_submission", REPO / "docs" / "make_submission.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    _, n = mod.abstract()
    assert n < 500
    assert f"## Abstract ({n} words)" in (REPO / "SUBMISSION.md").read_text(encoding="utf-8")


def test_no_tbd_in_writeup():
    for name in ("README.md", "ABSTRACT.md", "SUBMISSION.md"):
        assert not re.search(r"\bTBD\b", (REPO / name).read_text(encoding="utf-8")), name
