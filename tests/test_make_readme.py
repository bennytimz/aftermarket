import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "paper"))
import make_readme as mr  # noqa: E402

TEX = r"""\begin{tabular}{l r rr}
\toprule
Fix & prompt tok & \multicolumn{2}{c}{T4} \\
 & & tok/s & \$/1M \\
\midrule
\run{BASE} & 377 & 783 & 0.142 \\
\run{AM (LoRA)} & 377 & 511 & 0.218 \\
\bottomrule
\end{tabular}"""


def test_cost_table():
    md = mr.cost_table_md(TEX).splitlines()
    assert md[0] == "| Fix | prompt tok | T4 tok/s | T4 $/1M |"
    assert md[2:] == ["| BASE | 377 | 783 | 0.142 |", "| AM (LoRA) | 377 | 511 | 0.218 |"]


def test_render_fills_and_strips_macros():
    out = mr.render("x {{a}} y {{b}}", {"a": "1", "b": r"\run{ST} \att{A4q3}"}, TEX)
    assert out == "x 1 y ST A4q3"


def test_render_missing_key_fails():
    with pytest.raises(SystemExit, match="nope"):
        mr.render("{{nope}}", {}, TEX)


def test_splice_inserts_once_then_replaces():
    r = "# T\n\n## Problem statement and approach\nbody\n"
    once = mr.splice(r, "## Stage 2\nv1")
    twice = mr.splice(once, "## Stage 2\nv2")
    assert once.index(mr.START) < once.index("## Problem statement")
    assert twice.count(mr.START) == 1 and "v2" in twice and "v1" not in twice
