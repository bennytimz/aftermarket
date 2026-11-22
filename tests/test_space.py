"""Space: the public bundle allowlist, and the static page's data functions (space/app.js, run with node)."""

import json
import shutil
import subprocess
from pathlib import Path

import pandas as pd
import pytest

from aftermarket import space_push as S

REPO = Path(__file__).resolve().parents[1]


def js(expr: str):
    """Evaluate `expr` with space/app.js loaded as `A`; returns the JSON-decoded result."""
    node = shutil.which("node")
    if not node:
        pytest.skip("node not installed")
    code = f"const A = require({json.dumps(str(REPO / 'space' / 'app.js'))}); console.log(JSON.stringify({expr}));"
    r = subprocess.run([node, "-e", code], capture_output=True, text=True, encoding="utf-8", check=True)
    return json.loads(r.stdout)


def ci_rows():
    return pd.DataFrame([
        {"stage": "test", "run": "BASE", "attack": "A0", "metric": "asr_lg", "value": .1, "lo": .05, "hi": .15, "n": 200},
        {"stage": "test", "run": "SDT", "attack": "A3", "metric": "asr_lg", "value": .7, "lo": .6, "hi": .8, "n": 200},
        {"stage": "dev", "run": "BASE", "attack": "A0", "metric": "asr_lg", "value": .2, "lo": .1, "hi": .3, "n": 100},
        {"stage": "test", "run": "AMc_k6", "attack": "A0", "metric": "asr_lg", "value": .0, "lo": 0, "hi": 0, "n": 200}])


def tables(tmp_path, extra_cols=None):
    d = tmp_path / "tables"
    d.mkdir()
    ci = ci_rows()
    for k, v in (extra_cols or {}).items():
        ci[k] = v
    ci.to_csv(d / "cell_ci.csv", index=False)
    pd.DataFrame([{"stage": "test", "run": "BASE", "metric": "asr_lg", "retention": .5, "lo": .4, "hi": .6, "n": 200,
                   "families": "A1,A2,A3,A4,A5", "attacks": "A1", "partial": False}]).to_csv(d / "retention.csv",
                                                                                           index=False)
    pd.DataFrame([{"x": 1}]).to_csv(d / "pairwise.csv", index=False)  # not allowlisted: must not be copied
    return d


def test_bundle_keeps_test_rows_and_allowlisted_files_only(tmp_path):
    out = tmp_path / "data"
    counts = S.build_bundle(tables(tmp_path), out)
    assert counts == {"cell_ci.csv": 2, "retention.csv": 1}
    assert sorted(p.name for p in out.iterdir()) == ["cell_ci.csv", "retention.csv"]
    ci = pd.read_csv(out / "cell_ci.csv")
    assert set(ci["stage"]) == {"test"} and "AMc_k6" not in set(ci["run"])


def test_bundle_drops_unlisted_columns_and_rejects_text(tmp_path):
    out = tmp_path / "data"
    S.build_bundle(tables(tmp_path, {"response": "SECRET"}), out)  # unlisted column: dropped on copy
    assert "response" not in pd.read_csv(out / "cell_ci.csv").columns
    pd.DataFrame([{"stage": "test", "run": "BASE", "prompt": "SECRET"}]).to_csv(out / "cell_ci.csv", index=False)
    with pytest.raises(S.UnsafeBundle):
        S.check_bundle(out)  # anything that slips in is refused before upload
    (out / "cell_ci.csv").unlink()
    (out / "notes.txt").write_text("x")
    with pytest.raises(S.UnsafeBundle):
        S.check_bundle(out)


def test_bundle_needs_results(tmp_path):
    with pytest.raises(FileNotFoundError):
        S.build_bundle(tmp_path / "nothing", tmp_path / "data")


CSV = ("stage,run,attack,metric,value,lo,hi,n\r\n"
       "test,SDT,A3,asr_lg,0.7,0.6,0.8,200\r\ntest,BASE,A0,asr_lg,0.1,0.05,0.15,200\r\n"
       "test,BASE,A3max,asr_lg,0.7,0.63,0.76,200\r\n")


def parsed(csv: str) -> str:
    return f"A.parseCSV({json.dumps(csv)})"


def test_js_parse_csv_quotes_and_crlf():
    assert js(parsed('a,b,c\n1,"x,y","say ""hi"""\n')) == [{"a": "1", "b": "x,y", "c": 'say "hi"'}]
    assert js(parsed("")) == []
    assert len(js(parsed(CSV))) == 3


def test_js_matrix_filters_order_and_format():
    m = js(f"(() => {{ const m = A.cellMatrix({parsed(CSV)}, 'asr_lg'); const c = m.cell('SDT', 'A3'); "
           "return [m.runs, m.attacks, A.fmtCi(c.value, c.lo, c.hi), m.cell('SDT', 'A0') === undefined]; })()")
    assert m == [["BASE", "SDT"], ["A0", "A3", "A3max"], "70% [60%–80%]", True]
    assert js(f"A.cellMatrix({parsed(CSV)}, 'asr_lg', ['SDT'], ['A3']).runs") == ["SDT"]
    assert js(f"A.cellMatrix({parsed(CSV)}, 'mmlu').runs") == []
    assert js("A.fmtCi(NaN, 0, 0)") == "-"


def test_js_ramp_ends_and_ink():
    assert js("[A.rampColor(0), A.rampColor(1), A.rampColor(2), A.rampColor(0, 'dark'), A.rampColor(1, 'dark')]") == \
        ["#f0efec", "#0d366b", "#0d366b", "#262624", "#cde2fb"]
    assert js("[A.inkOn(0.9), A.inkOn(0.1), A.inkOn(0.9, 'dark'), A.inkOn(0.1, 'dark')]") == \
        ["#ffffff", "#0b0b0b", "#0b0b0b", "#ffffff"]


def test_js_retention_rows_vs_base():
    ret = ("stage,run,metric,retention,lo,hi,n,families,attacks,partial,ranked\n"
           'test,GC,asr_lg,0.736,0.71,0.76,200,"A1,A2",x,False,True\n'
           'test,AM_all,asr_lg,0.56,0.54,0.58,200,"A2,A3",x,True,False\n')
    rpw = ("stage,metric,attack,run,vs,n,baseline,candidate,delta,p_value,p_holm,significant_holm\n"
           "test,retention,x,GC,BASE,200,0.59,0.74,0.145,0.0001,0.0011,True\n")
    rows = js(f"A.retentionRows({parsed(ret)}, {parsed(rpw)})")
    assert [r["run"] for r in rows] == ["GC", "AM_all"]
    assert rows[0]["ranked"] and not rows[1]["ranked"] and rows[1]["partial"]
    assert rows[0]["vsBase"] == "+0.14 (p_holm 0.001)" and rows[0]["families"] == "A1,A2"


def test_js_markdown_cards_escape_and_tables():
    md = ("# Card: GC\n\n![report card](GC.png)\n\n**bold** <script>x</script> [link](https://a.b/c)\n\n"
          "| a | b |\n|---|---|\n| 1 | `2` |\n")
    html = js(f"A.mdToHtml({json.dumps(md)})")
    assert "<h2>Card: GC</h2>" in html and "<img" not in html and "<script>" not in html and "&lt;script&gt;" in html
    assert "<strong>bold</strong>" in html and '<a href="https://a.b/c"' in html
    assert "<th>a</th>" in html and "<td><code>2</code></td>" in html and "---" not in html


def test_space_page_references_existing_files():
    html = (REPO / "space" / "index.html").read_text(encoding="utf-8")
    for f in S.SPACE_FILES:
        assert (REPO / "space" / f).exists()
    assert 'src="app.js"' in html and 'href="style.css"' in html
    assert "sdk: static" in (REPO / "space" / "README.md").read_text(encoding="utf-8")


def test_report_cards_index(tmp_path):
    src, out = tmp_path / "cards", tmp_path / "out"
    src.mkdir()
    for f in ("GC.png", "GC.md", "BASE.png", "BASE.md", "SP.md"):
        (src / f).write_text("x")
    assert S.build_cards(src, out) == 2
    assert json.loads((out / "index.json").read_text()) == ["BASE", "GC"]


def test_js_pct_matches_python_rounding():
    import pandas as pd

    vals = pd.read_csv(REPO / "results" / "tables" / "cell_ci.csv")["value"].dropna().tolist()[:300]
    vals += [0.085, 0.825, 0.745, 0.005, 0.995, 0.0, 1.0]
    assert js(f"{json.dumps(vals)}.map(A.pct)") == [f"{v:.0%}" for v in vals]


def test_js_markdown_stage2_images_lists_notes():
    md = ("> **Paper:** [p](https://a.b/p.pdf)\n\n1. **one** a\n2. two *it*\n\n"
          "![fig](stage2/fig1_heatmap.png)\n![evil](https://x.y/a.png)\n![up](../cards/GC.png)\n")
    html = js(f"A.mdToHtml({json.dumps(md)}, {{images: true}})")
    assert '<p class="note"><strong>Paper:</strong>' in html
    assert "<ol><li><strong>one</strong> a</li><li>two <em>it</em></li></ol>" in html
    assert html.count("<img") == 1 and 'src="stage2/fig1_heatmap.png"' in html
    assert "<img" not in js(f"A.mdToHtml({json.dumps(md)})")  # cards: images still dropped


def test_stage2_markdown_unwraps_and_rewrites_paths():
    block = ("<!-- stage2:start -->\n## Stage 2 results\n\n> **Paper:** [P](docs/aftermarket_stage2.pdf)\n> more\n\n"
             "1. **a** b\n   c\n2. d\n\n![f](paper/figures/fig1_heatmap.png)\n*cap\nmore*\n\n| x |\n|---|\n<!-- stage2:end -->")
    md = S.stage2_markdown(block)
    assert "## Stage 2" not in md and "<!--" not in md
    assert f"> **Paper:** [P]({S.PAPER_URL}) more" in md
    assert "1. **a** b c\n2. d" in md and "![f](stage2/fig1_heatmap.png)\n*cap more*" in md and "| x |\n|---|" in md


def test_stage2_bundle_allowlist(tmp_path):
    (tmp_path / "stage2.md").write_text("![f](stage2/fig1_heatmap.png)", encoding="utf-8")
    S.check_stage2(tmp_path)
    (tmp_path / "stage2.md").write_text("![f](stage2/other.png)", encoding="utf-8")
    with pytest.raises(S.UnsafeBundle, match="outside the allowlist"):
        S.check_stage2(tmp_path)
    (tmp_path / "stage2.md").write_text("ok", encoding="utf-8")
    (tmp_path / "completions.jsonl").write_text("{}", encoding="utf-8")
    with pytest.raises(S.UnsafeBundle, match="unexpected files"):
        S.check_stage2(tmp_path)
