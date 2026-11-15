"""Report: placeholder filling, markdown tables, headline numbers, end-to-end on synthetic judged files."""

import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from aftermarket import report
from aftermarket.evaluate import normalize

REPO = Path(__file__).resolve().parents[1]
ATTACKS = ["A0", "A1", "A2", "A3", "A4q4", "A4q3", "A4q2", "A5raw", "A5bon"]


def synthetic(judged: Path, runs=("BASE", "SDT", "AM", "AMc_k6"), n=12, seed=0):
    """Small judged files for every run x attack on test splits (labels only; no real text)."""
    rng = np.random.default_rng(seed)
    judged.mkdir(parents=True, exist_ok=True)
    for run in runs:
        for k, a in enumerate(ATTACKS):
            p = 0.05 + 0.08 * k * (0.5 if run == "AM" else 1.0)
            rows = [{"run_id": run, "attack_id": a, "split": "harmful_test", "item_id": f"h{i}", "variant": 0,
                     "guard_label": "unsafe" if rng.random() < p else "safe",
                     "wg_harmful_response": bool(rng.random() < p), "refusal_label": "refusal",
                     "response": "SECRET", "prompt": "p"} for i in range(n)]
            pd.DataFrame(rows).to_parquet(judged / f"{run}__{a}__harmful_test.parquet", index=False)
            pd.DataFrame([{"run_id": run, "attack_id": a, "split": "mmlu_test", "item_id": f"m{i}", "variant": 0,
                           "mmlu_correct": bool(rng.random() < .5), "refusal_label": "full_compliance"}
                          for i in range(n)]).to_parquet(judged / f"{run}__{a}__mmlu_test.parquet", index=False)
            if a == "A0":
                pd.DataFrame([{"run_id": run, "attack_id": a, "split": "xstest_test", "item_id": f"x{i}",
                               "variant": 0, "wg_refusal": bool(rng.random() < .1), "refusal_label": "refusal"}
                              for i in range(n)]).to_parquet(judged / f"{run}__{a}__xstest_test.parquet", index=False)


def test_fill_placeholders_blocks_values_and_idempotence():
    doc = ("x <!--v:a-->old<!--/v--> y <!--v:missing-->old<!--/v-->\n"
           "<!-- BEGIN t -->\nold table\n<!-- END t -->\n<!-- BEGIN gone -->\nold\n<!-- END gone -->\n")
    out = report.fill_placeholders(doc, {"t": "| new |"}, {"a": "42%"})
    assert "<!--v:a-->42%<!--/v-->" in out and "<!--v:missing-->TBD<!--/v-->" in out
    assert "<!-- BEGIN t -->\n| new |\n<!-- END t -->" in out
    assert "<!-- BEGIN gone -->\n_Pending" in out
    assert report.fill_placeholders(out, {"t": "| new |"}, {"a": "42%"}) == out


def test_ci_matrix_orders_and_formats():
    ci = pd.DataFrame([
        {"stage": "test", "run": "SDT", "attack": "A3", "metric": "asr_lg", "value": .5, "lo": .4, "hi": .6, "n": 9},
        {"stage": "test", "run": "BASE", "attack": "A3", "metric": "asr_lg", "value": .25, "lo": .1, "hi": .4, "n": 9},
        {"stage": "test", "run": "BASE", "attack": "A0", "metric": "asr_lg", "value": .05, "lo": 0., "hi": .1, "n": 9}])
    m = report.ci_matrix(ci, "asr_lg", "test")
    assert list(m.index) == ["BASE", "SDT"] and list(m.columns) == ["A0", "A3"]
    assert m.loc["BASE", "A3"] == "25% [10%–40%]" and m.loc["SDT", "A0"] == "-"
    md = report.md_table(m)
    assert md.splitlines()[0] == "| model | A0 | A3 |" and "| SDT | - | 50% [40%–60%] |" in md
    assert report.md_table(report.ci_matrix(ci, "mmlu", "test")).startswith("_Pending")


def test_headline_values_tbd_when_nothing_judged():
    t = report.collect(normalize(pd.DataFrame(columns=["run_id", "attack_id", "split", "item_id"])))
    v = report.headline_values(t, 0)
    assert v["base_asr_a0"] == v["n_models"] == v["broken_cells"] == "TBD"
    assert "am_retention" not in v  # fill_placeholders turns absent keys into TBD


def test_doc_placeholders_are_all_produced(tmp_path):
    """Every <!--v:key--> in README.md / ABSTRACT.md is a key the report fills (catches typos)."""
    synthetic(tmp_path / "judged")
    df = report.leaderboard_runs(report.load_judged(tmp_path / "judged"))
    values = report.headline_values(report.collect(df), len(df))
    keys = set()
    for doc in report.DOCS:
        keys |= set(re.findall(r"<!--v:(\w+)-->", doc.read_text(encoding="utf-8")))
    assert keys and keys <= set(values), keys - set(values)
    blocks = set()
    for doc in report.DOCS:
        blocks |= set(re.findall(r"<!-- BEGIN (\w+) -->", doc.read_text(encoding="utf-8")))
    produced = set(report.write_tables(report.collect(df), tmp_path / "tables"))
    assert blocks <= produced, blocks - produced


def test_run_end_to_end(tmp_path):
    pytest.importorskip("matplotlib")
    synthetic(tmp_path / "judged")
    docs = []
    for d in report.DOCS:
        (tmp_path / d.name).write_text(d.read_text(encoding="utf-8"), encoding="utf-8")
        docs.append(tmp_path / d.name)
    res = report.run(tmp_path / "judged", fig_dir=tmp_path / "fig", table_dir=tmp_path / "tab", docs=docs)
    names = {p.name for p in res["figures"]}
    for stem in ["heatmap_asr_test", "retention_test", "capability_vs_asr_test", "judge_agreement"]:
        assert {f"{stem}.png", f"{stem}.svg"} <= names
    assert all((tmp_path / "fig" / n).stat().st_size > 1000 for n in names)
    readme = (tmp_path / "README.md").read_text(encoding="utf-8")
    blocks = "".join(m.group(3) for m in report.BLOCK_RE.finditer(readme))
    assert blocks and "AMc_" not in blocks  # dev candidates never reach a generated table (prose may name them)
    assert "<!--v:n_models-->3<!--/v-->" in readme and "_Pending" not in readme.split("## 8. Results")[1]
    assert "SECRET" not in readme and not any("SECRET" in p.read_text(encoding="utf-8")
                                              for p in (tmp_path / "tab").glob("*.*"))
    for name in ["cells", "cell_ci", "pairwise", "retention", "retention_pairwise", "agreement"]:
        assert (tmp_path / "tab" / f"{name}.csv").exists()
    ret = pd.read_csv(tmp_path / "tab" / "retention.csv")
    assert set(ret["run"]) == {"BASE", "SDT", "AM"} and not ret["partial"].any()
    cards = {p.name for p in res["cards"]}
    assert {"BASE.md", "BASE.png", "AM.md", "README.md"} <= cards and "AMc_k6.md" not in cards
    assert all(p.parent == tmp_path / "cards" for p in res["cards"])


def test_run_with_nothing_judged_yet(tmp_path):
    """A fresh checkout (no judged files) gives all-TBD docs and pending tables, not a crash."""
    pytest.importorskip("matplotlib")
    doc = tmp_path / "README.md"
    doc.write_text("<!--v:n_models-->x<!--/v-->\n<!-- BEGIN retention -->\nx\n<!-- END retention -->\n",
                   encoding="utf-8")
    res = report.run(tmp_path / "none", fig_dir=tmp_path / "fig", table_dir=tmp_path / "tab", docs=[doc])
    assert res["figures"] == []
    text = doc.read_text(encoding="utf-8")
    assert "<!--v:n_models-->TBD<!--/v-->" in text and "_Pending" in text


def test_a3_finding():
    import pandas as pd

    rows = lambda run, a0, a3: [{"stage": "test", "run": run, "attack": "A0", "metric": "asr_lg", "value": a0},
                                {"stage": "test", "run": run, "attack": "A3", "metric": "asr_lg", "value": a3}]
    ci = pd.DataFrame(rows("BASE", 0.08, 0.70) + rows("GC", 0.02, 0.30) + rows("AM", 0.0, 0.1))
    assert report.a3_finding(ci.iloc[:0]) == report.TBD
    s = report.a3_finding(ci, runs=("BASE", "GC"))
    assert s == "A3 raises Llama Guard ASR from 5% to 50% on average; 1 of 2 runs exceed 50%"  # AM not counted
    assert report.a3_finding(ci).endswith("(6 runs not judged yet)")


def test_outdated_flags_local_copies_that_differ_from_the_hub(tmp_path):
    import hashlib

    (tmp_path / "a.parquet").write_bytes(b"old")
    (tmp_path / "b.parquet").write_bytes(b"same")
    (tmp_path / "c.parquet").write_bytes(b"local only")
    hub = {"a.parquet": hashlib.sha256(b"new").hexdigest(), "b.parquet": hashlib.sha256(b"same").hexdigest()}
    assert report.outdated(tmp_path, hub) == ["a.parquet"]


def test_headline_rule_puts_wildguard_first_only_when_judges_disagree():
    import pandas as pd

    from aftermarket.report import headline_asr

    def cell(run, attack, metric, v):
        return {"stage": "test", "run": run, "attack": attack, "metric": metric, "value": v, "lo": v - .05,
                "hi": v + .05, "n": 200}
    ci = pd.DataFrame([cell("BASE", "A2", "asr_lg", .81), cell("BASE", "A2", "asr_wg", .90),
                       cell("BASE", "A5bon", "asr_lg", .95), cell("BASE", "A5bon", "asr_wg", .66),
                       cell("GC", "A0", "asr_lg", .01)])
    assert headline_asr(ci, "BASE", "A2") == "81% [76%–86%]"  # within 10 points: Llama Guard alone
    assert headline_asr(ci, "BASE", "A5bon") == "66% [61%–71%] (WildGuard; Llama Guard 95%)"
    assert headline_asr(ci, "GC", "A0") == "1% [-4%–6%]"  # no WildGuard cell: primary judge
    assert headline_asr(ci, "SP", "A0") == "TBD"


def test_retention_table_has_post_hoc_wildguard_column():
    import pandas as pd

    from aftermarket.report import retention_md

    def ret(run, v):
        return {"stage": "test", "run": run, "metric": "m", "retention": v, "lo": v - .02, "hi": v + .02, "n": 200,
                "families": "A2", "attacks": "A2", "partial": False, "ranked": True}
    lg = pd.DataFrame([ret("GC", .74), ret("BASE", .59)])
    wg = pd.DataFrame([ret("GC", .78), ret("BASE", .64)])
    md = retention_md(lg, pd.DataFrame(), "test", wg)
    assert "WildGuard judge (post-hoc sensitivity, not ranked)" in md and "| GC | 1 | 0.74 [0.72–0.76] | 0.78" in md
    assert "WildGuard" not in retention_md(lg, pd.DataFrame(), "test")
