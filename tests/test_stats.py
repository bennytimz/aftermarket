"""Bootstrap CIs, paired tests, Holm, retention (pandas/numpy only)."""

import math

import numpy as np
import pandas as pd
import pytest

from aftermarket.evaluate import (_local_paired_bootstrap, bootstrap_ci, cell_ci_table, families_of, holm,
                                  item_labels, item_retention, leaderboard_runs, normalize, pairwise_table,
                                  retention_pairwise, retention_table)


def harmful(run, attack, unsafe_by_item, split="harmful_test", variants=1):
    """unsafe_by_item: {item_id: bool or list of bools per variant}."""
    rows = []
    for item, lab in unsafe_by_item.items():
        labs = lab if isinstance(lab, list) else [lab] * variants
        for v, u in enumerate(labs):
            rows.append({"run_id": run, "attack_id": attack, "split": split, "item_id": item, "variant": v,
                         "guard_label": None if u is None else ("unsafe" if u else "safe"),
                         "refusal_label": "refusal"})
    return rows


ITEMS = [f"h{i}" for i in range(10)]


def frame(spec, split="harmful_test"):
    """spec: {(run, attack): n_unsafe} -> first n items unsafe."""
    rows = []
    for (run, attack), k in spec.items():
        rows += harmful(run, attack, {it: i < k for i, it in enumerate(ITEMS)}, split)
    return normalize(pd.DataFrame(rows))


def test_bootstrap_ci_deterministic_and_brackets_mean():
    v = np.array([1.0] * 30 + [0.0] * 70)
    lo, hi = bootstrap_ci(v)
    assert lo < 0.3 < hi and (lo, hi) == bootstrap_ci(v)
    assert bootstrap_ci(np.ones(50)) == (1.0, 1.0)
    assert all(math.isnan(x) for x in bootstrap_ci([1.0]))


def test_cell_ci_table_bon_folds_variants():
    rows = harmful("BASE", "A5bon", {"h0": [False, True], "h1": [False, False], "h2": [None, None]})
    t = cell_ci_table(normalize(pd.DataFrame(rows)), metrics=("asr_lg",))
    r = t.iloc[0]
    assert r.value == 0.5 and r.n == 2  # h2 never parsed: left out


def test_local_paired_bootstrap_behaviour():
    same = [(1.0, 1.0), (0.0, 0.0)] * 20
    assert _local_paired_bootstrap(same)["p_value"] == 1.0
    worse = [(0.0, 1.0)] * 30 + [(0.0, 0.0)] * 10
    r = _local_paired_bootstrap(worse)
    assert r["delta"] == 0.75 and r["p_value"] < 0.01 and r["n"] == 40
    assert "error" in _local_paired_bootstrap([(0.0, 1.0)])


def test_holm():
    assert holm([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.06, 0.06])
    out = holm([0.5, float("nan")])
    assert out[0] == 0.5 and math.isnan(out[1])


def test_pairwise_pairs_by_item_and_skips_missing_reference():
    df = frame({("BASE", "A3"): 2, ("SDT", "A3"): 9, ("SDT", "A2"): 5})
    t = pairwise_table(df, metrics=("asr_lg",))
    assert len(t) == 1  # SDT/A2 has no BASE/A2 cell; no AM run at all
    r = t.iloc[0]
    assert (r.run, r.vs, r.attack, r.n) == ("SDT", "BASE", "A3", 10)
    assert r.delta == pytest.approx(0.7) and r.p_value < 0.05 and r.p_holm == r.p_value
    assert r.engine.endswith("paired_bootstrap")


def test_families_of():
    assert families_of(["A0", "A3", "A4q3", "A4q2", "A4q4", "A5bon"]) == {"A3": ["A3"], "A4": ["A4q3"],
                                                                         "A5": ["A5bon"]}
    assert families_of(["A0", "A4q4", "A4q2"]) == {}  # extra bit widths never stand in for the A4 headline


def test_retention_weights_each_family_equally():
    # A0 ASR 0; A3 ASR 1.0; A4 = A4q3 ASR 0.2 (q4 / q2 are extra data, ignored). Retention = 1 - (1.0 + 0.2) / 2 = 0.4
    df = frame({("BASE", "A0"): 0, ("BASE", "A3"): 10, ("BASE", "A4q4"): 0, ("BASE", "A4q3"): 2,
                ("BASE", "A4q2"): 9})
    lab = item_labels(df, "asr_lg")
    mat = lab.pivot_table(index="item_id", columns="attack_id", values="y", aggfunc="first")
    r = item_retention(mat, mat.columns)
    assert r.mean() == pytest.approx(0.4) and len(r) == 10
    t = retention_table(df)
    assert t.iloc[0].retention == pytest.approx(0.4) and t.iloc[0].partial  # A1, A2, A5 missing
    assert t.iloc[0].families == "A3,A4"


def test_retention_needs_a0():
    df = frame({("BASE", "A3"): 5})
    assert len(retention_table(df)) == 0


def test_retention_pairwise_uses_shared_attacks():
    # BASE: A0 0, A2 10, A3 10. CTRL (reduced battery): A0 0, A3 2 -> compared on A3 only: 0.0 vs 0.8
    df = frame({("BASE", "A0"): 0, ("BASE", "A2"): 10, ("BASE", "A3"): 10,
                ("CTRL", "A0"): 0, ("CTRL", "A3"): 2})
    t = retention_pairwise(df, refs=("BASE",))
    r = t.iloc[0]
    assert (r.run, r.vs, r.attack) == ("CTRL", "BASE", "A3")
    assert r.baseline == pytest.approx(0.0) and r.candidate == pytest.approx(0.8)
    full = retention_table(df).set_index("run")
    assert full.loc["BASE", "retention"] == pytest.approx(0.0) and full.loc["CTRL", "partial"]


def test_leaderboard_runs_drops_dev_candidates():
    df = frame({("BASE", "A0"): 1, ("AMc_k6", "A0"): 1})
    assert set(leaderboard_runs(df)["run_id"]) == {"BASE"}
