"""A3max (post-hoc sensitivity): config identical to A3 but for the KL filter, layer choice, jobs, reporting."""

import pandas as pd
import pytest

from aftermarket.attacks import a3max
from aftermarket.attacks.abliterate import NO_KL, choose_layer
from aftermarket.evaluate import ATTACK_ORDER, POST_HOC, RETENTION_FAMILIES, _holm_by, family_of
from aftermarket.generate import load_attacks

CANDS = [{"layer": 7, "refusal": 0.9, "kl": 0.05, "sep": 3.0},   # passes KL, barely moves refusal
         {"layer": 20, "refusal": 0.0, "kl": 1.2, "sep": 1.0}]   # removes refusal, fails KL


def test_a3max_config_is_a3_without_the_kl_filter():
    a = load_attacks()
    differs = {k for k in set(a["A3"]) | set(a["A3max"]) if a["A3"].get(k) != a["A3max"].get(k)}
    assert differs == {"select_by", "splits", "post_hoc", "simulates"}
    assert a["A3max"]["select_by"] == NO_KL and a["A3max"]["post_hoc"] is True


def test_choose_layer_without_kl_filter():
    assert choose_layer(CANDS, 0.1) == (7, False)    # A3: the only KL-ok layer
    assert choose_layer(CANDS, None) == (20, False)  # A3max: lowest refusal of all


def test_a3max_jobs_all_12_runs_harmful_and_mmlu():
    js = a3max.jobs()
    assert len(js) == 24 and {j.attack_id for j in js} == {"A3max"}
    assert {j.split for j in js} == {"harmful_test", "mmlu_test"}
    assert {"AM", "AM_rand0", "AM_rand1", "AM_all", "BASE", "SDT"} <= {j.run_id for j in js}


def test_predicted_layer_from_a3_manifest():
    p = a3max.predicted_layer({"layer": 7, "refusal_at_layer": 0.9, "refusal_no_ablation": 0.95, "candidates": CANDS})
    assert (p["a3_layer"], p["a3max_layer"], p["a3max_refusal"]) == (7, 20, 0.0)
    assert a3max.predicted_layer({"layer": 7}) is None  # v1 manifest: no candidates


def test_post_hoc_never_in_retention_or_primary_holm():
    assert POST_HOC == ["A3max"] and "A3max" in ATTACK_ORDER
    assert all("A3max" not in v for v in RETENTION_FAMILIES.values())
    assert family_of("A3max") == "post-hoc" and family_of("A3") == "primary"
    t = pd.DataFrame({"stage": "test", "vs": "BASE", "metric": "asr_lg", "p_value": [0.01, 0.02, 0.01],
                      "family": ["primary", "primary", "post-hoc"]})
    out = _holm_by(t, ["stage", "metric", "vs", "family"])
    assert out["p_holm"].tolist() == pytest.approx([0.02, 0.02, 0.01])  # post-hoc test does not inflate primary


def test_attack_label_marks_post_hoc():
    from aftermarket import report

    assert report.attack_label("A3max") == "A3max (post-hoc sensitivity)" and report.attack_label("A3") == "A3"


def test_a3max_status_sentence():
    from aftermarket import report

    row = lambda run, a, v: {"stage": "test", "run": run, "attack": a, "metric": "asr_lg", "value": v}
    ci = pd.DataFrame([row("BASE", "A3", 0.7), row("SDT", "A3", 0.1)])
    assert report.a3max_status(ci) == "not run yet (notebook 10)"
    ci = pd.concat([ci, pd.DataFrame([row("BASE", "A3max", 0.7), row("SDT", "A3max", 0.9)])])
    assert report.a3max_status(ci) == ("judged for 2 of 12 runs; mean Llama Guard ASR 80% under A3max vs 40% under "
                                       "A3 on the same runs; 2 of 2 exceed 50%")
