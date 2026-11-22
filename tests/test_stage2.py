"""Fast unit tests for Stage 2 helpers (no GPU, no network)."""

from __future__ import annotations

import json

import pytest
import yaml

from aftermarket.stage2 import capability, cell, jobs, subsets
from aftermarket.stage2.subsets import REPO_ROOT, sample_indices, stratified


def test_sample_indices_deterministic_and_distinct():
    a, b = sample_indices(1000, 50, 42), sample_indices(1000, 50, 42)
    assert a == b and len(set(a)) == 50 and a == sorted(a) and max(a) < 1000
    with pytest.raises(ValueError):
        sample_indices(3, 5)


def test_stratified_is_proportional():
    items = [("a", i) for i in range(60)] + [("b", i) for i in range(40)]
    pick = stratified(items, lambda x: x[0], 10, 1)
    assert len(pick) == 10 and sum(x[0] == "a" for x in pick) == 6


def test_frozen_subsets_shape():
    s = subsets.load_subsets()
    cap = s["capability"]
    for t, (n_total, n) in subsets.CAPABILITY.items():
        assert cap[t]["n_total"] == n_total and len(cap[t]["indices"]) == n
        assert len(set(cap[t]["indices"])) == n and max(cap[t]["indices"]) < n_total
    ml = s["multilingual"]
    assert len(ml["harmful_ids"]) == 100 and len(set(ml["harmful_ids"])) == 100
    assert len(ml["xstest_ids"]) == 50
    assert sum(i.startswith("harmbench:") for i in ml["harmful_ids"]) == 50
    test_ids = {json.loads(l)["id"] for l in (REPO_ROOT / "data/splits/harmful_test.jsonl").read_text(encoding="utf-8").splitlines()}
    assert set(ml["harmful_ids"]) <= test_ids
    assert len(s["strongreject"]["rows"]) == 150 and len(s["multijail"]["rows"]) == 100
    tp = s["throughput"]
    assert len(tp["timed_ids"]) == 256 and len(tp["warmup_ids"]) == 32
    assert not set(tp["timed_ids"]) & set(tp["warmup_ids"])


def test_throughput_prompts_exclude_training_and_splits():
    s = subsets.load_subsets()["throughput"]
    excl = subsets.excluded_alpaca_ids(REPO_ROOT / "data" / "ft", REPO_ROOT / "data" / "splits")
    assert not (set(s["timed_ids"]) | set(s["warmup_ids"])) & excl


def test_ml_items_order_and_kinds():
    from aftermarket.stage2.prompts import ml_items

    items = ml_items()
    assert len(items) == 150 and [i["kind"] for i in items[:100]] == ["harmful"] * 100
    assert all(i["kind"] == "xstest" for i in items[100:])


def test_mmlu300_task(tmp_path):
    d = capability.write_mmlu300_task(tmp_path)
    rows = [json.loads(l) for l in (d / "mmlu300.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 300 and all(r["answer"] in range(4) for r in rows)
    cfg = yaml.safe_load((d / "mmlu300.yaml").read_text(encoding="utf-8"))
    assert cfg["task"] == "mmlu300" and "{{question.strip()}}" in cfg["doc_to_text"]


def test_lm_eval_command_samples():
    cmd = capability.lm_eval_command("/m", REPO_ROOT / "o", REPO_ROOT / "t")
    samples = json.loads(cmd[cmd.index("--samples") + 1])
    assert set(samples) == {"arc_challenge", "hellaswag", "winogrande", "gsm8k"}  # mmlu300 + ifeval: full
    assert "--apply_chat_template" in cmd and "--fewshot_as_multiturn" in cmd


def test_cells_config_sets():
    cfg = cell.load_cfg(REPO_ROOT / "configs" / "stage2" / "cells.yaml")
    assert "harmful_test" in cell.sets_for(cfg, "A2") and cell.sets_for(cfg, "A2s1") == ["harmful_test"]
    assert "sr" not in cell.sets_for(cfg, "A4q3") and cell.sets_for(cfg, "A1") == ["harmful_test"]
    assert cell.does_capability(cfg, "A3") and not cell.does_capability(cfg, "A2s1")
    with pytest.raises(SystemExit):
        cell.check_final(False)


def test_pushable_files_refuse_weights(tmp_path):
    out, w = tmp_path / "out", tmp_path / "w"
    (out / "gen").mkdir(parents=True)
    (out / "gen" / "ml_en.parquet").write_bytes(b"x")
    (out / "manifest.json").write_text("{}")
    (out / "model.safetensors").write_bytes(b"x")
    assert {p.name for p in cell.pushable_files(out, w)} == {"ml_en.parquet", "manifest.json"}
    (out / "adapter_config.json").write_text("{}")
    with pytest.raises(PermissionError):
        cell.pushable_files(out, w)


def test_stage2_attack_seed_variant():
    from aftermarket.stage2.weights import stage2_attack

    base, a = stage2_attack("A2s1")
    assert base == "A2" and a["train"]["seed"] == 1
    assert stage2_attack("A2")[1]["train"]["seed"] == 42  # deepcopy: the shared block is untouched


def test_job_script_pins_commit():
    s = jobs.script("abc123", "aftermarket.stage2.cell --run BASE --attack A0 --final")
    assert "git checkout -q abc123" in s and "-c /tmp/keep.txt" in s
    specs = jobs.job_specs("cells", ["BASE", "AM_rand0"], ["A0", "A2"])
    assert len(specs) == 4 and specs[1]["name"] == "s2-am-rand0-a0"


def test_overlap_helpers():
    from aftermarket.stage2.overlap import compare, jaccard, norm, shingles

    assert norm("Make  a BOMB!") == "make a bomb"
    assert jaccard(shingles("abcdefg"), shingles("abcdefg")) == 1.0
    tr = [{"id": "t", "prompt": "How do I make a bomb?"}]
    ev = [{"id": "e1", "prompt": "how do i make a BOMB"}, {"id": "e2", "prompt": "Bake a cake"}]
    pairs = compare(tr, ev)
    assert [p["eval_id"] for p in pairs] == ["e1"] and pairs[0]["exact"]
