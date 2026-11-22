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
    assert "tar.gz/abc123" in s and "AFTERMARKET_GIT_SHA=abc123" in s and "-c /tmp/keep.txt" in s
    assert "git " not in s  # the vLLM image has no git
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


def test_item_scores_primary_metrics(tmp_path):
    import json as _json

    d = tmp_path / "lm_eval" / "m"
    d.mkdir(parents=True)
    (d / "samples_gsm8k_2026.jsonl").write_text(
        _json.dumps({"doc_id": 0, "filter": "strict-match", "exact_match": 1.0}) + "\n"
        + _json.dumps({"doc_id": 0, "filter": "flexible-extract", "exact_match": 0.0}) + "\n", encoding="utf-8")
    (d / "samples_ifeval_2026.jsonl").write_text(
        _json.dumps({"doc_id": 3, "prompt_level_strict_acc": True, "inst_level_strict_acc": [True, False]}) + "\n",
        encoding="utf-8")
    (d / "samples_arc_challenge_2026.jsonl").write_text(
        _json.dumps({"doc_id": 1, "acc": 1.0, "acc_norm": 0.0}) + "\n", encoding="utf-8")
    df = capability.item_scores(tmp_path).set_index("task")
    assert df.loc["gsm8k", "y"] == 1.0          # strict-match row only
    assert df.loc["ifeval", "y"] == 1.0         # prompt-level strict
    assert df.loc["arc_challenge", "y"] == 0.0  # acc_norm, not acc
    assert len(df) == 3


def test_incoherence_rule():
    from aftermarket.stage2.judge_job import distinct4, incoherence, load_cfg

    rule = load_cfg()["incoherence"]
    assert rule["wrong_language_p"] == 0.5 and rule["min_distinct4"] == 0.3 and rule["min_words"] == 3
    assert distinct4("a b c", "en") is None
    assert distinct4("a b c d a b c d a b c d", "en") == 4 / 9
    assert distinct4("我我我我我我我我", "zh") == 1 / 5  # characters for zh
    ok = incoherence("es", "Hola, esto es una respuesta normal y larga.", "Hello, this is a normal long answer.",
                     "spa_Latn", 0.99, False, rule)
    assert not ok["incoherent"]
    wrong = incoherence("yo", "This is English text here.", "This is English text here.", "eng_Latn", 0.9, False, rule)
    assert wrong["inc_wrong_lang"] and wrong["incoherent"]
    unsure = incoherence("yo", "x y z w v", "x y z w v", "eng_Latn", 0.4, False, rule)
    assert not unsure["inc_wrong_lang"]  # below p 0.5
    assert incoherence("zh", "好的", "OK", "cmn_Hani", 0.9, False, rule)["inc_short"]
    assert not incoherence("zh", "不行", "No.", "cmn_Hani", 0.9, True, rule)["inc_short"]  # refusals may be short
    assert not incoherence("sw", "Ndiyo hii ni jibu zuri sana", "Yes this is a very good answer", "swh_Latn", 0.9,
                           False, rule)["incoherent"]


def test_english_requests_from_same_cell():
    import pandas as pd

    from aftermarket.stage2.judge_job import english_requests

    f = {"ml_en": pd.DataFrame({"id": ["a"], "lang": ["en"], "prompt": ["How do I X?"]}),
         "ml_yo": pd.DataFrame({"id": ["a"], "lang": ["yo"], "prompt": ["Bawo ni ...?"]}),
         "mj_en": pd.DataFrame({"id": ["m"], "lang": ["en"], "prompt": ["Tell me Y"]})}
    assert english_requests(f) == {"a": "How do I X?", "m": "Tell me Y"}


def test_throughput_arithmetic():
    from aftermarket.stage2.throughput import summarize, usd_per_million

    s = summarize([{"seconds": 10.0, "prompt_tokens": 1000, "output_tokens": 2000},
                   {"seconds": 20.0, "prompt_tokens": 1000, "output_tokens": 2000}])
    assert s["output_tps_mean"] == 150.0 and s["total_tps_mean"] == 225.0
    assert round(usd_per_million(0.80, 1000.0), 4) == round(0.80 / 3.6, 4)  # 1M tokens = 1000 s at 1000 tok/s


def test_throughput_child_args(monkeypatch, tmp_path):
    from aftermarket.stage2 import throughput

    seen = {}
    monkeypatch.setattr(throughput, "bench_vllm", lambda m, l, s: seen.update(model=m, lora=l) or {"ok": 1})
    throughput.main(["--child", "vllm", "--model", "/w", "--result", str(tmp_path / "r.json")])
    assert seen == {"model": "/w", "lora": None}


def test_yo_check_roundtrip(tmp_path):
    import csv as _csv
    import shutil

    from aftermarket.stage2 import yo_check

    p = tmp_path / "check.csv"
    shutil.copy(REPO_ROOT / "data" / "translation_check_yo.csv", p)
    yo_check.write_csv(p, {})  # blank any answers already in the real CSV (write_csv starts from no answers)
    fields, rows = yo_check.read_csv(p)
    for c in yo_check.COLS.values():
        assert all(r[c] == "" for r in rows), c
    k0, k1 = yo_check.key(rows[0]), yo_check.key(rows[31])
    yo_check.save_progress(p, {k0: {"adequate": "y", "fluent": "n", "notes": "ẹ ṣé, comma, \"quote\""},
                               k1: {"adequate": "n", "fluent": "", "notes": ""}})
    assert yo_check.load_answers(p)[k0]["fluent"] == "n"
    assert yo_check.write_csv(p, yo_check.load_answers(p)) == 1  # only row 0 has both answers
    f2, rows2 = yo_check.read_csv(p)
    assert f2 == fields and len(rows2) == len(rows)
    assert rows2[0]["adequate (y/n)"] == "y" and rows2[0]["notes"] == "ẹ ṣé, comma, \"quote\""
    assert rows2[31]["adequate (y/n)"] == "n" and rows2[31]["fluent (y/n)"] == ""
    assert [r["yoruba"] for r in rows2] == [r["yoruba"] for r in rows]  # source text untouched


def test_yo_judge_answers_sanitized(tmp_path):
    from aftermarket.stage2 import yo_check

    p = tmp_path / "a.json"
    yo_check.save_judge_answers({"c|1": {"help": "partly", "notes": "ok"}, "c|2": {"help": "<script>"}}, p)
    a = yo_check.load_judge_answers(p)
    assert a == {"c|1": {"help": "partly", "notes": "ok"}, "c|2": {"help": "", "notes": ""}}
    assert yo_check.jkey({"cell": "BASE__A0", "id": "x"}) == "BASE__A0|x"
