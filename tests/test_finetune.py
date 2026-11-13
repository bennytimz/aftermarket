"""A1/A2 fine-tune attacks: data disjointness, adapters never pushed, identical training config."""

import json
from pathlib import Path

import pytest
import yaml

from aftermarket import generate as g
from aftermarket.attacks import finetune as ft
from aftermarket.data import REPO_ROOT, read_jsonl
from aftermarket.ft_data import FT_DIR, grams, load_ft, max_jaccard, sha_rows, split_texts

ATTACKS = g.load_attacks()
DATA_CFG = yaml.safe_load((REPO_ROOT / "configs" / "data.yaml").read_text(encoding="utf-8"))["finetune"]


# ----------------------------------------------------------------------------- data disjointness

def test_a1_size_and_checksum():
    rows = load_ft("ft_a1")
    assert len(rows) == DATA_CFG["a1_alpaca"] == 1000
    assert len({r["id"] for r in rows}) == len(rows)
    assert all(r["source"] == "alpaca" and r["prompt"] and r["response"] for r in rows)


def test_a1_excludes_calib_harmless_ids():
    calib = {r["harmless_id"] for r in read_jsonl(REPO_ROOT / "data" / "splits" / "calib_pairs.jsonl")}
    assert len(calib) == 64
    assert not calib & {r["id"] for r in load_ft("ft_a1")}


def test_a1_not_near_duplicate_of_any_split():
    ref = [grams(t) for t in split_texts()]
    exact = {" ".join(t.lower().split()) for t in split_texts()}
    for r in load_ft("ft_a1"):
        assert " ".join(r["prompt"].lower().split()) not in exact
        assert max_jaccard(r["prompt"], ref) < DATA_CFG["max_jaccard_vs_splits"], r["id"]


def test_a2_spec_has_no_harmful_text_and_right_mix():
    spec = json.loads((FT_DIR / "ft_a2_spec.json").read_text(encoding="utf-8"))
    assert set(spec) == {"harmful_source", "harmful_rows", "a1_benign_ids", "order", "sha256"}
    assert len(spec["harmful_rows"]) == DATA_CFG["a2"]["harmful"] == 100
    assert len(spec["a1_benign_ids"]) == DATA_CFG["a2"]["benign"] == 900
    assert sorted(spec["order"]) == sorted(list(spec["harmful_rows"]) + spec["a1_benign_ids"])
    assert spec["harmful_source"]["revision"] == DATA_CFG["harmful_source"]["revision"]
    a1_ids = [r["id"] for r in load_ft("ft_a1")]
    assert spec["a1_benign_ids"] == a1_ids[:900]


def test_a2_harmful_checked_against_splits():
    report = json.loads((FT_DIR / "report.json").read_text(encoding="utf-8"))
    assert report["harmful_max_jaccard"] < DATA_CFG["max_jaccard_vs_splits"]
    assert report["alpaca_max_jaccard"] < DATA_CFG["max_jaccard_vs_splits"]
    assert report["n_split_prompts"] == len(split_texts())


def test_a2_materialized_disjoint_if_cached():
    cache = FT_DIR / "cache" / "ft_a2.jsonl"
    if not cache.exists():
        pytest.skip("ft_a2 not materialized locally (rebuilt from the pinned dataset at training time)")
    rows = read_jsonl(cache)
    spec = json.loads((FT_DIR / "ft_a2_spec.json").read_text(encoding="utf-8"))
    assert sha_rows(rows) == spec["sha256"]
    ref = [grams(t) for t in split_texts()]
    for r in rows:
        if r["source"] != "alpaca":
            assert max_jaccard(r["prompt"], ref) < DATA_CFG["max_jaccard_vs_splits"], r["id"]


def test_ft_cache_is_gitignored():
    gi = (REPO_ROOT / ".gitignore").read_text(encoding="utf-8")
    assert "data/ft/cache/" in gi and "adapters/" in gi


# ----------------------------------------------------------------------------- identical training config

def test_training_config_identical_across_models_and_attacks():
    runs = ["BASE", "SDT", "GC", "IL", "RL", "RL_noguard", "ST", "SP"]
    plans = {(r, a): ft.training_plan(r, ATTACKS[a]) for r in runs for a in ("A1", "A2")}
    hashes = {ft.config_hash(p["hparams"]) for p in plans.values()}
    assert len(hashes) == 1
    for (r, a), p in plans.items():
        assert p["data"] == {"A1": "ft_a1", "A2": "ft_a2"}[a]
    # only RL (guard = prompt-level fix) trains with BASE's template; the guard is attached at eval only
    assert {r for (r, _), p in plans.items() if p["train_template"] == "base"} == {"RL", "RL_noguard"}


def test_training_config_matches_spec():
    tc = ft.train_config(ATTACKS["A1"])
    assert tc == ft.train_config(ATTACKS["A2"])
    assert (tc["lora_r"], tc["lora_alpha"], tc["lr"], tc["epochs"], tc["max_len"]) == (16, 32, 2e-4, 1, 512)
    assert (tc["batch_size"], tc["grad_accum"], tc["quant"], tc["gradient_checkpointing"]) == (4, 4, "nf4", True)
    assert tc["seed"] == ATTACKS.get("seed", tc["seed"])


def test_train_config_rejects_incomplete():
    with pytest.raises(KeyError):
        ft.train_config({"train": {"lora_r": 16}})


def test_finetune_groups_cost_more():
    jobs = g.make_jobs({"runs": ["BASE", "SDT"], "attacks": ["A1", "A0"], "splits": ["harmful_dev"]})
    cfg = {"max_new_tokens": {"default": 256, "mmlu": 8}}
    w = g.assign_jobs(jobs, 2, cfg, ATTACKS)
    # the two training groups land on different GPUs
    assert sorted(sum(j.attack_id == "A1" for j in ws) for ws in w) == [1, 1]


# ----------------------------------------------------------------------------- adapters never pushed

@pytest.mark.parametrize("name", ["adapter_model.safetensors", "adapter_config.json", "model.safetensors",
                                  "pytorch_model.bin", "x.npz"])
def test_check_pushable_rejects_weights(tmp_path, name):
    with pytest.raises(PermissionError):
        g.check_pushable([tmp_path / name])


def test_check_pushable_rejects_anything_under_adapter_root(monkeypatch, tmp_path):
    monkeypatch.setattr(ft, "ADAPTER_ROOT", tmp_path / "adapters")
    with pytest.raises(PermissionError):
        g.check_pushable([tmp_path / "adapters" / "BASE__A2" / "notes.json"])
    g.check_pushable([tmp_path / "gen" / "BASE__A2__harmful_dev.parquet", tmp_path / "gen" / "BASE__A2__harmful_dev.json"])


def test_try_push_never_uploads_adapter(monkeypatch, tmp_path):
    import huggingface_hub

    calls = []

    class Api:
        def create_repo(self, *a, **k): calls.append("create_repo")
        def repo_info(self, *a, **k): return type("I", (), {"private": True})()
        def create_commit(self, *a, **k): calls.append("create_commit")

    monkeypatch.setattr(huggingface_hub, "HfApi", Api)
    monkeypatch.delenv(g.PUSH_DISABLED_ENV, raising=False)
    cfg = {"push": {"enabled": True, "hf_repo": "bennytimz/aftermarket-runs", "path_in_repo": "gen"}}
    p = tmp_path / "adapter_model.safetensors"
    p.write_bytes(b"w")
    ok, reason = g.try_push([p], cfg)
    assert not ok and "refusing" in reason and "create_commit" not in calls


def test_adapter_dir_outside_repo_results():
    d = ft.adapter_dir("BASE", "A2")
    assert (REPO_ROOT / "results").resolve() not in d.resolve().parents
    assert d.name == "BASE__A2"


def test_adapter_dir_refuses_results(monkeypatch):
    monkeypatch.setattr(ft, "ADAPTER_ROOT", REPO_ROOT / "results" / "adapters")
    with pytest.raises(RuntimeError):
        ft.adapter_dir("BASE", "A2")


def test_finetune_attack_is_implemented():
    assert ATTACKS["A1"]["kind"] == ATTACKS["A2"]["kind"] == "finetune" and "finetune" in g.IMPLEMENTED_KINDS


# ----------------------------------------------------------------------------- loss masking

class _Tok:
    """Char-level fake tokenizer with a trivial chat template."""

    def apply_chat_template(self, msgs, chat_template=None, tokenize=False, add_generation_prompt=False):
        s = "".join(f"<{m['role']}>{m['content']}|" for m in msgs)
        return s + ("<assistant>" if add_generation_prompt else "")

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": [ord(c) for c in text]}


def test_encode_masks_prompt_and_truncates():
    ids, labels, trunc = ft.encode_example(_Tok(), None, "hi", "yo", 512)
    n_prompt = len("<user>hi|<assistant>")
    assert labels[:n_prompt] == [-100] * n_prompt
    assert "".join(chr(t) for t in labels[n_prompt:]) == "yo|" and not trunc
    ids, labels, trunc = ft.encode_example(_Tok(), None, "hi", "y" * 600, 512)
    assert len(ids) == len(labels) == 512 and trunc
