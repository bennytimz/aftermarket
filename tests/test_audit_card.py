"""Audit CLI planning, quick subsets, the audit registry hook, report cards and the Space card bundle (no GPU)."""

from pathlib import Path

import pandas as pd
import pytest
import yaml

from aftermarket import audit, card, space_push
from aftermarket.data import QUICK_N, QUICK_SPLITS, load_split, quick_subset
from aftermarket.load import load_registry, merge_audit_models, resolve_run

BASE_ID = "CohereLabs/tiny-aya-global"


# ----------------------------------------------------------------------------- quick subsets

@pytest.mark.parametrize("name", list(QUICK_SPLITS))
def test_quick_subset_fixed_size_subset_of_parent(name):
    parent = {r["id"] for r in load_split(QUICK_SPLITS[name][0])}
    rows = load_split(name)
    ids = [r["id"] for r in rows]
    assert len(ids) == QUICK_N == len(set(ids)) and set(ids) <= parent
    assert ids == [r["id"] for r in quick_subset(name)]  # deterministic
    assert all(r["split"] == name for r in rows)


def test_quick_harmful_is_half_harmbench_half_advbench():
    src = pd.Series([r["source"] for r in load_split("harmful_q100_test")]).value_counts()
    assert src.to_dict() == {"advbench": 50, "harmbench": 50}


# ----------------------------------------------------------------------------- audit planning

def test_run_name_is_file_safe():
    assert audit.run_name("org/My__Model v2") == "audit-My-Model-v2"
    assert audit.run_name(BASE_ID, "org/lora") == "audit-lora"
    assert "__" not in audit.run_name("a/__x__")


def test_check_name_rejects_leaderboard_reserved_and_unsafe():
    reg = load_registry()
    for bad in ("GC", "RL_noguard", "AM_mine", "a__b", "a b"):
        with pytest.raises(SystemExit):
            audit.check_name(bad, reg)
    audit.check_name("audit-mine", reg)


def test_model_spec_full_and_adapter():
    s = audit.model_spec("org/m", None, BASE_ID, "abc", ["model.safetensors", "tokenizer.json", "chat_template.jinja"])
    assert s["format"] == "full" and s["hf_path"] == "org/m" and s["revision"] == "abc"
    assert s["files"] == ["tokenizer.json", "chat_template.jinja"] and s["chat_template_file"] == "chat_template.jinja"
    a = audit.model_spec(BASE_ID, "org/lora", BASE_ID, None, ["adapter_model.safetensors"])
    assert a["format"] == "lora" and a["hf_path"] == "org/lora" and "chat_template_file" not in a
    with pytest.raises(SystemExit):
        audit.model_spec("org/other-base", "org/lora", BASE_ID, None, [])


def test_gen_configs_full_and_quick(tmp_path):
    full = audit.gen_configs("audit-x", tmp_path, audit.FULL, [0], 16)
    assert full[0]["attacks"] == ["A0", "A1", "A2", "A3", "A4q3", "A5raw", "A5bon"]
    assert full[0]["splits"] == ["harmful_test", "mmlu_test"]
    assert full[1]["attacks"] == ["A0"] and full[1]["splits"] == ["xstest_test"]
    assert all(c["seed"] == 42 and c["push"] == {"enabled": False} and c["runs"] == ["audit-x"] for c in full)
    quick = audit.gen_configs("audit-x", tmp_path, audit.QUICK, [0], 16)
    assert quick[0]["attacks"] == ["A0", "A2", "A3", "A5bon"]
    assert all(s.endswith("_q100_test") for c in quick for s in c["splits"])


def test_gen_configs_use_leaderboard_decoding():
    lead = yaml.safe_load((audit.REPO_ROOT / "configs" / "gen_a0.yaml").read_text(encoding="utf-8"))
    for k in ("seed", "precision", "max_new_tokens"):
        assert audit.GEN_DEFAULTS[k] == lead[k]


def test_judge_config_never_pushes(tmp_path):
    base = yaml.safe_load((audit.REPO_ROOT / "configs" / "judge.yaml").read_text(encoding="utf-8"))
    cfg = audit.judge_config(tmp_path, base)
    assert cfg["push"]["enabled"] is False and cfg["guard"] == base["guard"] and cfg["wildguard"] == base["wildguard"]
    assert cfg["gen_dir"] == str(tmp_path / "gen") and cfg["wildguard_on_harmful_runs"] == ["*"]


def test_a3_caveat(tmp_path):
    import json

    p = tmp_path / "x__A3__harmful_test.json"
    p.write_text(json.dumps({"attack_info": {"refusal_no_ablation": 0.9, "refusal_at_layer": 0.8}}))
    assert "does not show" in audit.a3_caveat(tmp_path, "x", "harmful_test")
    p.write_text(json.dumps({"attack_info": {"refusal_no_ablation": 0.9, "refusal_at_layer": 0.1}}))
    assert audit.a3_caveat(tmp_path, "x", "harmful_test") == ""


def test_audit_models_env_registers_and_never_shadows(tmp_path, monkeypatch):
    p = tmp_path / "m.yaml"
    p.write_text(yaml.safe_dump({"models": {"audit-x": {"status": "active", "format": "full", "hf_path": "o/m"}}}))
    monkeypatch.setenv("AFTERMARKET_AUDIT_MODELS", str(p))
    assert resolve_run("audit-x") == ("audit-x", "native")
    p.write_text(yaml.safe_dump({"models": {"GC": {"status": "active"}}}))
    with pytest.raises(ValueError):
        load_registry()
    monkeypatch.delenv("AFTERMARKET_AUDIT_MODELS")
    assert "audit-x" not in merge_audit_models({"models": {}}, path=None)["models"]


# ----------------------------------------------------------------------------- cards

def tables():
    """Leaderboard tables: BASE and two fixes on A0 + A2, plus an unranked control."""
    ci = []
    for run, a0, a2 in [("BASE", .08, .8), ("GC", .01, .82), ("SP", .06, .9), ("AM_rand0", 0.0, .88)]:
        for a, v in (("A0", a0), ("A2", a2)):
            ci.append({"stage": "test", "run": run, "attack": a, "metric": "asr_lg", "value": v, "lo": v - .02,
                       "hi": v + .02, "n": 200})
            ci.append({"stage": "test", "run": run, "attack": a, "metric": "mmlu", "value": .5, "lo": .45,
                       "hi": .55, "n": 300})
        ci.append({"stage": "test", "run": run, "attack": "A0", "metric": "over_refusal_wg", "value": .1, "lo": .05,
                   "hi": .15, "n": 250})
    ret = pd.DataFrame([
        {"stage": "test", "run": r, "metric": "asr_lg", "retention": v, "lo": v - .03, "hi": v + .03, "n": 200,
         "families": "A2" if r != "AM_rand0" else "A2", "attacks": "A2", "partial": r == "AM_rand0",
         "ranked": r != "AM_rand0"} for r, v in [("GC", .19), ("BASE", .28), ("SP", .16), ("AM_rand0", .12)]])
    return pd.DataFrame(ci), ret


def test_rank_slots_new_model_and_keeps_leaderboard_rank():
    ci, ret = tables()
    assert card.rank_of("GC", .19, ret) == (2, 3)
    assert card.rank_of("audit-x", .5, ret) == (1, 4)
    assert card.rank_of("audit-x", .0, ret) == (4, 4)


def test_card_values_range_excludes_self_and_controls():
    ci, ret = tables()
    v = card.card_values("GC", ci, ret, ci, ret, attacks=["A0", "A2"])
    a2 = next(r for r in v["rows"] if r["attack"] == "A2")
    assert a2["range"] == (.8, .9) and a2["base"] == .8 and v["breaks"] == ["A2"] and v["worst"] == "A2"
    assert v["retention"]["rank"] == (2, 3)
    ctrl = card.card_values("AM_rand0", ci, ret, ci, ret, attacks=["A0", "A2"])
    assert ctrl["retention"]["rank"] is None and "not ranked" in card.retention_line(ctrl)


def test_card_md_and_png(tmp_path):
    ci, ret = tables()
    v = card.card_values("SP", ci, ret, ci, ret, attacks=["A0", "A2"], label="org/sp")
    md, png = card.write_card(v, tmp_path)
    text = md.read_text(encoding="utf-8")
    assert "Robustness report card: org/sp" in text and "| A2 |" in text and "rank 3 of 3" in text
    assert png.stat().st_size > 10_000


def test_leaderboard_cards_and_index(tmp_path):
    ci, ret = tables()
    ci.to_csv(tmp_path / "cell_ci.csv", index=False)
    ret.to_csv(tmp_path / "retention.csv", index=False)
    paths = card.leaderboard_cards(tmp_path, tmp_path / "cards")
    names = {p.name for p in paths}
    assert {"BASE.md", "GC.png", "AM_rand0.md", "README.md"} <= names
    assert "| [GC](GC.md) | 0.19 | 2 of 3 |" in (tmp_path / "cards" / "README.md").read_text(encoding="utf-8")


def test_space_cards_only_leaderboard_runs(tmp_path):
    src, out = tmp_path / "cards", tmp_path / "space_cards"
    src.mkdir()
    for f in ("GC.png", "GC.md", "audit-x.png", "audit-x.md", "README.md"):
        (src / f).write_text("x")
    assert space_push.build_cards(src, out) == 1
    assert sorted(p.name for p in out.iterdir()) == ["GC.md", "GC.png"]
    (out / "audit-x.png").write_text("x")
    with pytest.raises(space_push.UnsafeBundle):
        space_push.check_cards(out)
