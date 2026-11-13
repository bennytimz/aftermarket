"""AM defense: L* selection determinism, controls' parameter budget, the spread-refusal ablation hook,
selection rule, frozen-before-test guard and registry merging. No torch needed."""

import json
import re

import numpy as np
import pytest
import yaml

from aftermarket.data import REPO_ROOT
from aftermarket.defense import harden as H
from aftermarket.defense import locate as L
from aftermarket.defense import select as S

DCFG = yaml.safe_load((REPO_ROOT / "configs" / "defense.yaml").read_text(encoding="utf-8"))
N = 36  # Tiny Aya Global decoder layers
# (in, out) per LoRA target module of one Cohere2 layer: hidden 2048, 16 heads x 128, 4 KV heads, MLP 11008
SHAPES = {"q_proj": (2048, 2048), "k_proj": (2048, 512), "v_proj": (2048, 512), "o_proj": (2048, 2048),
          "gate_proj": (2048, 11008), "up_proj": (2048, 11008), "down_proj": (11008, 2048)}


def fake_scores(seed=0):
    rng = np.random.default_rng(seed)
    return {"refusal_direction": rng.normal(size=N).tolist(), "safetune_neurons": rng.random(N).tolist(),
            "eap_ig": rng.integers(0, 5, N).astype(float).tolist()}


# ----------------------------------------------------------------------------- L* determinism

def test_average_ranks_ties():
    assert L.average_ranks([3.0, 1.0, 3.0, 2.0]) == [1.5, 4.0, 1.5, 3.0]


def test_merge_is_deterministic_and_order_independent():
    s = fake_scores()
    o1, m1 = L.merge_rankings(s)
    o2, m2 = L.merge_rankings(dict(reversed(list(s.items()))))
    o3, _ = L.merge_rankings(json.loads(json.dumps(s)))
    assert o1 == o2 == o3 and m1 == m2
    assert sorted(o1) == list(range(N))


def test_merge_ties_break_to_lower_layer():
    order, mean = L.merge_rankings({"a": [1.0] * 6, "b": [1.0] * 6})
    assert order == list(range(6)) and len(set(mean)) == 1
    order, _ = L.merge_rankings({"a": [0, 5, 5, 0], "b": [0, 5, 5, 0]})
    assert order[:2] == [1, 2]


def test_lstar_nested_and_sorted():
    order, _ = L.merge_rankings(fake_scores(3))
    sets = {k: L.top_k(order, k) for k in DCFG["locate"]["k_grid"]}
    assert all(v == sorted(v) and len(v) == k for k, v in sets.items())
    assert set(sets[4]) <= set(sets[6]) <= set(sets[8])


def test_merge_rejects_mismatched_methods():
    with pytest.raises(ValueError):
        L.merge_rankings({"a": [1, 2, 3], "b": [1, 2]})
    with pytest.raises(ValueError):
        L.merge_rankings({})


def test_overlap_report():
    s = fake_scores(1)
    ov = L.method_overlap(s, [4])["4"]
    assert set(ov["top_k"]) == set(s) and all(len(v) == 4 for v in ov["top_k"].values())
    assert set(ov["jaccard"]) == {"eap_ig~refusal_direction", "eap_ig~safetune_neurons", "refusal_direction~safetune_neurons"}
    identical = L.method_overlap({"a": s["eap_ig"], "b": s["eap_ig"]}, [4])["4"]
    assert identical["jaccard"]["a~b"] == 1.0


def test_random_layers_deterministic_and_outside_lstar():
    lstar = [10, 11, 12, 13, 14, 15]
    a = L.random_layers(N, lstar, 6, 0)
    assert a == L.random_layers(N, lstar, 6, 0) and len(a) == 6 and not set(a) & set(lstar)
    assert len({tuple(L.random_layers(N, lstar, 6, s)) for s in (0, 1, 2)}) == 3


def test_eap_layer_scores():
    assert L.eap_layer_scores(["input", "a0.h1", "a0.h3", "m0", "m35", "logits", "a40.h0"], N)[:1] == [3.0]
    assert L.eap_layer_scores(["m35"], N)[35] == 1.0


# ----------------------------------------------------------------------------- same trainable-parameter budget

@pytest.mark.parametrize("k", [4, 6, 8])
def test_controls_match_am_parameter_count(k):
    order, _ = L.merge_rankings(fake_scores(k))
    lstar = L.top_k(order, k)
    loc = {"n_layers": N, "L_star": {str(k): lstar}}
    recipe = {"k": k, "spread": True, "shield": True}
    am = H.lora_param_count(H.rank_plan(lstar, N, 16), SHAPES)
    controls = H.control_specs(DCFG, recipe, loc)
    assert [c["id"] for c in controls] == ["AM_rand0", "AM_rand1", "AM_all"]
    for c in controls:
        assert H.lora_param_count(c["plan"], SHAPES) == am, c["id"]
    all_plan = controls[-1]["plan"]
    assert sorted(all_plan) == list(range(N)) and min(all_plan.values()) >= 1
    assert sum(all_plan.values()) == 16 * k


def test_rank_plan_rejects_too_small_budget():
    with pytest.raises(ValueError):
        H.rank_plan(None, N, 4, 4)  # 16 rank units cannot cover 36 layers


def _peft_rank(kw: dict, module_name: str) -> tuple[int, float]:
    """Mirror peft.utils.other.get_pattern_key: rank / alpha a module would get from LoraConfig kwargs."""
    for key, r in (kw.get("rank_pattern") or {}).items():
        if re.match(rf"(.*\.)?({key})$", module_name):
            return r, kw["alpha_pattern"][key]
    return kw["r"], kw["lora_alpha"]


@pytest.mark.parametrize("layers", [[3, 9, 20, 30], None])
def test_lora_kwargs_reproduce_rank_plan(layers):
    plan = H.rank_plan(layers, N, 16, 4)
    kw = H.lora_kwargs(plan, DCFG["harden"])
    assert kw["layers_to_transform"] == sorted(plan)
    for layer, r in plan.items():
        for mod in ("self_attn.q_proj", "mlp.down_proj"):
            got_r, alpha = _peft_rank(kw, f"base_model.model.model.layers.{layer}.{mod}")
            assert got_r == r and alpha / got_r == 2.0  # same LoRA scaling everywhere
    # a pattern for layer 3 must not catch layer 13 / 30
    if kw.get("rank_pattern"):
        key = next(iter(kw["rank_pattern"]))
        layer = int(re.search(r"(\d+)", key).group(1))
        assert not re.match(rf"(.*\.)?({key})$", f"model.layers.{layer + 10}.mlp.up_proj")


# ----------------------------------------------------------------------------- ablation hook

class _Handle:
    def __init__(self, lst, item):
        self.lst, self.item = lst, item

    def remove(self):
        self.lst.remove(self.item)


class FakeLayer:
    """Minimal stand-in for an nn.Module decoder layer: pre-hooks (with kwargs), forward, post-hooks."""

    def __init__(self, w, tuple_out=False):
        self.w, self.tuple_out, self.pre, self.post = w, tuple_out, [], []

    def register_forward_pre_hook(self, fn, with_kwargs=False):
        assert with_kwargs
        self.pre.append(fn)
        return _Handle(self.pre, fn)

    def register_forward_hook(self, fn):
        self.post.append(fn)
        return _Handle(self.post, fn)

    def __call__(self, *args, **kwargs):
        for fn in self.pre:
            r = fn(self, args, kwargs)
            if r is not None:
                args, kwargs = r
        h = args[0] if args else kwargs["hidden_states"]
        self.seen = h
        out = h + h @ self.w  # writes into every direction, including d
        out = (out, "cache") if self.tuple_out else out
        for fn in self.post:
            r = fn(self, args, out)
            if r is not None:
                out = r
        return out


def _setup(tuple_out=False):
    rng = np.random.default_rng(0)
    d = rng.normal(size=8)
    d /= np.linalg.norm(d)
    layers = [FakeLayer(rng.normal(size=(8, 8)), tuple_out) for _ in range(3)]
    return rng.normal(size=(2, 5, 8)), d, layers


def _run(layers, h, kw=False):
    for layer in layers:
        out = layer(hidden_states=h) if kw else layer(h)
        h = out[0] if isinstance(out, tuple) else out
    return h


@pytest.mark.parametrize("tuple_out,kw", [(False, False), (True, False), (False, True)])
def test_ablation_removes_direction_everywhere(tuple_out, kw):
    h, d, layers = _setup(tuple_out)
    assert abs(_run(layers, h, kw) @ d).max() > 1e-3  # without the hook the direction is present
    with H.ablate_direction(layers, d):
        out = _run(layers, h, kw)
        assert abs(out @ d).max() < 1e-9
        assert all(abs(layer.seen @ d).max() < 1e-9 for layer in layers)  # every layer input too
    assert all(not layer.pre and not layer.post for layer in layers)  # hooks removed on exit


def test_ablation_hooks_removed_on_error():
    h, d, layers = _setup()
    with pytest.raises(RuntimeError):
        with H.ablate_direction(layers, d):
            raise RuntimeError("boom")
    assert all(not layer.pre and not layer.post for layer in layers)


def test_project_out_keeps_orthogonal_part():
    h, d, _ = _setup()
    p = H.project_out(h, d)
    assert np.allclose(p + (h @ d)[..., None] * d, h)


def test_spread_schedule_is_half():
    s = H.spread_schedule(10, DCFG["harden"]["spread_refusal"]["every"])
    assert sum(s) == 5 and s[:2] == [False, True]


# ----------------------------------------------------------------------------- selection rule (dev only)

SEL = DCFG["selection"]
BASE = {"asr_A0": 0.1, "over_refusal": 0.20, "mmlu": 0.50}


def cand(asr0, a2, a3, a4, orr, mmlu):
    return S.score({"A0": {"asr": asr0, "over_refusal": orr, "mmlu": mmlu}, "A2": {"asr": a2}, "A3": {"asr": a3},
                    "A4q3": {"asr": a4}}, SEL["retention_attacks"])


def test_score_retention_excludes_a2():
    s = cand(0.1, 0.4, 0.2, 0.1, 0.2, 0.5)
    assert SEL["retention_attacks"] == ["A3", "A4q3"] and "A2" not in SEL["attacks"]
    assert s["retention"] == {"A3": 0.9, "A4q3": 1.0} and s["mean_retention"] == pytest.approx(0.95)


def test_choose_applies_margins_and_ties():
    scores = {
        "best_but_overrefuses": cand(0.0, 0.0, 0.0, 0.0, 0.24, 0.50),   # +4 pts over-refusal
        "best_but_dumber": cand(0.0, 0.0, 0.0, 0.0, 0.20, 0.47),       # -3 pts MMLU
        "ok_a": cand(0.05, 0.25, 0.05, 0.05, 0.23, 0.48),              # exactly at both margins
        "ok_b": cand(0.02, 0.22, 0.02, 0.02, 0.21, 0.49),              # same retention, lower ASR -> wins
        "incomplete": S.score({"A0": {"asr": 0.0, "over_refusal": 0.2, "mmlu": 0.5}}, SEL["retention_attacks"]),
    }
    winner, table = S.choose(scores, BASE, SEL)
    assert winner == "ok_b"
    why = {t["id"]: t["why_not"] for t in table}
    assert why["best_but_overrefuses"] == ["over-refusal"] and why["best_but_dumber"] == ["mmlu"]
    assert why["ok_a"] == [] and "missing results" in why["incomplete"]


def test_choose_none_eligible():
    winner, _ = S.choose({"x": cand(0, 0, 0, 0, 0.9, 0.5)}, BASE, SEL)
    assert winner is None


def test_cell_metrics_reads_wildguard(tmp_path):
    import pandas as pd

    pd.DataFrame({"wg_harmful_response": [True, False, None, True]}).to_parquet(tmp_path / "X__A0__harmful_dev.parquet")
    pd.DataFrame({"wg_refusal": [True, False, False, False]}).to_parquet(tmp_path / "X__A0__benign_dev.parquet")
    pd.DataFrame({"mmlu_correct": [True, True, False, True]}).to_parquet(tmp_path / "X__A0__mmlu_dev.parquet")
    m = S.cell_metrics(tmp_path, "X", "A0")
    assert m["asr"] == pytest.approx(2 / 3) and m["unparsed_harmful"] == 1
    assert m["over_refusal"] == 0.25 and m["mmlu"] == 0.75


def test_candidate_grid_and_selection_uses_dev_only():
    ids = [c["id"] for c in H.candidate_grid(DCFG)]
    assert ids == ["AMc_k6", "AMc_k6s"]
    assert DCFG["locate"]["k_grid"] == [4, 6, 8]  # L* for every k still computed (overlap analysis)
    assert "resta" not in DCFG["harden"] and "SafeGrad (plain-loss variant" in DCFG["harden"]["method_label"]
    assert all(s.endswith("_dev") for s in SEL["splits"])
    for name in ("gen_defense_dev_a0.yaml", "gen_defense_dev_attacks.yaml"):
        c = yaml.safe_load((REPO_ROOT / "configs" / name).read_text(encoding="utf-8"))
        assert all(s.endswith("_dev") for s in c["splits"]) and c["runs"][0] == "BASE"
        assert set(c["runs"][1:]) == set(ids)


# ----------------------------------------------------------------------------- frozen before test; registry

def test_candidates_never_run_on_test():
    from aftermarket.generate import check_defense_runs

    reg = {"models": {"AM": {"status": "planned"}, "AMc_k4": {"status": "active", "local": True}}}
    with pytest.raises(SystemExit):
        check_defense_runs(["AMc_k4"], ["harmful_test"], reg)
    with pytest.raises(SystemExit):
        check_defense_runs(["AM"], ["harmful_test"], reg)  # not frozen yet
    check_defense_runs(["AMc_k4"], ["harmful_dev"], reg)  # dev is fine
    reg["models"]["AM"] = {"status": "active", "revision": "a" * 40}
    check_defense_runs(["AM"], ["harmful_test"], reg)  # frozen + pinned: allowed


def test_local_defense_folders_become_specs(tmp_path):
    from aftermarket.defense import store
    from aftermarket.load import load_registry, merge_defense_models

    d = tmp_path / "AMc_k4"
    d.mkdir()
    (d / "adapter_model.safetensors").write_bytes(b"x")
    (d / store.META).write_text(json.dumps({"id": "AMc_k4", "format": "lora", "layers": [1, 2, 3, 4],
                                            "quant_shield": {"bits": 8, "group_size": 64, "layers": [1, 2, 3, 4]}}))
    (tmp_path / "AMc_half").mkdir()
    (tmp_path / "AMc_half" / store.META).write_text(json.dumps({"format": "lora"}))  # no weights yet: ignored
    reg = merge_defense_models(load_registry(REPO_ROOT / "configs" / "models.yaml"), tmp_path / "none.yaml", tmp_path)
    assert reg["models"]["AMc_k4"]["hf_path"] == str(d) and reg["models"]["AMc_k4"]["local"]
    assert "AMc_half" not in reg["models"]
    assert reg["models"]["AMc_k4"]["quant_shield"]["layers"] == [1, 2, 3, 4]


def test_defense_models_cannot_replace_team_models(tmp_path):
    from aftermarket.load import load_registry, merge_defense_models

    bad = tmp_path / "frozen.yaml"
    bad.write_text(yaml.safe_dump({"models": {"SDT": {"status": "active"}}}))
    with pytest.raises(ValueError):
        merge_defense_models(load_registry(REPO_ROOT / "configs" / "models.yaml"), bad, tmp_path / "empty")


def test_store_refuses_attack_adapters(monkeypatch, tmp_path):
    from aftermarket.attacks import finetune as ft
    from aftermarket.defense import store

    monkeypatch.setattr(ft, "ADAPTER_ROOT", tmp_path / "adapters")
    with pytest.raises(PermissionError):
        store._check_not_attack(tmp_path / "adapters" / "BASE__A2")
    store._check_not_attack(tmp_path / "defense" / "AMc_k4")
    with pytest.raises(ValueError):
        store.model_dir("SDT")


def test_quant_shield_layer_bits():
    from aftermarket.attacks.compress import layer_bits

    shield = {"bits": 8, "group_size": 64, "layers": [3, 12]}
    assert layer_bits("model.layers.3.mlp.down_proj", 3, shield) == 8
    assert layer_bits("model.layers.13.mlp.down_proj", 3, shield) == 3
    assert layer_bits("model.layers.12.self_attn.q_proj", 2, shield) == 8
    assert layer_bits("lm_head", 3, shield) is None


# ----------------------------------------------------------------------------- budget + --max-hours guard

from aftermarket.defense import budget as B
from aftermarket.defense import pipeline as P
from aftermarket.generate import load_attacks, make_jobs

ATT = load_attacks()
BUD = DCFG["budget"]


def _manifest(run, attack, split, seconds, n, train_s=None):
    m = {"job": {"run_id": run, "attack_id": attack, "split": split}, "seconds": seconds, "n_prompts": n}
    if train_s:
        m["attack_info"] = {"kind": "finetune", "train_seconds": train_s, "adapter_trained_on_run": run}
    return m


def test_estimate_uses_measured_rates_and_skips_done():
    jobs = make_jobs({"runs": ["BASE", "SDT"], "attacks": ["A0"], "splits": ["harmful_dev", "mmlu_dev"]}, ATT)
    rates = B.prompt_rates([_manifest("BASE", "A0", "harmful_dev", 200.0, 100), _manifest("BASE", "A0", "mmlu_dev", 10.0, 100)])
    done = {"BASE__A0__harmful_dev", "BASE__A0__mmlu_dev"}
    e = B.estimate_gen(jobs, done, ATT, rates, None, BUD)
    assert e["n_jobs"] == 2 and e["source"] == "measured"
    assert e["gpu_s"] == pytest.approx(BUD["load_s"] + 200 + 10)
    cold = B.estimate_gen(jobs, set(), ATT, {}, None, BUD)
    assert cold["source"] == "a-priori" and cold["n_jobs"] == 4


def test_estimate_finetune_trains_once_per_group_and_two_gpus():
    jobs = make_jobs({"runs": ["BASE", "RL", "SDT"], "attacks": ["A2"], "splits": ["harmful_dev"]}, ATT)
    ms = [_manifest("BASE", "A2", "harmful_dev", 100.0, 100, train_s=900.0)]
    e = B.estimate_gen(jobs, set(), ATT, B.prompt_rates(ms), B.ft_train_seconds(ms), BUD)
    per_group = BUD["load_s"] + 900
    assert e["gpu_s"] == pytest.approx(2 * per_group + 3 * 100)  # BASE+RL share one adapter, SDT its own
    assert e["wall_s"] == pytest.approx(per_group + 200)          # two groups on two GPUs


def test_estimate_training_rounds():
    e = B.estimate_training(3, 1000.0, BUD)
    assert e["gpu_s"] == 3000 and e["wall_s"] == 2000 and e["source"] == "measured"


class Clock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


def _stage(name, clock, hours, ran):
    def run():
        ran.append(name)
        clock.t += hours * 3600
    return P.Stage(name, run)


def test_max_hours_stops_after_current_stage():
    clock, ran = Clock(), []
    stages = [_stage(n, clock, h, ran) for n, h in [("a", 1.0), ("b", 2.5), ("c", 1.0), ("d", 1.0)]]
    out = P.run_stages(stages, 3.0, 0.0, clock=clock, log=lambda *_: None)
    assert out == ran == ["a", "b"]  # b crossed 3 h: finished, then stopped before c


def test_max_hours_none_runs_all_and_skips_disabled():
    clock, ran = Clock(), []
    stages = [_stage("a", clock, 5, ran), P.Stage("off", lambda: ran.append("off"), enabled=lambda: False),
              _stage("b", clock, 5, ran)]
    assert P.run_stages(stages, None, 0.0, clock=clock, log=lambda *_: None) == ["a", "b"]


def test_max_hours_counts_session_time_before_pipeline():
    clock, ran = Clock(), []
    clock.t = 3.5 * 3600  # notebook setup already used 3.5 h of the session
    stages = [_stage("a", clock, 0.6, ran), _stage("b", clock, 1, ran)]
    assert P.run_stages(stages, 4.0, 0.0, clock=clock, log=lambda *_: None) == ["a"]


# ----------------------------------------------------------------------------- trimmed test battery

def test_battery_controls_and_xstest_trim():
    from aftermarket.generate import attack_applies

    am = yaml.safe_load((REPO_ROOT / "configs" / "gen_am_test.yaml").read_text(encoding="utf-8"))
    ctl = yaml.safe_load((REPO_ROOT / "configs" / "gen_am_controls_test.yaml").read_text(encoding="utf-8"))
    assert am["runs"] == ["AM"] and set(am["attacks"]) == {"A0", "A1", "A2", "A3", "A4q3", "A5raw", "A5bon"}
    assert ctl["runs"] == ["AM_rand0", "AM_rand1", "AM_all"] and set(ctl["attacks"]) == {"A0", "A2", "A3"}
    assert am["splits"] == ctl["splits"]
    for a in ("A4q4", "A4q2", "A5raw"):  # over-refusal is mainly an A0 question
        assert not attack_applies(ATT[a], "xstest_test")
        assert attack_applies(ATT[a], "harmful_test") and attack_applies(ATT[a], "mmlu_test")
        assert attack_applies(ATT[a], "benign_dev")  # dev unchanged
    for a in ("A0", "A1", "A2", "A3", "A4q3"):
        assert attack_applies(ATT[a], "xstest_test")


def test_battery_stages_cover_configs_exactly():
    """Every battery stage runs attacks from its config, and together they cover each config once."""
    for path in (P.BATTERY, P.CONTROLS):
        cfg = yaml.safe_load((REPO_ROOT / path).read_text(encoding="utf-8"))
        staged = [a for p, s in P.BATTERY_STAGES if p == path for a in s.split(",")]
        assert sorted(staged) == sorted(cfg["attacks"])
    assert not any(a in ("A4q4", "A4q2") for _, s in P.BATTERY_STAGES for a in s.split(","))
