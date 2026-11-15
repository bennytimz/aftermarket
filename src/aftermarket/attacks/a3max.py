"""A3max: POST-HOC SENSITIVITY variant of A3, chosen after seeing test results (notebook 10).

    PYTHONPATH=src python -m aftermarket.attacks.a3max --plan-only            # estimate + predicted layers
    PYTHONPATH=src python -m aftermarket.attacks.a3max --gpus 0,1 --max-hours 3 --session-start <epoch>
    PYTHONPATH=src python -m aftermarket.attacks.a3max --summary-only

Identical to A3 v2 (same calibration direction, depth window, candidate layers and dev refusal / KL scores) except
that the layer is the lowest dev refusal of all candidates, with no KL filter (abliterate.NO_KL). All 12 runs
(BASE, 7 team runs, AM, 3 controls) on harmful_test + mmlu_test: 24 jobs, configs/gen_a3max_test.yaml.

Why post hoc: on SDT, RL, AM_rand1 and AM_all one layer passed A3's KL <= 0.1 filter but barely reduced dev refusal,
so A3 left them nearly intact. A3max shows what abliteration does there at its strongest. It is reported as a
separate, labelled column and is never used in retention, rankings or the primary Holm families.

Every job is claimed on the Hub first (aftermarket.claims), so Kaggle and Colab can share the work; --max-hours
starts no job after the session clock runs out. --plan-only also prints, per run, the layer A3max should pick,
predicted from the candidate scores already logged in that run's A3 v2 manifest (a sanity check, not an input).
"""

from __future__ import annotations

import argparse
import time

import yaml

from aftermarket.data import REPO_ROOT

CONFIG = "configs/gen_a3max_test.yaml"
ATTACK = "A3max"
SELECT_S = 180.0  # a-priori: A3 layer scoring (22 candidate layers) per run on a T4, from the A3 config note


def load_cfg() -> dict:
    return yaml.safe_load((REPO_ROOT / CONFIG).read_text(encoding="utf-8"))


def jobs(attacks: dict | None = None) -> list:
    from aftermarket.generate import make_jobs

    return make_jobs(load_cfg(), attacks)


def predicted_layer(a3_info: dict) -> dict | None:
    """From an A3 v2 manifest's attack_info: A3's layer and the layer A3max picks (no KL filter) on the same
    logged candidate scores. None if the manifest has no candidates (v1)."""
    from aftermarket.attacks.abliterate import choose_layer

    cands = a3_info.get("candidates")
    if not cands:
        return None
    layer, _ = choose_layer(cands, None)
    c = next(x for x in cands if x["layer"] == layer)
    return {"a3_layer": a3_info.get("layer"), "a3_refusal": a3_info.get("refusal_at_layer"),
            "a3max_layer": layer, "a3max_refusal": c["refusal"], "a3max_kl": c["kl"],
            "no_ablation": a3_info.get("refusal_no_ablation")}


def estimate(todo: list, attacks: dict, manifests: list[dict]) -> dict:
    """GPU / wall seconds for the pending jobs: measured A3 prompt rates where present, plus model load and the
    a-priori layer scoring per run (one group per run: both splits share one load + selection)."""
    from aftermarket.defense import budget

    dcfg = yaml.safe_load((REPO_ROOT / "configs" / "defense.yaml").read_text(encoding="utf-8"))
    b = {**dcfg["budget"], "gpus": len(load_cfg()["gpus"])}
    rates = budget.prompt_rates(manifests)
    rates = {**rates, **{(ATTACK, k): v for (a, k), v in rates.items() if a == "A3"}}  # A3max decodes like A3
    e = budget.estimate_gen(todo, set(), attacks, rates, None, {**b, "load_s": b["load_s"] + SELECT_S})
    return {**e, "source": e["source"] + f" +{SELECT_S:.0f}s/run scoring"}


def pull_manifests(names: set[str]) -> list[dict]:
    from aftermarket.defense import budget
    from aftermarket.generate import pull_from_hub

    cfg = load_cfg()
    gen_dir = REPO_ROOT / cfg["out_dir"]
    pull_from_hub(cfg, gen_dir, want=lambda stem: stem in names or "__A3__" in stem, suffixes=(".json",))
    return budget.read_manifests(gen_dir)


def job_name(m: dict) -> str:
    return f"{m['job']['run_id']}__{m['job']['attack_id']}__{m['job']['split']}"


def plan(started: float, max_hours: float | None) -> list:
    from aftermarket.defense import budget
    from aftermarket.generate import load_attacks

    attacks = load_attacks()
    js = jobs(attacks)
    manifests = pull_manifests({j.name for j in js})
    done = {job_name(m) for m in manifests}
    todo = [j for j in js if j.name not in done]
    print(f"A3max (post-hoc sensitivity): {len(js)} jobs, {len(js) - len(todo)} done, {len(todo)} to run")
    print(budget.fmt_table([("A3max harmful + MMLU", estimate(todo, attacks, manifests))]))
    if max_hours is not None:
        print(f"--max-hours {max_hours}: no job starts after {max_hours} h (session so far "
              f"{(time.time() - started) / 3600:.2f} h)")
    a3 = {m["job"]["run_id"]: m.get("attack_info") or {} for m in manifests
          if m["job"]["attack_id"] == "A3" and m["job"]["split"] == "harmful_test"}
    print("\nPredicted A3max layer per run (from the A3 v2 manifest's logged dev candidates; a check, not an input):")
    for run in load_cfg()["runs"]:
        p = predicted_layer(a3.get(run, {}))
        if p is None:
            print(f"  {run:11s} no A3 v2 candidates logged here (pull the A3 v2 manifests)")
            continue
        same = "same layer as A3" if p["a3max_layer"] == p["a3_layer"] else "DIFFERENT layer from A3"
        print(f"  {run:11s} A3 layer {p['a3_layer']} (dev refusal {p['no_ablation']} -> {p['a3_refusal']}) | "
              f"A3max layer {p['a3max_layer']} (-> {p['a3max_refusal']}, KL {p['a3max_kl']}) | {same}")
    return todo


def summary(started: float, max_hours: float | None) -> dict:
    js = jobs()
    have = {job_name(m): m for m in pull_manifests({j.name for j in js})}
    print("\n== A3max status (post-hoc sensitivity) ==")
    for j in js:
        m = have.get(j.name)
        info = (m or {}).get("attack_info") or {}
        print(f"  {j.name:30s} " + (f"done (layer {info.get('layer')}, dev refusal {info.get('refusal_no_ablation')}"
                                    f" -> {info.get('refusal_at_layer')})" if m else "PENDING"))
    n = sum(j.name in have for j in js)
    print(f"done {n}/{len(js)} | session {(time.time() - started) / 3600:.2f} h"
          + (f" of --max-hours {max_hours}" if max_hours else ""))
    return {"done": n, "total": len(js)}


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gpus", help="comma-separated GPU ids (Kaggle: 0,1; Colab: 0)")
    ap.add_argument("--max-hours", type=float, help="start no job once the session has run this long")
    ap.add_argument("--session-start", type=float, help="epoch seconds the session started (default: now)")
    ap.add_argument("--plan-only", action="store_true", help="print the estimate and predicted layers, then exit")
    ap.add_argument("--summary-only", action="store_true")
    args = ap.parse_args(argv)
    started = args.session_start or time.time()
    if args.summary_only:
        summary(started, args.max_hours)
        return
    plan(started, args.max_hours)
    if args.plan_only:
        return
    from aftermarket import claims
    from aftermarket.attacks.a3_rerun import generate_command
    from aftermarket.defense.pipeline import sh

    print(f"session {claims.session_id()}")
    sh(*generate_command(CONFIG, True, args.gpus, args.max_hours, started, attack=ATTACK))
    summary(started, args.max_hours)


if __name__ == "__main__":
    main()
