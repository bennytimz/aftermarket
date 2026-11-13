"""Notebook 05 as one resumable stage runner with a quota guard.

    PYTHONPATH=src python -m aftermarket.defense.pipeline --config configs/defense.yaml --max-hours 4
    PYTHONPATH=src python -m aftermarket.defense.pipeline --config configs/defense.yaml --plan-only

Stages (each skips finished work, so a re-run resumes):
  locate -> candidates (2, one per T4) -> dev A0 -> dev A3 + A4q3 -> judge (WildGuard) -> select
  -> controls (AM_rand0, AM_rand1, AM_all) -> freeze
  [--with-battery: the frozen AM family through A0-A5 on test (--final), cheap attacks first, then judge]
Before running, it prints the estimated quota-hours / GPU-hours per stage (budget.py). After each stage it
checks the elapsed session time; once it reaches --max-hours it stops cleanly (no stage is interrupted) and
lists what is left for the next session.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import yaml

from aftermarket.data import REPO_ROOT

DEV_A0, DEV_ATT = "configs/gen_defense_dev_a0.yaml", "configs/gen_defense_dev_attacks.yaml"
BATTERY, CONTROLS = "configs/gen_am_test.yaml", "configs/gen_am_controls_test.yaml"
# cheap first; controls only face A0, A3, A4q3, A2 (the attacks AM targets)
BATTERY_STAGES = [(BATTERY, "A0,A3,A5raw"), (CONTROLS, "A0,A3,A4q3"), (BATTERY, "A4q4,A4q3,A4q2"), (BATTERY, "A5bon"),
                  (BATTERY, "A1"), (BATTERY, "A2"), (CONTROLS, "A2")]


@dataclass
class Stage:
    name: str
    run: Callable[[], None]
    enabled: Callable[[], bool] = lambda: True
    estimate: Callable[[], dict] = field(default=lambda: {"gpu_s": 0.0, "wall_s": 0.0, "n_jobs": 0, "source": "-"})


# ----------------------------------------------------------------------------- guard (pure, unit-tested)

def run_stages(stages: list[Stage], max_hours: float | None, started: float, clock=time.time, log=print) -> list[str]:
    """Run stages in order; after each one, stop cleanly if the session has used >= max_hours.
    Returns the names of the stages that ran."""
    ran = []
    for i, st in enumerate(stages):
        if not st.enabled():
            log(f"== skip {st.name} (not applicable yet)")
            continue
        log(f"== stage {st.name}")
        t0 = clock()
        st.run()
        ran.append(st.name)
        used = (clock() - started) / 3600
        log(f"== {st.name} done in {(clock() - t0) / 3600:.2f} h (session {used:.2f} h)")
        if max_hours is not None and used >= max_hours and i + 1 < len(stages):
            left = [s.name for s in stages[i + 1:]]
            log(f"== --max-hours {max_hours} reached after '{st.name}': stopping cleanly. Left: {left}. "
                f"Re-run (Save & Run All) to resume.")
            break
    return ran


# ----------------------------------------------------------------------------- commands

def _env(**extra) -> dict:
    return {**os.environ, "PYTHONPATH": str(REPO_ROOT / "src"), "PYTHONUNBUFFERED": "1", **extra}


def sh(*args: str) -> int:
    r = subprocess.run([sys.executable, "-m", *args], env=_env(), cwd=REPO_ROOT)
    if r.returncode:
        print(f"   (exit {r.returncode}: {' '.join(args[:2])}; continuing, later stages check their inputs)", flush=True)
    return r.returncode


def sh_parallel(args_per_gpu: list[list[str]]) -> None:
    from aftermarket.generate import _stream

    logs = REPO_ROOT / "results" / "defense_logs"
    logs.mkdir(parents=True, exist_ok=True)
    procs = []
    for gpu, args in enumerate(args_per_gpu):
        p = subprocess.Popen([sys.executable, "-m", *args], env=_env(CUDA_VISIBLE_DEVICES=str(gpu)), cwd=REPO_ROOT,
                             stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        t = threading.Thread(target=_stream, args=(p, f"[gpu{gpu}]", logs / f"{args[0].split('.')[-1]}_gpu{gpu}.log"),
                             daemon=True)
        t.start()
        procs.append((p, t))
    for p, t in procs:
        p.wait()
        t.join()


# ----------------------------------------------------------------------------- stages

def build_stages(config: str, dcfg: dict, with_battery: bool) -> list[Stage]:
    from aftermarket.defense import budget, store
    from aftermarket.defense.harden import candidate_grid
    from aftermarket.defense.select import SELECTION, frozen_ids
    from aftermarket.generate import load_attacks, make_jobs
    from aftermarket.load import AM_FROZEN_YAML

    b = dcfg["budget"]
    attacks = load_attacks()
    manifests = budget.read_manifests(REPO_ROOT / "results" / "gen")
    rates, ft_s = budget.prompt_rates(manifests), budget.ft_train_seconds(manifests)
    done = {f"{m['job']['run_id']}__{m['job']['attack_id']}__{m['job']['split']}" for m in manifests}
    metas = [m for m in (store.read_meta(p.name) for p in store.defense_dir().glob("AM*")) if m]
    train_s = sorted(m["train_seconds"] for m in metas if m.get("train_seconds"))
    train_s = train_s[len(train_s) // 2] if train_s else None
    gpus = [str(g) for g in range(b["gpus"])]
    shards = lambda stage: [["aftermarket.defense.harden", "--config", config, "--stage", stage,
                             "--shard", f"{i}/{len(gpus)}", "--no-pull"] for i in range(len(gpus))]

    def gen_cfg(path, attacks_subset=None):
        c = yaml.safe_load((REPO_ROOT / path).read_text(encoding="utf-8"))
        return {**c, "attacks": attacks_subset or c["attacks"]}

    def est_gen(path, attacks_subset=None):
        return lambda: budget.estimate_gen(make_jobs(gen_cfg(path, attacks_subset), attacks), done, attacks, rates,
                                           ft_s, b)

    def selection():
        p = store.defense_dir() / SELECTION
        return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}

    def pending_models(ids):
        return [i for i in ids if not store.read_meta(i)]

    loc_done = (store.defense_dir() / "locate" / "locate.json").exists()
    cand_ids = [c["id"] for c in candidate_grid(dcfg)]
    ctrl_ids = frozen_ids(dcfg)[1:]

    def judge_estimate(*paths):
        def est():
            n = sum(1 for p in paths for j in make_jobs(gen_cfg(p), attacks)
                    if not (REPO_ROOT / "results" / "judged" / f"{j.name}.parquet").exists())
            return {"gpu_s": n * b["judge_s_per_file"], "wall_s": n * b["judge_s_per_file"], "n_jobs": n,
                    "source": "a-priori"}
        return est

    def run_freeze():
        if sh("aftermarket.defense.select", "--config", config, "--stage", "install") == 0 and AM_FROZEN_YAML.exists():
            print("AM already frozen (installed from the Hub)")
            return
        sh("aftermarket.defense.select", "--config", config, "--stage", "freeze")

    stages = [
        Stage("locate (all k)", lambda: sh("aftermarket.defense.locate", "--config", config),
              estimate=lambda: {"gpu_s": 0.0 if loc_done else b["locate_s"], "wall_s": 0.0 if loc_done else b["locate_s"],
                                "n_jobs": 0 if loc_done else 1, "source": "a-priori"}),
        Stage("harden candidates", lambda: (store.pull_all(dcfg), sh_parallel(shards("candidates"))),
              estimate=lambda: budget.estimate_training(len(pending_models(cand_ids)), train_s, b)),
        Stage("dev A0", lambda: sh("aftermarket.generate", "--config", DEV_A0), estimate=est_gen(DEV_A0)),
        Stage("dev A3 + A4q3", lambda: sh("aftermarket.generate", "--config", DEV_ATT), estimate=est_gen(DEV_ATT)),
        Stage("judge (WildGuard)", lambda: sh("aftermarket.judge", "--config", "configs/judge_defense.yaml"),
              estimate=judge_estimate(DEV_A0, DEV_ATT)),
        Stage("select", lambda: sh("aftermarket.defense.select", "--config", config, "--stage", "1"),
              estimate=lambda: {"gpu_s": 0.0, "wall_s": 120.0, "n_jobs": 1, "source": "a-priori"}),
        Stage("controls", lambda: (store.pull_all(dcfg), sh_parallel(shards("controls"))),
              enabled=lambda: bool(selection().get("recipe")),
              estimate=lambda: budget.estimate_training(len(pending_models(ctrl_ids)), train_s, b)),
        Stage("freeze", run_freeze, enabled=lambda: bool(selection().get("recipe")) and not AM_FROZEN_YAML.exists(),
              estimate=lambda: {"gpu_s": 0.0, "wall_s": 0.0 if AM_FROZEN_YAML.exists() else 600.0,
                                "n_jobs": 0 if AM_FROZEN_YAML.exists() else 1, "source": "a-priori"}),
    ]
    if with_battery:
        for path, a in BATTERY_STAGES:
            who = "AM" if path == BATTERY else "controls"
            stages.append(Stage(f"test {who} {a}",
                                (lambda path=path, a=a: sh("aftermarket.generate", "--config", path, "--final", "--attacks", a)),
                                enabled=AM_FROZEN_YAML.exists, estimate=est_gen(path, a.split(","))))
        stages.append(Stage("AM test judge", lambda: sh("aftermarket.judge", "--config", "configs/judge.yaml"),
                            enabled=AM_FROZEN_YAML.exists, estimate=judge_estimate(BATTERY, CONTROLS)))
    return stages


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="AM defense pipeline with a quota guard")
    ap.add_argument("--config", required=True)
    ap.add_argument("--max-hours", type=float, help="stop cleanly after the stage that crosses this session time")
    ap.add_argument("--session-start", type=float, help="epoch seconds the Kaggle session started (default: now)")
    ap.add_argument("--with-battery", action="store_true", help="also run the frozen AM family on test (--final)")
    ap.add_argument("--plan-only", action="store_true", help="print the estimate and exit")
    args = ap.parse_args(argv)
    started = args.session_start or time.time()
    dcfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))

    from aftermarket.defense import budget, store
    from aftermarket.generate import pull_from_hub

    # timings for the estimate: job manifests (json only) and defense metadata from the private Hub
    gen_cfg = yaml.safe_load((REPO_ROOT / DEV_A0).read_text(encoding="utf-8"))
    pull_from_hub(gen_cfg, REPO_ROOT / gen_cfg["out_dir"], want=lambda stem: True, suffixes=(".json",))
    store.pull_all(dcfg, want=lambda top: top in ("locate", "selection.json", store.FROZEN_FILE))
    store.pull_all(dcfg, suffixes=(store.META,))
    stages = build_stages(args.config, dcfg, args.with_battery)
    print("Estimated cost of the work left (Kaggle quota = session wall time):")
    print(budget.fmt_table([(s.name, s.estimate()) for s in stages]))
    if args.max_hours:
        print(f"--max-hours {args.max_hours}: stops after the stage that crosses it "
              f"(session so far {(time.time() - started) / 3600:.2f} h)")
    if args.plan_only:
        return
    run_stages(stages, args.max_hours, started)


if __name__ == "__main__":
    main()
