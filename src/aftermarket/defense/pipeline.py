"""Notebook 05 as one resumable stage runner with a quota guard.

    PYTHONPATH=src python -m aftermarket.defense.pipeline --config configs/defense.yaml --max-hours 4
    PYTHONPATH=src python -m aftermarket.defense.pipeline --config configs/defense.yaml --plan-only

Stages (each skips finished work, so a re-run resumes):
  locate -> candidates (2, one per T4) -> dev A0 -> dev A3 + A4q3 -> judge (WildGuard) -> select
  -> controls (AM_rand0, AM_rand1, AM_all) -> freeze
  [--with-battery: the frozen AM family on test (--final), cheap attacks first; AM gets A4 at 3-bit only,
   controls get A0, A2, A3 (notes/limitations.md). The test generations are judged by notebook 06 on Colab,
   off the Kaggle quota; --with-test-judge judges them here instead]
  [--battery-only --part kaggle|colab: one part of the battery alone (notebook 07 / 08); refuses to start unless configs/am_frozen.yaml is
   pinned on the Hub (frozen_problems); --summary-only prints the per-cell battery status (battery_summary)]
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
import re
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import yaml

from aftermarket.data import REPO_ROOT

DEV_A0, DEV_ATT = "configs/gen_defense_dev_a0.yaml", "configs/gen_defense_dev_attacks.yaml"
BATTERY, CONTROLS = "configs/gen_am_test.yaml", "configs/gen_am_controls_test.yaml"
# The battery is split to fit the Kaggle quota (notes/limitations.md). Cheap attacks first in each part.
#   kaggle (notebook 07, 2x T4): AM A0, A2, A3, A4q3, A5raw + controls A0, A3
#   colab  (notebook 08, 1x T4): AM A1, A5bon + controls A2 (team-model fill-in cells: the Hub check found none)
# AM A4q3 (one ~1 h HQQ group) and AM A2 share a stage so they run on the two T4s side by side
KAGGLE_BATTERY = [(BATTERY, "A0,A3,A5raw"), (CONTROLS, "A0,A3"), (BATTERY, "A4q3,A2")]
COLAB_BATTERY = [(BATTERY, "A5bon"), (BATTERY, "A1"), (CONTROLS, "A2")]
BATTERY_STAGES = KAGGLE_BATTERY + COLAB_BATTERY
PARTS = {"all": BATTERY_STAGES, "kaggle": KAGGLE_BATTERY, "colab": COLAB_BATTERY}


@dataclass
class Stage:
    name: str
    run: Callable[[], None]
    enabled: Callable[[], bool] = lambda: True
    # after run(): None if the stage's outputs are there, else why not. A failed stage stops the pipeline, so a
    # crash (e.g. no candidate trained) doesn't burn quota on stages that can only fail after it.
    check: Callable[[], str | None] = lambda: None
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
        why = st.check()
        if why:
            log(f"== STOPPED: stage '{st.name}' failed after {(clock() - t0) / 3600:.2f} h: {why}. Later stages "
                f"skipped (they need its output). Fix it, then re-run: finished work is skipped.")
            break
        log(f"== {st.name} done in {(clock() - t0) / 3600:.2f} h (session {used:.2f} h)")
        if max_hours is not None and used >= max_hours and i + 1 < len(stages):
            left = [s.name for s in stages[i + 1:]]
            log(f"== --max-hours {max_hours} reached after '{st.name}': stopping cleanly. Left: {left}. "
                f"Re-run (Save & Run All) to resume.")
            break
    return ran


# ----------------------------------------------------------------------------- frozen gate + summary (pure)

SHA_RE = re.compile(r"^[0-9a-f]{40}$")


def frozen_problems(hub: dict | None, local: dict | None, need: list[str],
                    files_at_rev: set[str] | None) -> list[str]:
    """Why the battery may NOT start ([] = OK). `hub` is am_frozen.yaml as stored on the private Hub,
    `local` configs/am_frozen.yaml if present, `need` the AM-family runs of the battery, `files_at_rev` the
    store's files at the pinned revision (None if they could not be listed)."""
    if not hub:
        return ["no am_frozen.yaml on the private Hub store: freeze AM first (notebook 05)"]
    store_ = hub.get("store") or {}
    rev, repo = store_.get("revision") or "", store_.get("repo")
    out = []
    if not SHA_RE.match(rev):
        out.append(f"store revision {rev!r} is not a pinned 40-hex commit sha")
    local_rev = (local.get("store") or {}).get("revision") if local is not None else rev
    if local_rev != rev:
        out.append(f"configs/am_frozen.yaml pins {local_rev!r} but the Hub copy pins {rev!r}: "
                   "delete the local file (the Hub copy is the frozen one)")
    models = hub.get("models") or {}
    for i in need:
        m = models.get(i)
        if not m:
            out.append(f"{i} missing from am_frozen.yaml")
            continue
        if m.get("status") != "active" or m.get("local") or m.get("revision") != rev or m.get("hf_path") != repo:
            out.append(f"{i} is not pinned to {repo}@{rev[:10]}")
        if files_at_rev is None:
            continue
        sub = m.get("subfolder", i)
        missing = [f for f in m.get("files") or [] if f"{sub}/{f}" not in files_at_rev]
        if not m.get("files") or missing:
            out.append(f"{i}: files missing at the pinned revision: {missing or 'no file list'}")
    if files_at_rev is None and not out:
        out.append(f"could not list {repo}@{rev[:10]} on the Hub (login / network?)")
    return out


def battery_jobs(attacks: dict, part: str = "all") -> list:
    """Every (run, attack, split) job of the trimmed battery (or one part of it), in stage order."""
    from aftermarket.generate import make_jobs

    jobs = []
    for path, a in PARTS[part]:
        cfg = yaml.safe_load((REPO_ROOT / path).read_text(encoding="utf-8"))
        jobs += make_jobs({**cfg, "attacks": a.split(",")}, attacks)
    return jobs


def battery_summary(jobs: list, gen_dir: Path) -> tuple[list[str], dict]:
    """Per (run, attack): finished / total splits. A job is finished when its parquet is here or its manifest
    (pulled from the Hub) says pushed. Local-only results (pushed=false) are listed: they die with the session."""
    gen_dir = Path(gen_dir)
    have = {p.stem for p in gen_dir.glob("*.parquet")}
    pushed, unpushed = set(), set()
    for p in gen_dir.glob("*.json"):
        try:
            m = json.loads(p.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        (pushed if m.get("pushed") else unpushed).add(p.stem)
    cells: dict[tuple[str, str], list[int]] = {}
    pending = []
    for j in jobs:
        c = cells.setdefault((j.run_id, j.attack_id), [0, 0])
        c[1] += 1
        if j.name in have or j.name in pushed:
            c[0] += 1
        else:
            pending.append(j.name)
    lines = [f"{'run':10s} {'attack':7s} done"]
    lines += [f"{r:10s} {a:7s} {d}/{n}" + ("" if d == n else "  <- pending") for (r, a), (d, n) in cells.items()]
    names = {j.name for j in jobs}
    return lines, {"done": len(jobs) - len(pending), "total": len(jobs), "pending": pending,
                   "unpushed": sorted(unpushed & have & names)}


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

def build_stages(config: str, dcfg: dict, with_battery: bool, test_judge: bool = False,
                 battery_only: bool = False, part: str = "all", gpus: str | None = None) -> list[Stage]:
    """`part` picks the battery stages (PARTS); `gpus` (e.g. "0" on Colab) overrides the configs' GPUs for the
    battery and the estimate."""
    from aftermarket.defense import budget, store
    from aftermarket.defense.harden import candidate_grid
    from aftermarket.defense.select import SELECTION, frozen_ids
    from aftermarket.generate import load_attacks, make_jobs
    from aftermarket.load import AM_FROZEN_YAML

    b = {**dcfg["budget"], **({"gpus": len(gpus.split(","))} if gpus else {})}
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

    def pending_candidates():  # missing, or trained on an older L* (locate re-run since)
        from aftermarket.defense.harden import stale_candidate

        loc = json.loads(loc_path.read_text(encoding="utf-8")) if loc_ok() else None
        L = {c["id"]: loc["L_star"][str(c["k"])] for c in candidate_grid(dcfg)} if loc else {}
        return [i for i in cand_ids if not store.read_meta(i) or (i in L and stale_candidate(store.read_meta(i), L[i]))]

    def missing(path, sub_dir="gen"):
        jobs = make_jobs(gen_cfg(path), attacks)
        gone = [j.name for j in jobs if not (REPO_ROOT / "results" / sub_dir / f"{j.name}.parquet").exists()]
        return f"{len(gone)} of {len(jobs)} files missing in results/{sub_dir}: {gone[:4]}" if gone else None

    def check_models(ids_fn, what):
        return lambda: (f"{what} not trained: {ids_fn()} (see the worker tracebacks above)" if ids_fn() else None)

    from aftermarket.defense.locate import current

    loc_path = store.defense_dir() / "locate" / "locate.json"
    loc_ok = lambda: loc_path.exists() and current(json.loads(loc_path.read_text(encoding="utf-8")))
    loc_done = loc_ok()
    cand_ids = [c["id"] for c in candidate_grid(dcfg)]
    ctrl_ids = frozen_ids(dcfg)[1:]

    def judge_estimate(*paths):
        def est():
            judged = REPO_ROOT / "results" / "judged"  # parquet here, or its manifest pulled from the Hub
            n = sum(1 for p in paths for j in make_jobs(gen_cfg(p), attacks)
                    if not any((judged / f"{j.name}{s}").exists() for s in (".parquet", ".json")))
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
              check=lambda: None if loc_ok() else "no current locate.json",
              estimate=lambda: {"gpu_s": 0.0 if loc_done else b["locate_s"], "wall_s": 0.0 if loc_done else b["locate_s"],
                                "n_jobs": 0 if loc_done else 1, "source": "a-priori"}),
        Stage("harden candidates", lambda: (store.pull_all(dcfg), sh_parallel(shards("candidates"))),
              check=check_models(pending_candidates, "candidates"),
              estimate=lambda: budget.estimate_training(len(pending_candidates()), train_s, b)),
        Stage("dev A0", lambda: sh("aftermarket.generate", "--config", DEV_A0), check=lambda: missing(DEV_A0),
              estimate=est_gen(DEV_A0)),
        Stage("dev A3 + A4q3", lambda: sh("aftermarket.generate", "--config", DEV_ATT), check=lambda: missing(DEV_ATT),
              estimate=est_gen(DEV_ATT)),
        Stage("judge (WildGuard)", lambda: sh("aftermarket.judge", "--config", "configs/judge_defense.yaml"),
              check=lambda: missing(DEV_A0, "judged") or missing(DEV_ATT, "judged"),
              estimate=judge_estimate(DEV_A0, DEV_ATT)),
        Stage("select", lambda: sh("aftermarket.defense.select", "--config", config, "--stage", "1"),
              estimate=lambda: {"gpu_s": 0.0, "wall_s": 120.0, "n_jobs": 1, "source": "a-priori"}),
        Stage("controls", lambda: (store.pull_all(dcfg), sh_parallel(shards("controls"))),
              enabled=lambda: bool(selection().get("recipe")), check=check_models(lambda: pending_models(ctrl_ids), "controls"),
              estimate=lambda: budget.estimate_training(len(pending_models(ctrl_ids)), train_s, b)),
        Stage("freeze", run_freeze, enabled=lambda: bool(selection().get("recipe")) and not AM_FROZEN_YAML.exists(),
              estimate=lambda: {"gpu_s": 0.0, "wall_s": 0.0 if AM_FROZEN_YAML.exists() else 600.0,
                                "n_jobs": 0 if AM_FROZEN_YAML.exists() else 1, "source": "a-priori"}),
    ]
    if battery_only:
        stages, with_battery = [], True
    if with_battery:
        gpu_arg = ["--gpus", gpus] if gpus else []
        for path, a in PARTS[part]:
            who = "AM" if path == BATTERY else "controls"
            stages.append(Stage(f"test {who} {a}",
                                (lambda path=path, a=a: sh("aftermarket.generate", "--config", path, "--final", "--attacks", a,
                                                            *gpu_arg)),
                                enabled=AM_FROZEN_YAML.exists, estimate=est_gen(path, a.split(","))))
    if with_battery and test_judge:
        stages.append(Stage("AM test judge", lambda: sh("aftermarket.judge", "--config", "configs/judge.yaml"),
                            enabled=AM_FROZEN_YAML.exists, estimate=judge_estimate(BATTERY, CONTROLS)))
    return stages


def require_frozen(dcfg: dict, part: str = "all") -> None:
    """Battery gate: the Hub copy of am_frozen.yaml must pin every battery run (frozen_problems); it is
    installed into configs/ if missing here. Exits with code 2 and the reasons otherwise."""
    from huggingface_hub import HfApi, hf_hub_download

    from aftermarket.defense import store
    from aftermarket.generate import load_attacks
    from aftermarket.load import AM_FROZEN_YAML

    hub, files, text = None, None, None
    try:
        repo = store._repo(dcfg)
        hub_path = hf_hub_download(repo, store.FROZEN_FILE, repo_type="model", force_download=True)
        text = Path(hub_path).read_text(encoding="utf-8")
        hub = yaml.safe_load(text)
        rev = (hub.get("store") or {}).get("revision")
        files = set(HfApi().list_repo_files(repo, repo_type="model", revision=rev)) if rev else None
    except Exception as e:
        print(f"Hub check: {type(e).__name__}: {e}", flush=True)
    local = yaml.safe_load(AM_FROZEN_YAML.read_text(encoding="utf-8")) if AM_FROZEN_YAML.exists() else None
    need = sorted({j.run_id for j in battery_jobs(load_attacks(), part) if j.run_id.startswith("AM")})
    problems = frozen_problems(hub, local, need, files)
    if problems:
        print("REFUSED: the AM battery needs a frozen AM pinned on the Hub.\n  - " + "\n  - ".join(problems), flush=True)
        raise SystemExit(2)
    if local is None:
        AM_FROZEN_YAML.write_text(text, encoding="utf-8")
        print(f"installed configs/{AM_FROZEN_YAML.name} from the Hub", flush=True)
    s = hub["store"]
    print(f"frozen AM OK: {', '.join(need)} pinned at {s['repo']}@{s['revision'][:10]} "
          f"(frozen {hub.get('frozen_at_utc')}, recipe {hub.get('recipe')})", flush=True)


def print_battery_summary(started: float, max_hours: float | None, frozen: bool = True, part: str = "all") -> dict:
    from aftermarket.generate import load_attacks

    lines, s = battery_summary(battery_jobs(load_attacks(), part), REPO_ROOT / "results" / "gen")
    print(f"\n== AM battery summary, part {part} (test, --final) ==")
    print("\n".join(lines))
    print(f"jobs done {s['done']}/{s['total']} | session {(time.time() - started) / 3600:.2f} h"
          + (f" of --max-hours {max_hours}" if max_hours else ""))
    if s["unpushed"]:
        print(f"WARNING {len(s['unpushed'])} results are local only (push failed), lost when the session ends: "
              f"{s['unpushed']}. Push them: python -m aftermarket.generate --config <its config> --push-pending")
    if not frozen:
        print("REFUSED: no frozen AM pinned on the Hub, so nothing ran on test. Freeze AM in notebook 05 first.")
    elif s["pending"]:
        print(f"{len(s['pending'])} jobs left: commit this notebook again to resume")
    else:
        print("battery complete: judge it with notebooks/06_judge.ipynb on Colab")
    return s


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="AM defense pipeline with a quota guard")
    ap.add_argument("--config", required=True)
    ap.add_argument("--max-hours", type=float, help="stop cleanly after the stage that crosses this session time")
    ap.add_argument("--session-start", type=float, help="epoch seconds the Kaggle session started (default: now)")
    ap.add_argument("--with-battery", action="store_true", help="also run the frozen AM family on test (--final)")
    ap.add_argument("--battery-only", action="store_true",
                    help="only the AM test battery (notebook 07); refuses unless AM is frozen and pinned on the Hub")
    ap.add_argument("--with-test-judge", action="store_true",
                    help="also judge the battery here (default: notebook 06 judges it on Colab, off the Kaggle quota)")
    ap.add_argument("--plan-only", action="store_true", help="print the estimate and exit")
    ap.add_argument("--summary-only", action="store_true", help="print the AM battery status and exit")
    ap.add_argument("--part", choices=sorted(PARTS), default="all",
                    help="battery part: kaggle (notebook 07), colab (notebook 08) or all")
    ap.add_argument("--gpus", help="comma-separated GPUs for the battery (default: the configs'; Colab: 0)")
    args = ap.parse_args(argv)
    started = args.session_start or time.time()
    dcfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))

    from aftermarket.defense import budget, store
    from aftermarket.generate import pull_from_hub

    gen_cfg = yaml.safe_load((REPO_ROOT / DEV_A0).read_text(encoding="utf-8"))
    if args.summary_only:  # manifests from the Hub count jobs finished in earlier sessions
        from aftermarket.load import AM_FROZEN_YAML

        pull_from_hub(gen_cfg, REPO_ROOT / gen_cfg["out_dir"], want=lambda stem: True, suffixes=(".json",))
        print_battery_summary(started, args.max_hours, frozen=AM_FROZEN_YAML.exists(), part=args.part)
        return
    if args.battery_only and not args.plan_only:  # the estimate itself is harmless; running needs the frozen AM
        require_frozen(dcfg, args.part)
    # timings for the estimate: job manifests (json only) and defense metadata from the private Hub
    pull_from_hub(gen_cfg, REPO_ROOT / gen_cfg["out_dir"], want=lambda stem: True, suffixes=(".json",))
    jcfg = yaml.safe_load((REPO_ROOT / "configs" / "judge_defense.yaml").read_text(encoding="utf-8"))
    pull_from_hub(jcfg, REPO_ROOT / jcfg["out_dir"], jcfg["push"]["path_in_repo"], suffixes=(".json",))
    if not args.battery_only:
        store.pull_all(dcfg, want=lambda top: top in ("locate", "selection.json", store.FROZEN_FILE))
        store.pull_all(dcfg, suffixes=(store.META,))
    stages = build_stages(args.config, dcfg, args.with_battery, args.with_test_judge, args.battery_only,
                          args.part, args.gpus)
    print("Estimated cost of the work left (Colab, one T4: no Kaggle quota):" if args.battery_only and args.part == "colab"
          else "Estimated cost of the work left (Kaggle quota = session wall time):")
    print(budget.fmt_table([(s.name, s.estimate()) for s in stages]))
    if args.max_hours:
        print(f"--max-hours {args.max_hours}: stops after the stage that crosses it "
              f"(session so far {(time.time() - started) / 3600:.2f} h)")
    if args.plan_only:
        return
    run_stages(stages, args.max_hours, started)


if __name__ == "__main__":
    main()
