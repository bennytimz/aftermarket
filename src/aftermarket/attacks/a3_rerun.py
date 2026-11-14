"""A3 v2 re-run (notebook 09), split into parts that Colab and Kaggle can share through Hub claims.

    PYTHONPATH=src python -m aftermarket.attacks.a3_rerun --part team --gpus 0,1 --max-hours 2 --session-start <epoch>
    PYTHONPATH=src python -m aftermarket.attacks.a3_rerun --part all --summary-only

Parts (A3 only, same seed / batch size / settings as every other cell):
  dev   BASE, SDT on harmful_dev, benign_dev, mmlu_dev (the per-layer table; no --final)       6 jobs
  team  BASE, SDT, GC, IL, RL, RL_noguard, ST, SP on harmful/xstest/mmlu_test (--final)       24 jobs
  am    frozen AM + AM_rand0, AM_rand1, AM_all on the same test splits (--final; refuses
        unless AM is frozen and pinned on the Hub)                                             12 jobs
  all   dev, team, am in that order

Every job is claimed on the Hub before it starts (aftermarket.claims): finished jobs and jobs locked by another
session less than 2 h ago are skipped, so two notebooks running at once never do the same job. No choice is made
from test cells: each model's layer is picked on dev prompts inside the attack.
"""

from __future__ import annotations

import argparse
import time

import yaml

from aftermarket.data import REPO_ROOT

PARTS = {
    "dev": [("configs/gen_cheap_dev.yaml", False)],
    "team": [("configs/gen_cheap_test.yaml", True)],
    "am": [("configs/gen_am_test.yaml", True), ("configs/gen_am_controls_test.yaml", True)],
}
ORDER = ["dev", "team", "am"]


def parts_of(part: str) -> list[str]:
    return ORDER if part == "all" else [part]


def part_jobs(part: str, attacks: dict | None = None) -> list:
    from aftermarket.generate import make_jobs

    jobs = []
    for p in parts_of(part):
        for path, _ in PARTS[p]:
            cfg = yaml.safe_load((REPO_ROOT / path).read_text(encoding="utf-8"))
            jobs += make_jobs({**cfg, "attacks": ["A3"]}, attacks)
    return jobs


def generate_command(path: str, final: bool, gpus: str | None, max_hours: float | None,
                     started: float) -> list[str]:
    """`python -m` args of one A3 generate call with claims and the session clock."""
    cmd = ["aftermarket.generate", "--config", path, "--attacks", "A3", "--claim", "--session-start", repr(started)]
    cmd += ["--final"] if final else []
    cmd += ["--gpus", gpus] if gpus else []
    cmd += ["--max-hours", repr(max_hours)] if max_hours is not None else []
    return cmd


def current_results(gen_dir, names: set[str]) -> set[str]:
    """Names with a v2 manifest here (select_by=bypass); a v1 leftover in a reused session doesn't count."""
    import json

    from aftermarket.attacks.a3_archive import is_stale

    out = set()
    for n in names:
        p = gen_dir / f"{n}.json"
        try:
            if p.exists() and not is_stale(json.loads(p.read_text(encoding="utf-8"))):
                out.add(n)
        except (OSError, ValueError):
            pass
    return out


def summary(part: str, started: float, max_hours: float | None) -> dict:
    """Per (run, split): done (parquet or pushed manifest here / on the Hub), locked elsewhere, or pending."""
    from aftermarket import claims
    from aftermarket.generate import load_attacks, pull_from_hub

    jobs = part_jobs(part, load_attacks())
    cfg = yaml.safe_load((REPO_ROOT / PARTS["team"][0][0]).read_text(encoding="utf-8"))
    gen_dir = REPO_ROOT / cfg["out_dir"]
    names = {j.name for j in jobs}
    pull_from_hub(cfg, gen_dir, want=lambda stem: stem in names, suffixes=(".json",))
    done = current_results(gen_dir, names)
    locked = {}
    hub, sub = claims.hub_for(cfg)
    if hub is not None:
        try:
            locked = claims.held_by_others(hub, sub, claims.session_id())
        except Exception as e:
            print(f"(locks not listed: {type(e).__name__}: {e})")
    print(f"\n== A3 v2 status, part {part} ==")
    pending = []
    for j in jobs:
        if j.name in done:
            state = "done"
        elif j.name in locked:
            state = f"running elsewhere ({locked[j.name].get('owner')}, since {locked[j.name].get('time_utc')})"
        else:
            state = "PENDING"
            pending.append(j.name)
        print(f"  {j.name:36s} {state}")
    n_done = sum(j.name in done for j in jobs)
    print(f"done {n_done}/{len(jobs)}, running elsewhere {len(locked.keys() & names - done)}, pending {len(pending)}"
          f" | session {(time.time() - started) / 3600:.2f} h" + (f" of --max-hours {max_hours}" if max_hours else ""))
    if pending:
        print("re-run this notebook (Colab: Run all; Kaggle: Save & Run All) to continue; claims prevent double work")
    return {"done": n_done, "total": len(jobs), "pending": pending}


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--part", choices=ORDER + ["all"], default="all")
    ap.add_argument("--gpus", help="comma-separated GPU ids (Colab: 0, Kaggle: 0,1)")
    ap.add_argument("--max-hours", type=float, help="start no job once the session has run this long")
    ap.add_argument("--session-start", type=float, help="epoch seconds the session started (default: now)")
    ap.add_argument("--summary-only", action="store_true")
    args = ap.parse_args(argv)
    started = args.session_start or time.time()
    if args.summary_only:
        summary(args.part, started, args.max_hours)
        return

    from aftermarket import claims
    from aftermarket.defense.pipeline import require_frozen, sh

    claims.session_id()  # one owner id for every generate call of this run (inherited through the environment)
    print(f"session {claims.session_id()}")
    for p in parts_of(args.part):
        if args.max_hours is not None and (time.time() - started) / 3600 >= args.max_hours:
            print(f"--max-hours {args.max_hours} reached: part {p} not started")
            break
        print(f"\n== part {p}", flush=True)
        if p == "am":
            try:
                dcfg = yaml.safe_load((REPO_ROOT / "configs" / "defense.yaml").read_text(encoding="utf-8"))
                require_frozen(dcfg, "a3")
            except SystemExit:
                print("part am skipped: AM is not frozen and pinned on the Hub")
                continue
        for path, final in PARTS[p]:
            sh(*generate_command(path, final, args.gpus, args.max_hours, started))
    summary(args.part, started, args.max_hours)


if __name__ == "__main__":
    main()
