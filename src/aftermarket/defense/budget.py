"""Quota estimate for the defense pipeline (notebook 05).

Kaggle charges session wall time, so the headline number is quota-hours (wall); GPU-hours (sum over both T4s)
are shown too. Every rate comes from a measured timing when one exists, otherwise from the a-priori
fallbacks in configs/defense.yaml `budget` (marked "a-priori" in the table):
  generation  seconds per prompt from the job manifests (results/gen/*.json, pulled from the private Hub),
              keyed by (attack, split kind); same split kind under A0 as the next fallback
  A1/A2       median attack_info.train_seconds of fine-tune manifests
  defense     median train_seconds of defense_meta.json (candidates / controls)
  locate      locate.json seconds
Only work still to do is counted, so the estimate shrinks as a resumed run progresses.
"""

from __future__ import annotations

import json
import math
import statistics
from pathlib import Path

KINDS = ("harmful", "benign", "xstest", "mmlu")


def split_kind(split: str) -> str:
    return next(k for k in KINDS if split.startswith(k))


def read_manifests(gen_dir: Path) -> list[dict]:
    out = []
    for p in sorted(Path(gen_dir).glob("*.json")):
        try:
            out.append(json.loads(p.read_text(encoding="utf-8")))
        except (OSError, ValueError):
            continue
    return out


def prompt_rates(manifests: list[dict]) -> dict[tuple[str, str], float]:
    """(attack, split kind) -> median seconds per prompt."""
    acc: dict[tuple[str, str], list[float]] = {}
    for m in manifests:
        j = m.get("job", {})
        if m.get("seconds") and m.get("n_prompts"):
            acc.setdefault((j["attack_id"], split_kind(j["split"])), []).append(m["seconds"] / m["n_prompts"])
    return {k: statistics.median(v) for k, v in acc.items()}


def ft_train_seconds(manifests: list[dict]) -> float | None:
    seen = {}
    for m in manifests:
        ai = m.get("attack_info") or {}
        if ai.get("kind") == "finetune" and ai.get("train_seconds"):
            seen[(ai.get("adapter_trained_on_run", m["job"]["run_id"]), m["job"]["attack_id"])] = ai["train_seconds"]
    return statistics.median(seen.values()) if seen else None


def job_prompts(job, attacks: dict) -> int:
    from aftermarket.generate import SPLIT_SIZES

    a = attacks.get(job.attack_id, {})
    return SPLIT_SIZES.get(job.split, 100) * (a.get("n", 1) if a.get("mode") == "bon" else 1)


def wall_from_groups(group_seconds: list[float], gpus: int) -> float:
    """Longest-first onto the least-loaded GPU (as generate.assign_jobs does)."""
    loads = [0.0] * max(1, gpus)
    for s in sorted(group_seconds, reverse=True):
        loads[loads.index(min(loads))] += s
    return max(loads)


def estimate_gen(jobs: list, done: set[str], attacks: dict, rates: dict, ft_train_s: float | None, b: dict) -> dict:
    """Pending generation jobs -> {gpu_s, wall_s, n_jobs, source}."""
    from aftermarket.generate import group_key

    groups: dict[tuple, float] = {}
    measured = True
    for j in jobs:
        if j.name in done:
            continue
        kind = split_kind(j.split)
        rate = rates.get((j.attack_id, kind)) or rates.get(("A0", kind))
        if rate is None:
            rate, measured = b["gen_s"][kind] / 100, False
        try:
            key = group_key(j, attacks)
        except KeyError:  # run not in the registry yet (AM before freezing): one group per run
            key = (j.run_id, j.attack_id)
        if key not in groups:
            groups[key] = b["load_s"]
            if attacks[j.attack_id]["kind"] == "finetune":
                groups[key] += ft_train_s or b["ft_train_s"]
                measured = measured and ft_train_s is not None
        groups[key] += rate * job_prompts(j, attacks)
    gpu = sum(groups.values())
    return {"gpu_s": gpu, "wall_s": wall_from_groups(list(groups.values()), b["gpus"]), "n_jobs": sum(
        1 for j in jobs if j.name not in done), "source": "measured" if measured else "a-priori"}


def estimate_training(n_pending: int, seconds: float | None, b: dict) -> dict:
    s = seconds or b["train_s"]
    return {"gpu_s": n_pending * s, "wall_s": math.ceil(n_pending / b["gpus"]) * s, "n_jobs": n_pending,
            "source": "measured" if seconds else "a-priori"}


def fmt_table(rows: list[tuple[str, dict]]) -> str:
    lines = [f"{'stage':28s} {'todo':>5s} {'quota-h':>8s} {'GPU-h':>7s}  source"]
    tw = tg = 0.0
    for name, e in rows:
        tw += e["wall_s"]
        tg += e["gpu_s"]
        lines.append(f"{name:28s} {e['n_jobs']:>5d} {e['wall_s'] / 3600:>8.2f} {e['gpu_s'] / 3600:>7.2f}  {e['source']}")
    lines.append(f"{'TOTAL':28s} {'':>5s} {tw / 3600:>8.2f} {tg / 3600:>7.2f}")
    return "\n".join(lines)
