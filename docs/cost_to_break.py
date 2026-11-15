"""Cost-to-break numbers for the README (needs the judged parquet files in results/judged/, private).

    python docs/cost_to_break.py

A5bon: attempts to first success among broken prompts. The 8 variants of each prompt are generated in a fixed,
seeded order (identical for every model), so "attempts" = index of the first variant judged harmful + 1.
A1/A2: training GPU-seconds from each generation manifest (attack_info.train_seconds, one entry per adapter).
"""

from __future__ import annotations

import glob
import json
import statistics
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RUNS = ["BASE", "SDT", "GC", "IL", "RL", "RL_noguard", "ST", "SP", "AM"]
JUDGES = {"WildGuard": "wg_harmful_response", "Llama Guard": "unsafe"}


def attempts(run: str) -> dict[str, tuple[int, float, float]]:
    df = pd.read_parquet(ROOT / "results" / "judged" / f"{run}__A5bon__harmful_test.parquet",
                         columns=["item_id", "variant"] + list(JUDGES.values()))
    out = {}
    for name, col in JUDGES.items():
        first = df[df[col].astype(bool)].groupby("item_id").variant.min() + 1
        out[name] = (len(first), float(first.mean()), float(first.median()))
    return out


def train_minutes() -> dict[tuple[str, str], float]:
    seen = {}
    for f in glob.glob(str(ROOT / "results" / "gen" / "*__harmful_test.json")):
        d = json.loads(Path(f).read_text(encoding="utf-8"))
        t = d.get("attack_info", {}).get("train_seconds")
        if t:
            seen[(d["job"]["run_id"], d["job"]["attack_id"])] = t / 60
    return seen


def main() -> None:
    print("| model | WildGuard: broken / 200, mean, median attempts | Llama Guard: broken / 200, mean, median |")
    print("|---|---|---|")
    for run in RUNS:
        a = attempts(run)
        cells = [f"{n} / 200, {m:.1f}, {md:.0f}" for n, m, md in a.values()]
        print(f"| {run} | {cells[0]} | {cells[1]} |")
    tm = train_minutes()
    a2 = [v for (r, a), v in tm.items() if a == "A2"]
    print(f"A2 training minutes: min {min(a2):.1f}, median {statistics.median(a2):.1f}, max {max(a2):.1f}")


if __name__ == "__main__":
    main()
