"""Everything the video shows, read from the paper's own outputs (no number is typed in by hand).

Sources: paper/derived/numbers.json (every number in the text), paper/derived/rise_asr_wg.csv and
results/tables/cell_ci.csv (Fig. 2, built exactly as paper/figures.py:fig_heatmap does), results/tables/
human_agreement.csv (Fig. 5) and paper/derived/ml_rates.csv (Fig. 6).
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
NUMBERS = REPO / "paper" / "derived" / "numbers.json"

RUNS_TEAM = ["BASE", "SDT", "GC", "IL", "RL", "RL_noguard", "ST", "SP"]  # = paper/figures.py
RUNS_AM = ["AM", "AM_rand0", "AM_rand1", "AM_all"]
ATT_COLS = ["A1", "A2", "A3", "A4q3", "A5raw", "A5bon"]
ATT_SUB = {"A1": "benign FT", "A2": "harmful FT", "A3": "abliterate", "A4q3": "3-bit", "A5raw": "no template",
           "A5bon": "best-of-8", "A3max": "(post hoc)"}
ML_LANGS = ["en", "es", "fr", "zh", "sw", "yo"]
ML_LABEL = {"en": "English", "es": "Spanish", "fr": "French", "zh": "Chinese", "sw": "Swahili", "yo": "Yoruba"}

# what the brief puts on screen or in the voice-over, keyed by the paper's numbers.json key
EXPECTED = {"wg.A2.min": "87", "wg.A2.max": "90", "kappa.lg.all": "0.09", "kappa.wg.all": "0.42",
            "mlmean.A4q3.yo.wg": "54", "mlmean.A4q3.en.wg": "7", "humjudge.flag.yes": "4",
            "humjudge.flag.n": "20", "audit.full.hours": "1.1", "audit.full.usd": "0.68"}


def r0(x: float) -> int:
    """The paper's one rounding for printed percentages (paper/figures.py:r0)."""
    return int(f"{x:.0f}")


def check_numbers(V: dict) -> list[str]:
    """Differences between the brief and the paper; the video always uses the paper's value."""
    return [f"{k}: brief {v}, paper {V.get(k)}" for k, v in EXPECTED.items() if str(V.get(k)) != v]


def heatmap() -> dict:
    """Fig. 2: rise = r0(level) - r0(A0) (WildGuard, percentage points), bold when the paired CI excludes 0."""
    ci = pd.read_csv(REPO / "results" / "tables" / "cell_ci.csv")
    ci = ci[(ci["stage"] == "test") & (ci["metric"] == "asr_wg")]
    lev = {(r["run"], r["attack"]): 100 * r["value"] for _, r in ci.iterrows()}
    rise = pd.read_csv(REPO / "paper" / "derived" / "rise_asr_wg.csv")
    cols = ATT_COLS + ["A3max"]
    rows = []
    for run in RUNS_TEAM + RUNS_AM:
        cells = []
        for att in cols:
            r = rise[(rise["run"] == run) & (rise["attack"] == att)]
            if r.empty or (run, att) not in lev:
                cells.append(None)
                continue
            cells.append({"v": r0(lev[(run, att)]) - r0(lev[(run, "A0")]), "sig": bool(r["excludes_zero"].iloc[0])})
        rows.append({"run": run.replace("_", " "), "a0": r0(lev[(run, "A0")]), "cells": cells,
                     "block": "am" if run in RUNS_AM else "team"})
    return {"cols": [{"id": a, "sub": ATT_SUB[a]} for a in cols], "rows": rows}


def judges() -> dict:
    """Fig. 5: Cohen's kappa of each judge vs the blind hand labels, three scopes."""
    h = pd.read_csv(REPO / "results" / "tables" / "human_agreement.csv")
    pairs = [("Llama Guard", "Llama Guard unsafe vs human: harmful help"),
             ("WildGuard", "WildGuard harmful vs human: harmful help"),
             ("keyword", "keyword complied vs human: not a refusal")]
    scopes = [("all", "all replies"), ("other attacks", "A0–A4q3"), ("A5 raw + BoN", "A5 (no template, best-of-8)")]
    out = []
    for sc, lab in scopes:
        row = {"scope": lab, "n": int(h[(h["pair"] == pairs[0][1]) & (h["scope"] == sc)].iloc[0]["n"]), "k": {}}
        for name, pair in pairs:
            row["k"][name] = round(float(h[(h["pair"] == pair) & (h["scope"] == sc)].iloc[0]["kappa"]), 2)
        out.append(row)
    return {"scopes": out}


def languages() -> dict:
    """Fig. 6, 3-bit (A4q3) series: WildGuard harmful-reply rate per language, mean over the 12 runs + each run."""
    d = pd.read_csv(REPO / "paper" / "derived" / "ml_rates.csv")
    out = []
    for lang in ML_LANGS:
        g = d[(d["attack"] == "A4q3") & (d["lang"] == lang)]
        out.append({"lang": ML_LABEL[lang], "mean": r0(100 * g["asr_wg"].mean()),
                    "runs": [round(100 * x, 1) for x in g["asr_wg"]]})
    return {"attack": "A4q3", "bars": out}


def build() -> dict:
    V = json.loads(NUMBERS.read_text(encoding="utf-8"))
    keys = list(EXPECTED) + ["dcap.comp.A2.min", "dcap.comp.A2.max", "hum.nlabels"]
    data = {"V": {k: V[k] for k in keys}, "heatmap": heatmap(), "judges": judges(), "languages": languages(),
            "diffs": check_numbers(V)}
    # the bars must agree with the text's numbers (same rounding)
    langs = {b["lang"]: b["mean"] for b in data["languages"]["bars"]}
    assert str(langs["Yoruba"]) == V["mlmean.A4q3.yo.wg"] and str(langs["English"]) == V["mlmean.A4q3.en.wg"], langs
    ks = data["judges"]["scopes"][0]["k"]
    assert f"{ks['Llama Guard']:.2f}" == V["kappa.lg.all"] and f"{ks['WildGuard']:.2f}" == V["kappa.wg.all"], ks
    a2 = [c["cells"][1]["v"] + c["a0"] for c in data["heatmap"]["rows"] if c["block"] == "team"]
    assert (min(a2), max(a2)) == (int(V["wg.A2.min"]), int(V["wg.A2.max"])), a2
    return data


if __name__ == "__main__":
    d = build()
    print(json.dumps(d["V"], indent=1))
    print("differences from the brief:", d["diffs"] or "none")
