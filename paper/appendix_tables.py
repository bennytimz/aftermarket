"""Appendix tables (A: models; C: full results) and the seed-check / ROI numbers. Called from make_assets.main.

Every table is generated from result files; nothing is typed by hand except the models table's one-line
descriptions and licences (read from each repo's Hub card at the pinned revision on 2026-11-22).
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import yaml

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
TABLES = HERE / "tables"
JUDGED = REPO / "results" / "stage2" / "_store" / "stage2" / "judged"
RUNS = ["BASE", "SDT", "GC", "IL", "RL", "RL_noguard", "ST", "SP", "AM", "AM_rand0", "AM_rand1", "AM_all"]
ATTS = ["A0", "A1", "A2", "A3", "A4q3", "A5raw", "A5bon", "A3max"]

# Licence = the `license` field of the repo's Hub card at the pinned revision (checked 2026-11-22);
# "none stated" = the card has no licence field.
MODELS = {
    "BASE": ("Tiny Aya Global", "Cohere Labs", "cc-by-nc-4.0", "the unfixed base model"),
    "SDT": ("Safety Doesn't Translate", "DhruvParmar051", "cc-by-nc-4.0",
            "weight-space repair: a drifted model with 1\\% of the base weights copied back"),
    "GC": ("Guardián colombiano", "vasmi-sas", "cc-by-nc-4.0",
           "LoRA (r=16) trained with Lisa; ships a short chat template"),
    "IL": ("Intentlock", "Sykeclonex", "none stated", "LoRA (r=16) trained with SafeGrad"),
    "RL": ("RedLens guard", "AmberNeel", "cc-by-nc-4.0",
           "a safety prompt in the chat template; weights unchanged"),
    "ST": ("ScarTissue", "amugoodbad229", "other", "sparse rank-one weight patches (2\\% support repair)"),
    "SP": ("SafetyPulse", "purvansh01", "cc-by-nc-4.0",
           "rank-8 safety-vector restore on a drifted, merged model"),
}


def tex(s: str) -> str:
    return s.replace("_", r"\_")


def run_tex(r: str) -> str:
    return r"\run{%s}" % tex(r)


def write(name: str, lines: list[str]) -> None:
    (TABLES / name).write_text("\n".join(lines) + "\n", encoding="utf-8")


# ----------------------------------------------------------------------------- Appendix A

def models_table() -> None:
    reg = yaml.safe_load((REPO / "configs" / "models.yaml").read_text(encoding="utf-8"))["models"]
    lines = [r"\begin{tabular}{@{}l p{0.34\linewidth} l l p{0.29\linewidth}@{}}", r"\toprule",
             r"Run & Fix (team, Hub account) & Revision & Licence & What it changes \\", r"\midrule"]
    for r, (name, team, lic, what) in MODELS.items():
        m = reg[r]
        lines.append(rf"{run_tex(r)} & {name} ({tex(team)})\newline {{\footnotesize\url{{{m['hf_path']}}}}} & "
                     rf"\texttt{{{m['revision'][:7]}}} & {lic} & {what} \\")
    lines.append(r"\run{RL\_noguard} & \run{RL}'s weights with the base chat template & -- & -- & control: no guard \\")
    lines.append(r"\run{AM} & ours\newline {\footnotesize\url{bennytimz/aftermarket-am}} & \texttt{d11af3a} & private repo & "
                 r"\S\ref{sec:defense}; three controls with random or all layers \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    write("models.tex", lines)


# ----------------------------------------------------------------------------- Appendix C

def cells() -> pd.DataFrame:
    d = pd.read_csv(REPO / "results" / "tables" / "cell_ci.csv")
    return d[d["stage"] == "test"]


def r0(x) -> str:
    return "--" if pd.isna(x) else f"{100 * x:.0f}"


def asr_table(ci: pd.DataFrame) -> None:
    atts = [a for a in ATTS if a in set(ci["attack"])] + [a for a in ("A4q4",) if a in set(ci["attack"])]
    lab = {a: a for a in atts}
    lines = [r"\begin{tabular}{@{}l " + "c" * len(atts) + "@{}}", r"\toprule",
             "Run & " + " & ".join(rf"\att{{{lab[a]}}}" for a in atts) + r" \\", r"\midrule"]
    for r in RUNS:
        cellv = []
        for a in atts:
            lg = ci[(ci.run == r) & (ci.attack == a) & (ci.metric == "asr_lg")]["value"]
            wg = ci[(ci.run == r) & (ci.attack == a) & (ci.metric == "asr_wg")]["value"]
            cellv.append("--" if lg.empty and wg.empty else
                         f"{r0(lg.iloc[0]) if len(lg) else '--'}/{r0(wg.iloc[0]) if len(wg) else '--'}")
        lines.append(run_tex(r) + " & " + " & ".join(cellv) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    write("asr_full.tex", lines)


def or_mmlu_table(ci: pd.DataFrame) -> None:
    atts = [a for a in ATTS if a in set(ci["attack"])]
    lines = [r"\begin{tabular}{@{}l " + "c" * len(atts) + "@{}}", r"\toprule",
             "Run & " + " & ".join(rf"\att{{{a}}}" for a in atts) + r" \\", r"\midrule"]
    for r in RUNS:
        cellv = []
        for a in atts:
            o = ci[(ci.run == r) & (ci.attack == a) & (ci.metric == "over_refusal_wg")]["value"]
            m = ci[(ci.run == r) & (ci.attack == a) & (ci.metric == "mmlu")]["value"]
            cellv.append("--" if o.empty and m.empty else
                         f"{r0(o.iloc[0]) if len(o) else '--'}/{r0(m.iloc[0]) if len(m) else '--'}")
        lines.append(run_tex(r) + " & " + " & ".join(cellv) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    write("or_mmlu_full.tex", lines)


def retention_table() -> None:
    lg = pd.read_csv(REPO / "results" / "tables" / "retention.csv")
    wg = pd.read_csv(REPO / "results" / "tables" / "retention_wg.csv")
    lines = [r"\begin{tabular}{@{}l c c l@{}}", r"\toprule",
             r"Run & Llama Guard & WildGuard & Families \\", r"\midrule"]
    for r in RUNS:
        a = lg[(lg.run == r) & (lg.stage == "test")]
        b = wg[(wg.run == r) & (wg.stage == "test")]
        if a.empty:
            continue
        f = lambda x: f"{x.retention.iloc[0]:.2f} [{x.lo.iloc[0]:.2f}, {x.hi.iloc[0]:.2f}]" if len(x) else "--"
        fam = a.families.iloc[0].replace(",", ", ") + ("" if bool(a.ranked.iloc[0]) else " (not ranked)")
        lines.append(f"{run_tex(r)} & {f(a)} & {f(b)} & {fam}" + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    write("retention_full.tex", lines)


def lang_table() -> None:
    f = HERE / "derived" / "ml_mean.csv"
    if not f.exists():
        return
    m = pd.read_csv(f)
    langs = ["en", "es", "fr", "zh", "sw", "yo"]
    lines = [r"\begin{tabular}{@{}l l " + "c" * len(langs) + "@{}}", r"\toprule",
             r"Attack & Measure & " + " & ".join(langs) + r" \\", r"\midrule"]
    meas = [("asr_wg", "ASR, WildGuard"), ("asr_lg", "ASR, Llama Guard"), ("over_refusal", "over-refusal"),
            ("incoherent", "incoherent"), ("incoherent_orig", "incoherent, no \\texttt{swc}")]
    for att in ("A0", "A2", "A3", "A4q3"):
        for i, (c, name) in enumerate(meas):
            x = m[m.attack == att].set_index("lang")
            vals = [r0(x.loc[l, c]) if l in x.index else "--" for l in langs]
            lines.append((rf"\att{{{att}}}" if i == 0 else "") + f" & {name} & " + " & ".join(vals) + r" \\")
        if att != "A4q3":
            lines.append(r"\midrule")
    lines += [r"\bottomrule", r"\end{tabular}"]
    write("lang_full.tex", lines)


def cost_full_table() -> dict:
    f = HERE / "derived" / "cost.csv"
    if not f.exists():
        return {}
    d = pd.read_csv(f)
    prices = yaml.safe_load((REPO / "configs" / "gpu_prices.yaml").read_text(encoding="utf-8"))["gpus"]
    key = {"T4": "T4", "L4": "L4", "A100": "A100_80GB"}
    cheap = {g: min(prices[k], key=lambda r: r["usd_per_hour"]) for g, k in key.items() if k in prices}
    gpus = [g for g in ("T4", "L4", "A100") if g in set(d.gpu)]
    names = {"AM_lora": "AM (LoRA)", "RL_noguard": "RL no guard", "hf_BASE": "BASE, HF fp16",
             "hf_BASE_hqq3": "BASE, HF HQQ 3-bit", "hf_AM_hqq3": "AM, HF HQQ 3-bit", "guard_BASE": "BASE + Llama Guard"}
    order = ["BASE", "SDT", "GC", "IL", "RL", "RL_noguard", "ST", "SP", "AM", "AM_lora", "guard_BASE",
             "hf_BASE", "hf_BASE_hqq3", "hf_AM_hqq3"]
    lines = [r"\begin{tabular}{@{}l " + "rrr" * len(gpus) + "@{}}", r"\toprule",
             "Config & " + " & ".join(rf"\multicolumn{{3}}{{c}}{{{g}}}" for g in gpus) + r" \\"
             + "".join(rf"\cmidrule(lr){{{2 + 3 * i}-{4 + 3 * i}}}" for i in range(len(gpus))),
             " & " + " & ".join(r"tok/s & HF \$ & min \$" for _ in gpus) + r" \\", r"\midrule"]
    for c in order:
        x = d[d.config == c]
        if x.empty:
            continue
        cellv = []
        for g in gpus:
            r = x[x.gpu == g]
            tps = r.pipeline_tps.iloc[0] if (len(r) and c == "guard_BASE" and pd.notna(r.pipeline_tps.iloc[0])) else \
                (r.tps.iloc[0] if len(r) else None)
            if tps is None or pd.isna(tps):
                cellv += ["--"] * 3
                continue
            hf = r.usd_m_hf.iloc[0] * (r.tps.iloc[0] / tps)
            mn = r.usd_m_min.iloc[0] * (r.tps.iloc[0] / tps)
            cellv += [f"{tps:,.0f}", f"{hf:.3f}", f"{mn:.3f}"]
        lines.append(tex(names.get(c, c)) + " & " + " & ".join(cellv) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    write("cost_full.tex", lines)  # was after the return, so the table was never rewritten
    return {g: (cheap[g]["provider"], cheap[g]["usd_per_hour"]) for g in gpus}


# ----------------------------------------------------------------------------- seed check (E5)

def seed_numbers(put) -> None:
    """A2 ASR (English harmful_test, 200 prompts) for training seeds 42 (Stage 2 retrain), 1 and 2."""
    rows = []
    for r in ("BASE", "GC", "AM"):
        for seed, cell in ((42, f"{r}__A2"), (1, f"{r}__A2s1"), (2, f"{r}__A2s2")):
            f = JUDGED / cell / "harmful_test.parquet"
            if not f.exists():
                continue
            d = pd.read_parquet(f)
            rows.append({"run": r, "seed": seed, "wg": float(d["wg_harmful_response"].astype(bool).mean()),
                         "lg": float((d["lg_label"] == "unsafe").mean()), "n": len(d)})
    df = pd.DataFrame(rows)
    if df.empty or df.groupby("run").size().min() < 3:
        return
    df.to_csv(HERE / "derived" / "seeds.csv", index=False)
    spread = df.groupby("run")[["wg", "lg"]].agg(lambda s: s.max() - s.min())
    ref = df[df.seed == 42].set_index("run")
    dev = {j: max(abs(r[j] - ref.loc[r["run"], j]) for _, r in df.iterrows()) for j in ("wg", "lg")}
    for j in ("wg", "lg"):  # largest change from seed 42, the seed every other cell uses
        put(f"seed.dev.{j}.max", f"{100 * dev[j]:.0f}")
    put("seed.dev.max", f"{100 * max(dev.values()):.0f}")
    for j in ("wg", "lg"):
        put(f"seed.spread.{j}.max", f"{100 * spread[j].max():.0f}")
        put(f"seed.{j}.min", r0(df[j].min()))
        put(f"seed.{j}.max", r0(df[j].max()))
    put("seed.n", int(df["n"].min()))
    s1 = cells()
    lines = [r"\begin{tabular}{@{}l c c c c@{}}", r"\toprule",
             r"Run & Stage 1 (seed 42) & seed 42 & seed 1 & seed 2 \\", r"\midrule"]
    for r in ("BASE", "GC", "AM"):
        s1v = []
        for j in ("lg", "wg"):
            v = s1[(s1.run == r) & (s1.attack == "A2") & (s1.metric == f"asr_{j}")]["value"]
            s1v.append(r0(v.iloc[0]) if len(v) else "--")
        x = df[df.run == r].set_index("seed")
        lines.append(f"{run_tex(r)} & {s1v[0]}/{s1v[1]} & " +
                     " & ".join(f"{r0(x.loc[s, 'lg'])}/{r0(x.loc[s, 'wg'])}" for s in (42, 1, 2)) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    write("seeds.tex", lines)


def appendix_tables(put) -> None:
    ci = cells()
    models_table()
    asr_table(ci)
    or_mmlu_table(ci)
    retention_table()
    lang_table()
    cheapest = cost_full_table()
    put("cost.cheapest", "; ".join(f"{g} {tex(p)} \\${u:.2f}/h" for g, (p, u) in cheapest.items()))
    seed_numbers(put)
    ov = REPO / "results" / "stage2" / "overlap" / "summary.json"
    if ov.exists():
        s = json.loads(ov.read_text(encoding="utf-8"))["sets"]["harmful_test"]
        put("overlap.n", s["n_flagged_pairs"])
        put("overlap.neval", s["n_eval"])
        put("overlap.maxcos", f"{s['max_cosine']:.2f}")
        put("overlap.maxjac", f"{s['max_char5_jaccard']:.2f}")
