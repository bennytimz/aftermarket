"""Every number, table and figure in the paper comes from result files through this script.

    PYTHONPATH=src python paper/make_assets.py         # -> paper/numbers.tex, paper/tables/*.tex, paper/figures/*.pdf

numbers.tex defines values looked up in the text with \\V{key} (an undefined key prints ??key and warns).
Inputs: results/tables/*.csv (Stage 1 aggregates, committed), results/judged/*.parquet (Stage 1 per-item labels,
private store; label columns only), configs/*.yaml, and results/stage2/** (Stage 2, added as it lands).
Derived per-item statistics (paired bootstrap deltas) are also written to paper/derived/*.csv, so each figure can be
re-plotted without the private data.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

PAPER = Path(__file__).resolve().parent
REPO = PAPER.parent
TABLES, FIGS, DERIVED = PAPER / "tables", PAPER / "figures", PAPER / "derived"
RUNS = ["BASE", "SDT", "GC", "IL", "RL", "RL_noguard", "ST", "SP", "AM", "AM_rand0", "AM_rand1", "AM_all"]
TEAM = ["SDT", "GC", "IL", "RL", "RL_noguard", "ST", "SP"]
CONTROLS = ["AM_rand0", "AM_rand1", "AM_all"]
ATTACKS = ["A1", "A2", "A3", "A4q3", "A5raw", "A5bon"]
N_BOOT, SEED = 10_000, 42

V: dict[str, str] = {}  # key -> LaTeX text


def put(key: str, value) -> None:
    V[key] = str(value)


def pct(x: float, nd: int = 0) -> str:
    return "--" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{100 * x:.{nd}f}"


def latex_name(run: str) -> str:
    return run.replace("_", r"\_")


# ----------------------------------------------------------------------------- Stage 1 aggregates

def stage1_cells() -> pd.DataFrame:
    d = pd.read_csv(REPO / "results" / "tables" / "cell_ci.csv")
    return d[d["stage"] == "test"]


def stage1_numbers() -> None:
    ci = stage1_cells()
    for _, r in ci.iterrows():
        m = {"asr_lg": "lg", "asr_wg": "wg", "over_refusal_wg": "or", "mmlu": "mmlu"}[r["metric"]]
        k = f"{m}.{r['run']}.{r['attack']}"
        put(k, pct(r["value"]))
        put(k + ".ci", f"[{pct(r['lo'])}, {pct(r['hi'])}]")
    for name, f in (("ret", "retention.csv"), ("retwg", "retention_wg.csv")):
        d = pd.read_csv(REPO / "results" / "tables" / f)
        d = d[d["stage"] == "test"]
        for _, r in d.iterrows():
            put(f"{name}.{r['run']}", f"{r['retention']:.2f}")
            put(f"{name}.{r['run']}.ci", f"[{r['lo']:.2f}, {r['hi']:.2f}]")
        ranked = d[d["ranked"]].sort_values("retention", ascending=False)
        for i, run in enumerate(ranked["run"], 1):
            put(f"{name}.rank.{run}", i)
        put(f"{name}.nranked", len(ranked))
    h = pd.read_csv(REPO / "results" / "tables" / "human_agreement.csv")
    scope = {"all": "all", "A5 raw + BoN": "afive", "other attacks": "other"}
    for _, r in h.iterrows():
        judge = "lg" if r["pair"].startswith("Llama Guard") else "wg" if r["pair"].startswith("WildGuard harmful") \
            else "wgref" if r["pair"].startswith("WildGuard refusal") else "kwref" if r["pair"].startswith("keyword refusal") else "kw"
        sc = scope.get(r["scope"], r["scope"])
        put(f"kappa.{judge}.{sc}", f"{r['kappa']:.2f}")
        put(f"hum.n.{judge}.{sc}", int(r["n"]))
        put(f"hum.rate.{judge}.{sc}", pct(r["judge_rate"]))
        put(f"hum.humanrate.{judge}.{sc}", pct(r["human_rate"]))
    # append-only file: the last line per id wins, so count unique ids
    put("hum.nlabels", len({json.loads(l)["id"] for l in open(REPO / "data" / "hand_labels.jsonl", encoding="utf-8")
                            if l.strip()}))
    # A2 range across the six public fixes and BASE (abstract / intro)
    a2 = ci[(ci["attack"] == "A2") & (ci["run"].isin(["BASE"] + TEAM)) & (ci["metric"] == "asr_lg")]["value"]
    a2wg = ci[(ci["attack"] == "A2") & (ci["run"].isin(["BASE"] + TEAM)) & (ci["metric"] == "asr_wg")]["value"]
    a0 = ci[(ci["attack"] == "A0") & (ci["run"].isin([r for r in TEAM if r != "RL_noguard"])) & (ci["metric"] == "asr_lg")]["value"]
    put("lg.A2.min", pct(a2.min()))
    put("lg.A2.max", pct(a2.max()))
    put("wg.A2.min", pct(a2wg.min()))
    put("wg.A2.max", pct(a2wg.max()))
    put("lg.A0.fixes.min", pct(a0.min()))
    put("lg.A0.fixes.max", pct(a0.max()))


def release_vs_retention() -> None:
    """Does safety at release predict retention? Spearman rho of A0 ASR vs retention over the ranked runs,
    with a permutation p-value (exact over all orderings is too many; 100k permutations, seed 42)."""
    from scipy.stats import spearmanr

    ci = stage1_cells()
    for judge, rfile in (("lg", "retention.csv"), ("wg", "retention_wg.csv")):
        ret = pd.read_csv(REPO / "results" / "tables" / rfile)
        ret = ret[(ret["stage"] == "test") & ret["ranked"]]
        a0 = ci[(ci["attack"] == "A0") & (ci["metric"] == f"asr_{judge}")].set_index("run")["value"]
        x = np.array([a0[r] for r in ret["run"]])
        y = ret["retention"].to_numpy()
        rho = spearmanr(x, y).statistic
        rng = np.random.default_rng(SEED)
        perm = np.array([spearmanr(x, rng.permutation(y)).statistic for _ in range(20_000)])
        p = (np.sum(np.abs(perm) >= abs(rho) - 1e-12) + 1) / (len(perm) + 1)
        put(f"rho.a0ret.{judge}", f"{rho:.2f}")
        put(f"rho.a0ret.{judge}.p", f"{p:.2f}")
        put(f"rho.a0ret.{judge}.n", len(x))


# ----------------------------------------------------------------------------- per-item (paired) statistics

def judged_items(metric: str) -> pd.DataFrame:
    from aftermarket.evaluate import item_labels, load_judged

    cache = DERIVED / "_judged_labels.parquet"
    if cache.exists():
        df = pd.read_parquet(cache)
    else:
        df = load_judged(REPO / "results" / "judged")
        DERIVED.mkdir(parents=True, exist_ok=True)
        df.to_parquet(cache, index=False)  # labels only (no text); gitignored
    lab = item_labels(df, metric)
    return lab[lab["stage"] == "test"]


def paired_delta_ci(a: np.ndarray, b: np.ndarray, seed: int = SEED) -> tuple[float, float, float]:
    """Mean of (b - a) over paired items with a percentile bootstrap 95% CI (10,000 resamples)."""
    d = b - a
    rng = np.random.default_rng(seed)
    means = d[rng.integers(0, len(d), size=(N_BOOT, len(d)))].mean(axis=1)
    return float(d.mean()), float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def rise_table(metric: str = "asr_wg") -> pd.DataFrame:
    """Rise in ASR after each attack vs the same run's A0, paired by prompt, with bootstrap CI."""
    lab = judged_items(metric)
    piv = {(r, a): g.set_index("item_id")["y"] for (r, a), g in lab.groupby(["run_id", "attack_id"])}
    recs = []
    for run in RUNS:
        if (run, "A0") not in piv:
            continue
        base = piv[(run, "A0")]
        for att in ATTACKS + ["A3max"]:
            if (run, att) not in piv:
                continue
            s = piv[(run, att)]
            common = sorted(set(base.index) & set(s.index))
            m, lo, hi = paired_delta_ci(base[common].to_numpy(), s[common].to_numpy())
            recs.append({"metric": metric, "run": run, "attack": att, "rise": m, "lo": lo, "hi": hi,
                         "n": len(common), "excludes_zero": lo > 0 or hi < 0})
    out = pd.DataFrame(recs)
    DERIVED.mkdir(parents=True, exist_ok=True)
    out.to_csv(DERIVED / f"rise_{metric}.csv", index=False)
    return out


# ----------------------------------------------------------------------------- config values

def config_numbers() -> None:
    att = yaml.safe_load((REPO / "configs" / "attacks.yaml").read_text(encoding="utf-8"))["attacks"]
    tr = att["A1"]["train"]
    for k in ("lora_r", "lora_alpha", "epochs", "batch_size", "grad_accum", "max_len"):
        put(f"cfg.{k}", tr[k])
    put("cfg.lr", f"{tr['lr']:.0e}".replace("e-0", r"\times10^{-") + "}")
    put("cfg.effbatch", tr["batch_size"] * tr["grad_accum"])
    put("cfg.bon.n", att["A5bon"]["n"])
    put("cfg.bon.sigma", att["A5bon"]["sigma"])
    put("cfg.hqq.group", att["A4q3"]["group_size"])
    put("cfg.a3.klmax", att["A3"]["select"]["kl_max"])
    put("cfg.a3.nharm", att["A3"]["select"]["n_harmful"])
    data = yaml.safe_load((REPO / "configs" / "data.yaml").read_text(encoding="utf-8"))
    ft = data["finetune"]
    put("cfg.a2.harmful", ft["a2"]["harmful"])
    put("cfg.a2.benign", ft["a2"]["benign"])
    put("cfg.a1.n", ft["a1_alpaca"])
    s = data["sizes"]
    put("cfg.n.harmful", sum(s["harmful_test"].values()))
    put("cfg.n.xstest", s["xstest_test"])
    put("cfg.n.mmlu", s["mmlu_test"])
    put("cfg.n.calib", sum(s["calib_pairs"].values()))
    am = yaml.safe_load((REPO / "configs" / "am_frozen.yaml").read_text(encoding="utf-8"))
    put("cfg.am.k", am["recipe"]["k"])
    put("cfg.am.layers", ", ".join(str(x) for x in am["L_star"]))
    put("cfg.am.shieldbits", am["models"]["AM"]["quant_shield"]["bits"])
    dfn = yaml.safe_load((REPO / "configs" / "defense.yaml").read_text(encoding="utf-8"))
    put("cfg.am.lr", f"{dfn['harden']['lr']:.0e}".replace("e-0", r"\times10^{-") + "}")
    put("cfg.am.epochs", dfn["harden"]["epochs"])
    sub = yaml.safe_load((REPO / "configs" / "stage2" / "subsets.yaml").read_text(encoding="utf-8"))
    for t in ("arc_challenge", "hellaswag", "winogrande", "gsm8k"):
        put(f"cfg.cap.{t}", sub["capability"][t]["n"])
    put("cfg.ml.harmful", len(sub["multilingual"]["harmful_ids"]))
    put("cfg.ml.xstest", len(sub["multilingual"]["xstest_ids"]))
    put("cfg.sr.n", len(sub["strongreject"]["rows"]))
    put("cfg.mj.n", len(sub["multijail"]["rows"]))


# ----------------------------------------------------------------------------- Stage 2 (added as results land)

def overlap_numbers() -> None:
    p = REPO / "results" / "stage2" / "overlap" / "summary.json"
    if not p.exists():
        return
    s = json.loads(p.read_text(encoding="utf-8"))
    for name, v in s["sets"].items():
        k = name.replace("_", "")
        put(f"ov.{k}.flagged", v["n_flagged_pairs"])
        put(f"ov.{k}.exact", v["n_exact"])
        put(f"ov.{k}.maxjac", f"{v['max_char5_jaccard']:.2f}")
        put(f"ov.{k}.maxcos", f"{v['max_cosine']:.2f}")


# ----------------------------------------------------------------------------- E1 capability (Stage 2 cells)

CAP_TASKS = ["mmlu300", "arc_challenge", "hellaswag", "winogrande", "ifeval", "gsm8k"]
CAP_SHORT = {"mmlu300": "mmlu", "arc_challenge": "arc", "hellaswag": "hs", "winogrande": "wino", "ifeval": "ifeval",
             "gsm8k": "gsm"}
CELLS_DIR = REPO / "results" / "stage2" / "_store" / "stage2" / "cells"  # filled by paper/fetch_stage2.py


def capability_items() -> pd.DataFrame:
    """Per-item primary scores of every finished Stage 2 cell (benign benchmarks only) -> derived/capability_items.csv."""
    from aftermarket.stage2.capability import item_scores

    recs = []
    if CELLS_DIR.exists():
        for d in sorted(CELLS_DIR.iterdir()):
            man = d / "manifest.json"
            if len(d.name.split("__")) != 2 or not man.exists() or json.loads(man.read_text(encoding="utf-8")).get("error"):
                continue
            run, att = d.name.split("__")
            df = item_scores(d)
            df.insert(0, "attack", att)
            df.insert(0, "run", run)
            recs.append(df)
    out = pd.concat(recs, ignore_index=True) if recs else pd.DataFrame(columns=["run", "attack", "task", "doc_id", "y"])
    if len(out):
        out.to_csv(DERIVED / "capability_items.csv", index=False)
    elif (DERIVED / "capability_items.csv").exists():  # rebuild from the committed per-item file
        out = pd.read_csv(DERIVED / "capability_items.csv")
    return out


def _cell_matrix(items: pd.DataFrame, run: str, att: str) -> dict[str, pd.Series] | None:
    g = items[(items["run"] == run) & (items["attack"] == att)]
    if g.empty or set(g["task"]) != set(CAP_TASKS):
        return None
    return {t: gt.set_index("doc_id")["y"].sort_index() for t, gt in g.groupby("task")}


def cap_delta(a: dict[str, pd.Series], b: dict[str, pd.Series], seed: int = SEED) -> dict[str, tuple]:
    """Paired change b - a per task and for the composite (unweighted mean of the 6 task accuracies), each with a
    percentile bootstrap 95% CI: items resampled within each task, the same draw for a and b (10,000 resamples)."""
    rng = np.random.default_rng(seed)
    out, boots = {}, []
    for t in CAP_TASKS:
        ids = sorted(set(a[t].index) & set(b[t].index))
        d = b[t][ids].to_numpy() - a[t][ids].to_numpy()
        bm = d[rng.integers(0, len(d), size=(N_BOOT, len(d)))].mean(axis=1)
        out[t] = (float(d.mean()), float(np.quantile(bm, 0.025)), float(np.quantile(bm, 0.975)))
        boots.append(bm)
    comp = np.mean(boots, axis=0)
    out["comp"] = (float(np.mean([out[t][0] for t in CAP_TASKS])), float(np.quantile(comp, 0.025)),
                   float(np.quantile(comp, 0.975)))
    return out


def capability_numbers() -> pd.DataFrame:
    """Levels per cell, change vs the run's own A0 (attacks) and vs BASE at A0 (fixes). -> derived/capability_delta.csv"""
    items = capability_items()
    recs = []
    if items.empty:
        return pd.DataFrame()
    for (run, att), _ in items.groupby(["run", "attack"]):
        m = _cell_matrix(items, run, att)
        if m is None:
            continue
        lv = {t: float(m[t].mean()) for t in CAP_TASKS}
        lv["comp"] = float(np.mean(list(lv.values())))
        for t, v in lv.items():
            put(f"cap.{CAP_SHORT.get(t, t)}.{run}.{att}", pct(v, 1))
        ref = ("BASE", "A0") if att == "A0" else (run, "A0")
        if ref == (run, att):
            continue
        r = _cell_matrix(items, *ref)
        if r is None:
            continue
        for t, (d, lo, hi) in cap_delta(r, m).items():
            k = f"dcap.{CAP_SHORT.get(t, t)}.{run}.{att}"
            put(k, f"{100 * d:+.1f}".replace("-", "$-$"))
            put(k + ".ci", f"[{100 * lo:+.1f}, {100 * hi:+.1f}]".replace("-", "$-$"))
            recs.append({"run": run, "attack": att, "ref": "/".join(ref), "task": t, "delta": d, "lo": lo, "hi": hi,
                         "excludes_zero": lo > 0 or hi < 0})
    out = pd.DataFrame(recs)
    out.to_csv(DERIVED / "capability_delta.csv", index=False)
    a2 = out[(out["attack"] == "A2") & (out["task"] == "comp")]
    if len(a2):
        put("dcap.comp.A2.min", f"{100 * a2['delta'].min():+.1f}".replace("-", "$-$"))
        put("dcap.comp.A2.max", f"{100 * a2['delta'].max():+.1f}".replace("-", "$-$"))
        put("dcap.comp.A2.nsig", int(a2["excludes_zero"].sum()))
        put("dcap.comp.A2.n", len(a2))
    a0 = out[(out["attack"] == "A0") & (out["task"] == "comp")]
    if len(a0):
        put("dcap.comp.A0.nsig", int(a0["excludes_zero"].sum()))
        put("dcap.comp.A0.n", len(a0))
    mm = items[items["task"] == "mmlu300"].groupby(["run", "attack"])["y"].mean()
    put("cap.mmlu.min", pct(mm.min(), 1))
    put("cap.mmlu.min.cell", r"\run{%s} \att{%s}" % (latex_name(mm.idxmin()[0]), mm.idxmin()[1]))
    capability_table(items, out)
    sig = out[out["excludes_zero"] & (out["task"] != "comp")]
    for att in ("A0", "A2"):
        put(f"dcap.ntasksig.{att}", len(sig[sig["attack"] == att]))
    return out


def capability_table(items: pd.DataFrame, delta: pd.DataFrame) -> None:
    """Appendix table: per-benchmark accuracy at A0, then the composite change after each attack available."""
    lv = items.groupby(["run", "attack", "task"])["y"].mean().unstack("task")
    atts = [a for a in ("A2", "A3", "A4q3", "A1") if a in set(items["attack"])]
    names = {"mmlu300": "MMLU", "arc_challenge": "ARC-C", "hellaswag": "HellaS.", "winogrande": "Wino.",
             "ifeval": "IFEval", "gsm8k": "GSM8K"}
    head = " & ".join(["Run"] + [names[t] for t in CAP_TASKS] + ["Mean"] + [rf"$\Delta$\att{{{a}}}" for a in atts])
    lines = [r"\begin{tabular}{l" + "r" * (len(CAP_TASKS) + 1 + len(atts)) + "}", r"\toprule",
             head + r" \\", r"\midrule"]
    get = {(r["run"], r["attack"], r["task"]): r for _, r in delta.iterrows()}
    for run in RUNS:
        if (run, "A0") not in lv.index:
            continue
        row = lv.loc[(run, "A0")]
        cells = [f"{100 * row[t]:.1f}" for t in CAP_TASKS] + [f"{100 * row[CAP_TASKS].mean():.1f}"]
        for a in atts:
            r = get.get((run, a, "comp"))
            v = f"{100 * r['delta']:+.1f}" if r is not None else "--"
            cells.append(rf"\textbf{{{v}}}" if r is not None and r["excludes_zero"] else v)
        lines.append(rf"\run{{{latex_name(run)}}} & " + " & ".join(c.replace("-", "$-$") if c != "--" else c
                                                                   for c in cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (TABLES / "capability.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")


# ----------------------------------------------------------------------------- output

def write_numbers() -> None:
    lines = ["% Generated by paper/make_assets.py. Do not edit by hand."]
    for k in sorted(V):
        lines.append(f"\\expandafter\\def\\csname v@{k}\\endcsname{{{V[k]}}}")
    (PAPER / "numbers.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (DERIVED / "numbers.json").write_text(json.dumps(V, indent=1, sort_keys=True), encoding="utf-8")


def main() -> None:
    for d in (TABLES, FIGS, DERIVED):
        d.mkdir(parents=True, exist_ok=True)
    config_numbers()
    stage1_numbers()
    release_vs_retention()
    overlap_numbers()
    capability_numbers()
    from multilingual import ml_numbers

    ml_numbers(put, pct)
    from figures import make_all  # paper/figures.py

    make_all(V)
    write_numbers()
    print(f"{len(V)} values -> numbers.tex")


if __name__ == "__main__":
    main()
