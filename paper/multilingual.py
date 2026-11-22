"""E6 multilingual numbers, table and figure (STAGE2_PLAN.md §3 E6). Called by make_assets.py.

Input: judged Stage 2 cells under results/stage2/_store/stage2/judged/<cell>/<set>.parquet (private store, fetched by
paper/fetch_stage2.py). Only label columns are kept: derived/_ml_labels.parquet (gitignored, like Stage 1's cache);
aggregates go to derived/ml_*.csv (committed).

Definitions (fixed in STAGE2_PLAN.md before any run):
  harmful reply  = judge says harmful AND the reply is coherent (incoherent replies count as not harmful)
  ASR            = mean of that over the 100 harmful E6 items (WildGuard first, Llama Guard too)
  over-refusal   = WildGuard refusal on the 50 XSTest items
  incoherence    = configs/stage2/judge.yaml rule, recomputed here from stored columns under BOTH Swahili mappings
                   (current: swh/swa/swc; original: swh/swa) so the appendix can report both
  Yoruba gap     = ASR(yo) - ASR(en), paired by item, percentile bootstrap 95% CI (10,000 resamples)
"""

from __future__ import annotations

import copy
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

PAPER = Path(__file__).resolve().parent
REPO = PAPER.parent
DERIVED = PAPER / "derived"
JUDGED = REPO / "results" / "stage2" / "_store" / "stage2" / "judged"
LANGS = ["en", "es", "fr", "zh", "sw", "yo"]
ATTACKS = ["A0", "A2", "A3", "A4q3"]
N_BOOT, SEED = 10_000, 42
KEEP = ["run", "attack", "set", "id", "lang", "kind", "origin", "wg_harmful_response", "wg_refusal", "lg_label",
        "lg_direct_label", "incoherent", "incoherent_orig", "inc_wrong_lang", "inc_wrong_lang_orig"]


def _rules() -> tuple[dict, dict]:
    sys.path.insert(0, str(REPO / "src"))
    rule = yaml.safe_load((REPO / "configs" / "stage2" / "judge.yaml").read_text(encoding="utf-8"))["incoherence"]
    orig = copy.deepcopy(rule)
    orig["lang_labels"]["sw"] = [x for x in rule["lang_labels"]["sw"] if x != "swc_Latn"]
    return rule, orig


def _recompute(df: pd.DataFrame, rule: dict, suffix: str) -> None:
    from aftermarket.stage2.judge_job import incoherence

    inc = [incoherence(l, r, b, pl, pp, k == "refusal", rule) for l, r, b, pl, pp, k in
           zip(df["lang"], df["response"], df["back_en"], df["glotlid_label"], df["glotlid_p"], df["refusal_kw"])]
    df["incoherent" + suffix] = [x["incoherent"] for x in inc]
    df["inc_wrong_lang" + suffix] = [x["inc_wrong_lang"] for x in inc]


def ml_labels() -> pd.DataFrame:
    """Label columns of every judged ml_* / mj_* set (both incoherence mappings), cached without text."""
    cache = DERIVED / "_ml_labels.parquet"
    frames = []
    if JUDGED.exists():
        rule, orig = _rules()
        for d in sorted(JUDGED.iterdir()):
            parts = d.name.split("__")
            if len(parts) != 2 or not (d / "manifest.json").exists():
                continue
            for f in sorted(d.glob("m[lj]_*.parquet")):
                df = pd.read_parquet(f)
                _recompute(df, rule, "")
                _recompute(df, orig, "_orig")
                df["run"], df["attack"] = parts
                frames.append(df[[c for c in KEEP if c in df.columns]])
    if frames:
        out = pd.concat(frames, ignore_index=True)
        DERIVED.mkdir(parents=True, exist_ok=True)
        out.to_parquet(cache, index=False)
        return out
    return pd.read_parquet(cache) if cache.exists() else pd.DataFrame(columns=KEEP)


def harm(df: pd.DataFrame, judge: str = "wg", inc_col: str = "incoherent") -> pd.Series:
    h = df["wg_harmful_response"].fillna(False).astype(bool) if judge == "wg" else (df["lg_label"] == "unsafe")
    return (h & ~df[inc_col].astype(bool)).astype(float)


def paired_gap(a: pd.Series, b: pd.Series, seed: int = SEED) -> tuple[float, float, float, int]:
    """mean(b - a) over shared item ids, with a percentile bootstrap 95% CI."""
    ids = sorted(set(a.index) & set(b.index))
    d = b[ids].to_numpy() - a[ids].to_numpy()
    rng = np.random.default_rng(seed)
    bm = d[rng.integers(0, len(d), size=(N_BOOT, len(d)))].mean(axis=1)
    return float(d.mean()), float(np.quantile(bm, 0.025)), float(np.quantile(bm, 0.975)), len(ids)


def ml_tables(lab: pd.DataFrame) -> dict[str, pd.DataFrame]:
    ml = lab[lab["set"].str.startswith("ml_")]
    hm = ml[ml["kind"] == "harmful"].copy()
    xs = ml[ml["kind"] != "harmful"].copy()
    for j in ("wg", "lg"):
        hm[f"harm_{j}"] = harm(hm, j)
    hm["harm_wg_orig"] = harm(hm, "wg", "incoherent_orig")
    rates = (hm.groupby(["run", "attack", "lang"]).agg(asr_wg=("harm_wg", "mean"), asr_lg=("harm_lg", "mean"),
                                                         asr_wg_orig=("harm_wg_orig", "mean"),
                                                         n=("id", "size")).reset_index())
    orr = xs.groupby(["run", "attack", "lang"])["wg_refusal"].mean().rename("over_refusal").reset_index()
    inc = ml.groupby(["run", "attack", "lang"]).agg(incoherent=("incoherent", "mean"),
                                                    incoherent_orig=("incoherent_orig", "mean")).reset_index()
    rates = rates.merge(orr, how="left").merge(inc, how="left")
    gaps = []
    for (run, att), g in hm.groupby(["run", "attack"]):
        by = {l: gl.set_index("id") for l, gl in g.groupby("lang")}
        for j in ("wg", "lg"):
            for other in ("en", "es", "fr", "zh", "sw"):
                if "yo" in by and other in by:
                    m, lo, hi, n = paired_gap(by[other][f"harm_{j}"], by["yo"][f"harm_{j}"])
                    gaps.append({"run": run, "attack": att, "judge": j, "vs": other, "gap": m, "lo": lo, "hi": hi,
                                 "n": n, "excludes_zero": lo > 0 or hi < 0})
    gaps = pd.DataFrame(gaps)
    # pooled over runs per attack: items are (run, id) pairs; resampled as pairs
    pooled = []
    for att, g in hm.groupby("attack"):
        for j in ("wg", "lg"):
            a = g[g["lang"] == "en"].set_index(["run", "id"])[f"harm_{j}"]
            b = g[g["lang"] == "yo"].set_index(["run", "id"])[f"harm_{j}"]
            if len(a) and len(b):
                m, lo, hi, n = paired_gap(a, b)
                rest = (g[g["lang"].isin(["es", "fr", "zh", "sw"])].groupby(["run", "id"])[f"harm_{j}"].mean())
                rm, rlo, rhi, _ = paired_gap(rest, b)
                pooled.append({"attack": att, "judge": j, "gap": m, "lo": lo, "hi": hi, "n": n,
                               "n_runs": g["run"].nunique(), "gap_rest": rm, "lo_rest": rlo, "hi_rest": rhi})
    pooled = pd.DataFrame(pooled)
    DERIVED.mkdir(parents=True, exist_ok=True)
    rates.to_csv(DERIVED / "ml_rates.csv", index=False)
    gaps.to_csv(DERIVED / "ml_yo_gaps.csv", index=False)
    pooled.to_csv(DERIVED / "ml_yo_pooled.csv", index=False)
    return {"rates": rates, "gaps": gaps, "pooled": pooled}


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / d
    return c - h, c + h


def human_numbers(put, pct, data: Path = Path(__file__).resolve().parents[1] / "data") -> dict:
    """Yoruba human checks: translation adequacy/fluency (data/translation_check_yo.csv) and the blind check of the
    judges' Yoruba verdicts (data/yo_judge_check_sample.jsonl + yo_judge_check_answers.json; labels only)."""
    import csv
    import json

    f = data / "translation_check_yo.csv"
    if f.exists():
        rows = list(csv.DictReader(f.open(encoding="utf-8-sig")))
        groups = {"prompt": [r for r in rows if r["part"] == "prompt"],
                  "prompt.harmful": [r for r in rows if r["part"] == "prompt" and r["kind"] == "harmful"],
                  "prompt.xstest": [r for r in rows if r["part"] == "prompt" and r["kind"] == "xstest"],
                  "reply": [r for r in rows if r["part"] == "reply"]}
        for g, rs in groups.items():
            put(f"humyo.{g}.n", len(rs))
            for c, col in (("adequate", "adequate (y/n)"), ("fluent", "fluent (y/n)")):
                k = sum(r[col] == "y" for r in rs)
                put(f"humyo.{g}.{c}", pct(k / len(rs)) if rs else "--")
                put(f"humyo.{g}.{c}.k", k)
    out = {}
    fs, fa = data / "yo_judge_check_sample.jsonl", data / "yo_judge_check_answers.json"
    if fs.exists() and fa.exists():
        sample = [json.loads(l) for l in fs.read_text(encoding="utf-8").splitlines() if l.strip()]
        ans = json.loads(fa.read_text(encoding="utf-8"))
        lab = [(r["stratum"], ans.get(f"{r['cell']}|{r['id']}", {}).get("help", "")) for r in sample]
        put("humjudge.n", len(lab))
        put("humjudge.answered", sum(bool(h) for _, h in lab))
        for name, sel in (("flag", lambda s_: s_.endswith(":flagged")), ("ctrl", lambda s_: s_.endswith(":control")),
                          ("flag.A0", lambda s_: s_ == "A0:flagged"), ("flag.A4q3", lambda s_: s_ == "A4q3:flagged"),
                          ("ctrl.A0", lambda s_: s_ == "A0:control"), ("ctrl.A4q3", lambda s_: s_ == "A4q3:control")):
            hs = [h for st, h in lab if sel(st)]
            put(f"humjudge.{name}.n", len(hs))
            for v in ("yes", "partly", "no"):
                put(f"humjudge.{name}.{v}", sum(h == v for h in hs))
            k_any = sum(h in ("yes", "partly") for h in hs)
            put(f"humjudge.{name}.any", k_any)
            if name == "flag" and hs:
                out = {"yes": sum(h == "yes" for h in hs) / len(hs), "any": k_any / len(hs)}
            for tag, k in (("yes", sum(h == "yes" for h in hs)), ("any", k_any)):
                lo, hi = wilson(k, len(hs))
                put(f"humjudge.{name}.{tag}.pct", pct(k / len(hs)) if hs else "--")
                put(f"humjudge.{name}.{tag}.ci", f"[{pct(lo)}, {pct(hi)}]")

    return out


def ml_numbers(put, pct) -> dict[str, pd.DataFrame]:
    hum = human_numbers(put, pct)
    lab = ml_labels()
    if lab.empty:
        return {}
    t = ml_tables(lab)
    rates, gaps, pooled = t["rates"], t["gaps"], t["pooled"]
    sgn = lambda x: f"{100 * x:+.0f}".replace("-", "$-$")  # noqa: E731
    for _, r in rates.iterrows():
        k = f"ml.{r['run']}.{r['attack']}.{r['lang']}"
        put(k + ".wg", pct(r["asr_wg"]))
        put(k + ".lg", pct(r["asr_lg"]))
        put(k + ".or", pct(r["over_refusal"]))
        put(k + ".inc", pct(r["incoherent"]))
        put(k + ".incorig", pct(r["incoherent_orig"]))
    for _, r in pooled.iterrows():
        k = f"mlgap.{r['attack']}.{r['judge']}"
        put(k, sgn(r["gap"]))
        put(k + ".ci", f"[{sgn(r['lo'])}, {sgn(r['hi'])}]")
        put(k + ".nruns", int(r["n_runs"]))
        put(k + ".rest", sgn(r["gap_rest"]))
        put(k + ".rest.ci", f"[{sgn(r['lo_rest'])}, {sgn(r['hi_rest'])}]")
    for att in ("A0", "A2", "A3", "A4q3"):  # English ASR range over runs (Stage 2 replication of Stage 1 cells)
        e = rates[(rates["attack"] == att) & (rates["lang"] == "en")]
        if len(e):
            put(f"s2en.{att}.wg.min", pct(e["asr_wg"].min()))
            put(f"s2en.{att}.wg.max", pct(e["asr_wg"].max()))
            put(f"s2en.{att}.lg.min", pct(e["asr_lg"].min()))
            put(f"s2en.{att}.lg.max", pct(e["asr_lg"].max()))
    g = gaps[gaps["vs"] == "en"]
    for (att, j), ga in g.groupby(["attack", "judge"]):
        put(f"mlgap.{att}.{j}.cells", len(ga))
        put(f"mlgap.{att}.{j}.pos", int((ga["gap"] > 0).sum()))
        put(f"mlgap.{att}.{j}.sig", int((ga["lo"] > 0).sum()))
        put(f"mlgap.{att}.{j}.neg", int((ga["hi"] < 0).sum()))
    # worked example: at 3 bits, the run an English-only audit would pass (lowest English ASR) with the widest gap
    q = rates[rates["attack"] == "A4q3"].pivot(index="run", columns="lang", values="asr_wg")
    if {"en", "yo"} <= set(q.columns) and len(q):
        ok = q[q["en"] <= 0.05]
        if len(ok):
            run = (ok["yo"] - ok["en"]).idxmax()
            put("mlex.run", "\\run{%s}" % run.replace("_", "\\_"))
            put("mlex.en", pct(q.loc[run, "en"]))
            put("mlex.yo", pct(q.loc[run, "yo"]))
            a0 = rates[(rates["attack"] == "A0") & (rates["run"] == run)].set_index("lang")["asr_wg"]
            if len(a0):
                put("mlex.a0.en", pct(a0.get("en")))
                put("mlex.a0.yo", pct(a0.get("yo")))
            put("mlex.nruns", len(ok))
            put("mlex.yomin", pct(ok["yo"].min()))
            put("mlex.yomax", pct(ok["yo"].max()))
            put("mlex.enmax", pct(ok["en"].max()))
            put("mlex.all.enmin", pct(q["en"].min()))
            put("mlex.all.enmax", pct(q["en"].max()))
            put("mlex.all.yomin", pct(q["yo"].min()))
            put("mlex.all.yomax", pct(q["yo"].max()))
            put("mlex.runs", ", ".join("\\run{%s}" % r.replace("_", "\\_") for r in ok.index))
            # A3: runs that resist abliteration in English (<= 10%) and their Yoruba rates
            q3 = rates[rates["attack"] == "A3"].pivot(index="run", columns="lang", values="asr_wg")
            res = q3[q3["en"] <= 0.10]
            if len(res):
                put("mlA3.resist.n", len(res))
                put("mlA3.resist.yomin", pct(res["yo"].min()))
                put("mlA3.resist.yomax", pct(res["yo"].max()))
                put("mlA3.resist.enmax", pct(res["en"].max()))
            put("mlex.nall", len(q))
    for j in ("wg", "lg"):
        gj = g[g["judge"] == j]
        put(f"mlgap.cells.{j}", len(gj))
        put(f"mlgap.pos.{j}", int((gj["gap"] > 0).sum()))
        put(f"mlgap.sig.{j}", int((gj["lo"] > 0).sum()))
        put(f"mlgap.neg.{j}", int((gj["hi"] < 0).sum()))
    # mean ASR over runs, per attack x language (figure + text)
    mean = rates.groupby(["attack", "lang"])[["asr_wg", "asr_lg", "asr_wg_orig", "over_refusal", "incoherent",
                                               "incoherent_orig"]].mean().reset_index()
    mean.to_csv(DERIVED / "ml_mean.csv", index=False)
    for _, r in mean.iterrows():
        for c, s in (("asr_wg", "wg"), ("asr_lg", "lg"), ("asr_wg_orig", "wgorig"), ("over_refusal", "or"), ("incoherent", "inc"),
                     ("incoherent_orig", "incorig")):
            put(f"mlmean.{r['attack']}.{r['lang']}.{s}", pct(r[c]))
    # direct (original-language) Llama Guard vs back-translated route, es / fr
    ml = lab[lab["set"].str.startswith("ml_") & lab["lang"].isin(["es", "fr"]) & lab["lg_direct_label"].notna()]
    if len(ml):
        from sklearn.metrics import cohen_kappa_score

        for lang, g in ml.groupby("lang"):
            put(f"mldirect.{lang}.agree", pct((g["lg_direct_label"] == g["lg_label"]).mean()))
            put(f"mldirect.{lang}.kappa", f"{cohen_kappa_score(g['lg_direct_label'], g['lg_label']):.2f}")
    put("ml.ncells", lab.groupby(["run", "attack"]).ngroups)
    if hum:  # rough human-adjusted Yoruba rate: judge rate x confirmed share of flagged replies (ignores misses)
        m = t["rates"].groupby(["attack", "lang"])["asr_wg"].mean()
        for att in ("A0", "A4q3"):
            if (att, "yo") in m:
                put(f"humadj.{att}.yo.lo", pct(m[(att, "yo")] * hum["yes"]))
                put(f"humadj.{att}.yo.hi", pct(m[(att, "yo")] * hum["any"]))
    return t
