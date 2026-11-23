"""E2 cost / ROI numbers for Section 4.7 from the throughput JSONs (results/stage2/_store/stage2/throughput/<gpu>/)
and configs/gpu_prices.yaml (HF Jobs on-demand price per GPU type is the headline; the cheapest listed is in the
appendix). Writes tables/cost.tex and derived/cost.csv; V keys cost.*."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
TP = ROOT / "results" / "stage2" / "_store" / "stage2" / "throughput"
AUDIT = ROOT / "results" / "stage2" / "_store" / "stage2" / "audit"
PRICES = ROOT / "configs" / "gpu_prices.yaml"
HERE = Path(__file__).resolve().parent
GPU_KEY = {"T4": "T4", "L4": "L4", "A100": "A100_80GB"}
ORDER = ["BASE", "SDT", "GC", "IL", "RL", "RL_noguard", "ST", "SP", "AM", "AM_lora"]


def gpu_of(dirname: str) -> str | None:
    for k in ("A100", "L4", "T4"):
        if k in dirname:
            return k
    return None


def prices() -> dict[str, dict]:
    g = yaml.safe_load(PRICES.read_text(encoding="utf-8"))["gpus"]
    out = {}
    for k, key in GPU_KEY.items():
        rows = g.get(key, [])
        hf = [r["usd_per_hour"] for r in rows if r["provider"] == "Hugging Face Jobs"]
        out[k] = {"hf": hf[0] if hf else None, "min": min(r["usd_per_hour"] for r in rows) if rows else None}
    return out


def usd_per_m(usd_h: float, tps: float) -> float:
    return usd_h / (tps * 3600) * 1e6


def load() -> pd.DataFrame:
    rows = []
    if not TP.exists():
        return pd.DataFrame()
    for d in TP.iterdir():
        g = gpu_of(d.name)
        if not g:
            continue
        for f in d.glob("*.json"):
            j = json.loads(f.read_text(encoding="utf-8"))
            if "error" in j and j["error"]:
                continue
            fx = (j.get("generation") or j).get("fixed", {})
            rows.append({"gpu": g, "config": f.stem, "tps": fx.get("output_tps_mean"), "sd": fx.get("output_tps_sd"),
                         "prompt_tok": (j.get("generation") or j).get("mean_prompt_tokens"),
                         "pipeline_tps": j.get("pipeline_output_tps"), "backend": (j.get("generation") or j).get("backend")})
    return pd.DataFrame(rows)


def cost_numbers(put, pct) -> pd.DataFrame:
    df = load()
    if df.empty:
        return df
    P = prices()
    df["usd_m_hf"] = [usd_per_m(P[g]["hf"], t) if t and P[g]["hf"] else None for g, t in zip(df.gpu, df.tps)]
    df["usd_m_min"] = [usd_per_m(P[g]["min"], t) if t and P[g]["min"] else None for g, t in zip(df.gpu, df.tps)]
    df.to_csv(HERE / "derived" / "cost.csv", index=False)
    gpus = [g for g in ("T4", "L4", "A100") if g in set(df.gpu)]
    put("cost.gpus", ", ".join(gpus))
    for g in gpus:
        put(f"cost.price.{g}", f"{P[g]['hf']:.2f}")
        put(f"cost.pricemin.{g}", f"{P[g]['min']:.2f}")
        x = df[df.gpu == g].set_index("config")
        if "BASE" not in x.index:
            continue
        b = x.loc["BASE", "tps"]
        for c in x.index:
            k = f"cost.{g}.{c}"
            if x.loc[c, "tps"]:
                put(k + ".tps", f"{x.loc[c, 'tps']:,.0f}")
                put(k + ".rel", f"{100 * (x.loc[c, 'tps'] / b - 1):+.0f}".replace("-", "$-$"))
                put(k + ".loss", f"{abs(100 * (x.loc[c, 'tps'] / b - 1)):.0f}")
                put(k + ".gain", f"{abs(100 * (x.loc[c, 'tps'] / b - 1)):.0f}")  # unsigned, for "N% faster"
            if x.loc[c, "usd_m_hf"]:
                put(k + ".usd", f"{x.loc[c, 'usd_m_hf']:.3f}")
                put(k + ".usdmin", f"{x.loc[c, 'usd_m_min']:.3f}")
            if x.loc[c, "prompt_tok"]:
                put(k + ".ptok", f"{x.loc[c, 'prompt_tok']:.0f}")
        if "guard_BASE" in x.index and x.loc["guard_BASE", "pipeline_tps"]:
            gen = x.loc["guard_BASE", "tps"]
            put(f"cost.{g}.guard.overhead", f"{100 * (1 - x.loc['guard_BASE', 'pipeline_tps'] / gen):.0f}")
        if {"hf_BASE", "hf_BASE_hqq3"} <= set(x.index):
            put(f"cost.{g}.hqq.slow", f"{x.loc['hf_BASE', 'tps'] / x.loc['hf_BASE_hqq3', 'tps']:.1f}")
        if "AM_lora" in x.index and "AM" in x.index:
            put(f"cost.{g}.lora.overhead", f"{100 * (1 - x.loc['AM_lora', 'tps'] / x.loc['AM', 'tps']):.0f}")
        merged = x.loc[[c for c in ORDER[:-1] if c in x.index and c not in ("GC", "RL")], "tps"]
        put(f"cost.{g}.samemin", f"{abs(100 * (merged.min() / b - 1)):.0f}")
        put(f"cost.{g}.samemax", f"{abs(100 * (merged.max() / b - 1)):.0f}")
    x = df[df.gpu == "L4"].set_index("config")
    if {"BASE", "RL"} <= set(x.index):
        weights = [c for c in ("BASE", "SDT", "IL", "RL_noguard", "ST", "SP", "AM") if c in x.index]
        month = lambda c: x.loc[c, "usd_m_hf"] * 1000  # 1B output tokens = 1,000 x 1M
        put("roi.L4.weights.min", f"{min(month(c) for c in weights):.0f}")
        put("roi.L4.weights.max", f"{max(month(c) for c in weights):.0f}")
        put("roi.L4.BASE", f"{month('BASE'):.0f}")
        put("roi.L4.RL", f"{month('RL'):.0f}")
        put("roi.L4.RL.extra", f"{month('RL') - month('BASE'):.0f}")
        put("roi.L4.RL.extrayear", f"{12 * (month('RL') - month('BASE')):.0f}")
    cost_table(df, gpus)
    audit_numbers(put, P, df)
    return df


def audit_numbers(put, P, df: pd.DataFrame | None = None) -> None:
    """E7: timed full audit (GC) -> GPU-hours, $ and break-even in served tokens."""
    for mode in ("full", "quick"):
        f = AUDIT / f"audit-gc-{mode}" / "timing.json"
        if not f.exists():
            continue
        t = json.loads(f.read_text(encoding="utf-8"))
        if t.get("rc") != 0:
            continue
        put_audit(put, mode, t, df)


def put_audit(put, mode: str, t: dict, df: pd.DataFrame | None) -> None:
    g = gpu_of(t.get("gpu", "").replace("Tesla ", "")) or "T4"
    h = (t["audit_seconds"] + t.get("setup_seconds", 0)) / 3600
    put(f"audit.{mode}.hours", f"{h:.1f}")
    put(f"audit.{mode}.gpu", g)
    usd = h * 0.60  # HF Jobs t4-medium (1x T4, 8 vCPU), $0.60/h: the flavor every E7 job uses
    put(f"audit.{mode}.usd", f"{usd:.2f}")
    if df is not None:  # one-off audit cost in millions of tokens served (BASE on an L4 at HF price)
        b = df[(df.gpu == "L4") & (df.config == "BASE")]
        if len(b):
            put(f"audit.{mode}.mtok", f"{usd / b.usd_m_hf.iloc[0]:.0f}")
    v = t.get("validate") or {}
    if v.get("retention"):
        r = v["retention"]
        put(f"audit.{mode}.ret", f"{r['audit']:.2f}")
        put(f"audit.{mode}.retref", f"{r['reference']:.2f}")


def cost_table(df: pd.DataFrame, gpus: list[str]) -> None:
    names = {"AM_lora": "AM (LoRA)", "RL_noguard": "RL no guard"}
    lines = [r"\begin{tabular}{l r " + " ".join("rr" for _ in gpus) + "}", r"\toprule",
             "Fix & prompt tok & " + " & ".join(rf"\multicolumn{{2}}{{c}}{{{g}}}" for g in gpus) + r" \\",
             " & & " + " & ".join(r"tok/s & \$/1M" for _ in gpus) + r" \\", r"\midrule"]
    for c in ORDER:
        x = df[df.config == c]
        if x.empty:
            continue
        cells = []
        for g in gpus:
            r = x[x.gpu == g]
            cells += ([f"{r.tps.iloc[0]:,.0f}", f"{r.usd_m_hf.iloc[0]:.3f}"] if len(r) else ["--", "--"])
        pt = x.prompt_tok.dropna()
        lines.append(rf"\run{{{names.get(c, c).replace('_', chr(92) + '_')}}} & {pt.iloc[0]:.0f} & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (HERE / "tables" / "cost.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
