"""Paper figures (vector PDF). Called by make_assets.py; reads result files and paper/derived/*.csv.

Palette (dataviz reference instance, validated with validate_palette.js --pairs all, light): categorical slots
blue #2a78d6, orange #eb6834, aqua #1baf7a (aqua < 3:1 contrast -> always direct-labelled). Diverging for signed
change: blue #184f95 <- gray #f0efec -> red. Ink #0b0b0b / #52514e, muted #898781, grid #e1e0d9.
Every heatmap cell prints its value, so colour is never the only channel (print / CVD safe).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm  # noqa: E402

PAPER = Path(__file__).resolve().parent
REPO = PAPER.parent
FIGS = PAPER / "figures"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, MUTED, GRID, BASELINE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
DIVERGING = LinearSegmentedColormap.from_list("bgr", ["#184f95", "#86b6ef", "#f0efec", "#ef9a92", "#b3302f"])
COL_W, FULL_W = 3.03, 6.3  # inches: ACL single column / full width

RUNS_TEAM = ["BASE", "SDT", "GC", "IL", "RL", "RL_noguard", "ST", "SP"]
RUNS_AM = ["AM", "AM_rand0", "AM_rand1", "AM_all"]
ATT_COLS = ["A1", "A2", "A3", "A4q3", "A5raw", "A5bon"]
ATT_LABEL = {"A1": "A1\nbenign FT", "A2": "A2\nharmful FT", "A3": "A3\nabliterate", "A4q3": "A4\n3-bit",
             "A5raw": "A5raw\nno template", "A5bon": "A5bon\nbest-of-8", "A3max": "A3max\n(post hoc)"}


def style() -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 7, "axes.titlesize": 7.5, "axes.labelsize": 7,
        "xtick.labelsize": 6.5, "ytick.labelsize": 6.5, "legend.fontsize": 6.5, "axes.edgecolor": BASELINE,
        "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2, "text.color": INK,
        "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42, "svg.fonttype": "none",
        "axes.grid": False, "figure.dpi": 150, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
    })


def save(fig, name: str) -> None:
    FIGS.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGS / f"{name}.pdf")
    fig.savefig(FIGS / f"{name}.png", dpi=200)
    plt.close(fig)


# ----------------------------------------------------------------------------- Fig 1: rise heatmap

def fig_heatmap(rise: pd.DataFrame, a0: dict[str, float]) -> None:
    """Rows = runs (team block, then AM block); cols = attacks (+ A3max, separated). Cell = rise in WildGuard ASR
    vs the run's own A0 (percentage points), bold when the paired 95% CI excludes 0."""
    cols = ATT_COLS + ["A3max"]
    rows = RUNS_TEAM + RUNS_AM
    M = np.full((len(rows), len(cols)), np.nan)
    sig = np.zeros_like(M, dtype=bool)
    for _, r in rise.iterrows():
        if r["run"] in rows and r["attack"] in cols:
            i, j = rows.index(r["run"]), cols.index(r["attack"])
            M[i, j], sig[i, j] = 100 * r["rise"], bool(r["excludes_zero"])
    # visual gaps: one blank row between blocks, one blank column before A3max
    gap_r, gap_c = len(RUNS_TEAM), len(ATT_COLS)
    Mv = np.insert(np.insert(M, gap_r, np.nan, axis=0), gap_c, np.nan, axis=1)
    fig, ax = plt.subplots(figsize=(FULL_W * 0.86, 3.05))
    norm = TwoSlopeNorm(vmin=-30, vcenter=0, vmax=90)
    ax.imshow(np.ma.masked_invalid(Mv), cmap=DIVERGING, norm=norm, aspect="auto")
    R, C = Mv.shape
    for i in range(R):
        ri = i if i < gap_r else i - 1
        for j in range(C):
            cj = j if j < gap_c else j - 1
            if i == gap_r or j == gap_c:
                continue
            v = M[ri, cj]
            if np.isnan(v):
                ax.add_patch(plt.Rectangle((j - .5, i - .5), 1, 1, facecolor="white", edgecolor=GRID, hatch="////",
                                           linewidth=0))
                ax.text(j, i, "not run", ha="center", va="center", fontsize=5.5, color=MUTED)
                continue
            dark = abs(norm(v) - 0.5) > 0.40
            ax.text(j, i, f"{v:+.0f}", ha="center", va="center", fontsize=6.8,
                    fontweight="bold" if sig[ri, cj] else "normal", color="white" if dark else INK)
    ylabels = [f"{r.replace('_', ' ')}  ({a0.get(r, float('nan')):.0f}%)" for r in RUNS_TEAM] + [""] + \
              [f"{r.replace('_', ' ')}  ({a0.get(r, float('nan')):.0f}%)" for r in RUNS_AM]
    ax.set_yticks(range(R), ylabels)
    xl = [ATT_LABEL[c] for c in ATT_COLS] + [""] + [ATT_LABEL["A3max"]]
    ax.set_xticks(range(C), xl)
    ax.xaxis.tick_top()
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_xticks(np.arange(-.5, C, 1), minor=True)
    ax.set_yticks(np.arange(-.5, R, 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.5)
    ax.tick_params(which="minor", length=0)
    ax.text(-0.9, gap_r + 0.05, "our defense and its controls", fontsize=6, color=INK2, va="center", ha="right",
            style="italic", transform=ax.transData)
    save(fig, "fig1_heatmap")


# ----------------------------------------------------------------------------- Fig 2: retention

def fig_retention(ret_lg: pd.DataFrame, ret_wg: pd.DataFrame) -> None:
    ranked = ret_lg[ret_lg["ranked"]].sort_values("retention")
    order = list(ranked["run"])
    ctrl = [r for r in ["AM_rand0", "AM_rand1", "AM_all"] if r in set(ret_lg["run"])]
    rows = ctrl + [""] + order
    fig, ax = plt.subplots(figsize=(COL_W, 2.55))
    for judge, df, color, dy, mk in (("Llama Guard (pre-registered)", ret_lg, BLUE, 0.17, "o"),
                                     ("WildGuard", ret_wg, ORANGE, -0.17, "s")):
        d = df.set_index("run")
        for i, run in enumerate(rows):
            if not run or run not in d.index:
                continue
            r = d.loc[run]
            hollow = run in ctrl
            ax.errorbar(r["retention"], i + dy, xerr=[[r["retention"] - r["lo"]], [r["hi"] - r["retention"]]],
                        fmt=mk, ms=4, color=color, mfc="white" if hollow else color, mec=color, elinewidth=1,
                        capsize=0, label=judge if i == len(rows) - 1 else None)
    ax.set_yticks(range(len(rows)), [r.replace("_", " ") for r in rows])
    ax.axhline(len(ctrl), color=GRID, linewidth=0.8)
    ax.text(0.02, len(ctrl) - 0.45, "controls: A2 + A3 only, not ranked", fontsize=5.5, color=MUTED,
            transform=ax.get_yaxis_transform(), va="top")
    ax.set_xlabel("Retention = 1 − mean ASR rise over A1–A5 (higher is better)")
    ax.set_xlim(0, 1)
    ax.xaxis.grid(True, color=GRID, linewidth=0.5)
    ax.set_axisbelow(True)
    ax.legend(loc="lower center", bbox_to_anchor=(0.45, 1.0), ncol=2, frameon=False, handletextpad=0.3,
              columnspacing=1.0)
    save(fig, "fig2_retention")


# ----------------------------------------------------------------------------- Fig 4: judge agreement

def fig_judges(h: pd.DataFrame) -> None:
    judges = [("Llama Guard\n(unsafe)", "Llama Guard unsafe vs human: harmful help", BLUE),
              ("WildGuard\n(harmful)", "WildGuard harmful vs human: harmful help", ORANGE),
              ("keyword\n(complied)", "keyword complied vs human: not a refusal", AQUA)]
    scopes = [("all", "all harmful-prompt\nreplies"), ("other attacks", "A0–A4"), ("A5 raw + BoN", "A5 (no template,\nbest-of-8)")]
    fig, ax = plt.subplots(figsize=(COL_W, 1.85))
    w = 0.26
    for k, (jl, pair, color) in enumerate(judges):
        for s, (sc, _) in enumerate(scopes):
            r = h[(h["pair"] == pair) & (h["scope"] == sc)].iloc[0]
            x = s + (k - 1) * w
            ax.bar(x, r["kappa"], width=w - 0.03, color=color, label=jl.replace("\n", " ") if s == 0 else None)
            ax.text(x, max(r["kappa"], 0) + 0.015, f"{r['kappa']:.2f}", ha="center", va="bottom", fontsize=5.8)
    ns = {sc: int(h[(h["pair"] == judges[0][1]) & (h["scope"] == sc)].iloc[0]["n"]) for sc, _ in scopes}
    ax.set_xticks(range(len(scopes)), [f"{lab}\n(n = {ns[sc]})" for sc, lab in scopes])
    ax.set_ylabel("Cohen's κ vs blind hand labels")
    ax.set_ylim(0, 0.72)
    ax.yaxis.grid(True, color=GRID, linewidth=0.5)
    ax.set_axisbelow(True)
    ax.axhline(0, color=BASELINE, linewidth=0.8)
    ax.legend(ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.18), frameon=False, columnspacing=0.8)
    save(fig, "fig4_judges")


# ----------------------------------------------------------------------------- Fig 3: capability (E1)

CAP_COLS = ["mmlu300", "arc_challenge", "hellaswag", "winogrande", "ifeval", "gsm8k", "comp"]
CAP_LABEL = {"mmlu300": "MMLU", "arc_challenge": "ARC-C", "hellaswag": "HellaS.", "winogrande": "Wino.",
             "ifeval": "IFEval", "gsm8k": "GSM8K", "comp": "mean"}


def fig_capability(delta: pd.DataFrame, levels: dict[str, float]) -> None:
    """Rows = runs. Left block: each fix at release vs BASE (composite). Right block: change after A2 vs the run's
    own A0, per benchmark and composite; later attacks add one composite column each. Points, bold = paired 95%
    bootstrap CI excludes 0. Red = capability lost (same 'red = bad' reading as Fig. 1)."""
    rows = RUNS_TEAM + RUNS_AM
    later = [a for a in ("A3", "A4q3", "A1") if ((delta["attack"] == a) & (delta["task"] == "comp")).any()]
    cols = [("A0", "comp")] + [("A2", t) for t in CAP_COLS] + [(a, "comp") for a in later]
    get = {(r["run"], r["attack"], r["task"]): r for _, r in delta.iterrows()}
    M = np.full((len(rows), len(cols)), np.nan)
    sig = np.zeros_like(M, dtype=bool)
    for i, run in enumerate(rows):
        for j, key in enumerate(cols):
            r = get.get((run, *key))
            if r is not None:
                M[i, j], sig[i, j] = 100 * r["delta"], bool(r["excludes_zero"])
    if run_base := "BASE":  # BASE is the reference of the A0 column
        M[rows.index(run_base), 0] = 0.0
    gap_r = len(RUNS_TEAM)
    gap_cs = [1] + ([1 + len(CAP_COLS)] if later else [])
    Mv = np.insert(M, gap_r, np.nan, axis=0)
    for g in reversed(gap_cs):
        Mv = np.insert(Mv, g, np.nan, axis=1)
    colmap = [j for j in range(len(cols))]
    vis_cols = []
    k = 0
    for j in range(Mv.shape[1]):
        if j in [g + n for n, g in enumerate(gap_cs)]:
            vis_cols.append(None)
        else:
            vis_cols.append(colmap[k])
            k += 1
    fig, ax = plt.subplots(figsize=(COL_W, 2.75))
    norm = TwoSlopeNorm(vmin=-8, vcenter=0, vmax=8)
    ax.imshow(np.ma.masked_invalid(Mv), cmap=DIVERGING.reversed(), norm=norm, aspect="auto")
    R, C = Mv.shape
    for i in range(R):
        if i == gap_r:
            continue
        ri = i if i < gap_r else i - 1
        for j in range(C):
            cj = vis_cols[j]
            if cj is None:
                continue
            v = M[ri, cj]
            if np.isnan(v):
                ax.text(j, i, "·", ha="center", va="center", fontsize=6, color=MUTED)
                continue
            if rows[ri] == "BASE" and cj == 0:
                ax.text(j, i, "ref", ha="center", va="center", fontsize=5, color=MUTED)
                continue
            dark = abs(norm(v) - 0.5) > 0.40
            ax.text(j, i, f"{v:+.1f}".replace("+0.0", "0.0").replace("-0.0", "0.0"), ha="center", va="center",
                    fontsize=5, fontweight="bold" if sig[ri, cj] else "normal", color="white" if dark else INK)
    ylabels = [f"{r.replace('_', ' ')} ({levels.get(r, float('nan')):.1f})" for r in RUNS_TEAM] + [""] + \
              [f"{r.replace('_', ' ')} ({levels.get(r, float('nan')):.1f})" for r in RUNS_AM]
    ax.set_yticks(range(R), ylabels)
    xl = []
    for j in range(C):
        cj = vis_cols[j]
        if cj is None:
            xl.append("")
            continue
        att, t = cols[cj]
        xl.append("A0 vs\nBASE" if att == "A0" else (CAP_LABEL[t] if att == "A2" else f"{att}\nmean"))
    ax.set_xticks(range(C), xl, rotation=90)
    ax.xaxis.tick_top()
    ax.tick_params(length=0, labelsize=5.5)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.set_xticks(np.arange(-.5, C, 1), minor=True)
    ax.set_yticks(np.arange(-.5, R, 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.2)
    ax.tick_params(which="minor", length=0)
    a2_first, a2_last = gap_cs[0] + 1, gap_cs[0] + len(CAP_COLS)
    ax.text((a2_first + a2_last) / 2, R - 0.3, "after A2 (harmful fine-tune), vs own A0", ha="center", va="top",
            fontsize=5.5, color=INK2)
    save(fig, "fig3_capability")


# ----------------------------------------------------------------------------- Fig 6: multilingual (E6)

ML_LANGS = ["en", "es", "fr", "zh", "sw", "yo"]
ML_LABEL = {"en": "English", "es": "Spanish", "fr": "French", "zh": "Chinese", "sw": "Swahili", "yo": "Yoruba"}
ML_ATT = [("A0", "as released", BLUE), ("A2", "harmful FT", ORANGE), ("A3", "abliterate", AQUA),
          ("A4q3", "3-bit", MUTED)]


def fig_multilingual(rates: pd.DataFrame) -> None:
    """Mean harmful-reply rate (WildGuard, incoherent = not harmful) over runs, per language and attack; small dots
    show each run, so the reader sees whether one model drives the mean."""
    fig, ax = plt.subplots(figsize=(COL_W, 2.1))
    atts = [a for a in ML_ATT if a[0] in set(rates["attack"])]
    w = 0.8 / max(len(atts), 1)
    for k, (att, lab, col) in enumerate(atts):
        g = rates[rates["attack"] == att]
        for i, lang in enumerate(ML_LANGS):
            v = 100 * g[g["lang"] == lang]["asr_wg"]
            if v.empty:
                continue
            x = i - 0.4 + w * (k + 0.5)
            ax.scatter(np.full(len(v), x), v, s=4, color=col, alpha=0.35, linewidths=0)
            ax.plot([x - w * 0.4, x + w * 0.4], [v.mean()] * 2, color=col, lw=1.6,
                    label=f"{att} {lab}" if i == 0 else None)
    ax.set_xticks(range(len(ML_LANGS)), [ML_LABEL[l] for l in ML_LANGS])
    ax.set_ylabel("Harmful-reply rate, %")
    ax.set_ylim(0, 100)
    ax.yaxis.grid(True, color=GRID, linewidth=0.5)
    ax.set_axisbelow(True)
    ax.legend(ncol=2, loc="upper left", frameon=False, handlelength=1.2, columnspacing=0.8)
    save(fig, "fig6_multilingual")


# ----------------------------------------------------------------------------- main-text scatter

def fig_scatter_am(ci: pd.DataFrame) -> None:
    """MMLU vs WildGuard ASR: each run's move from A0 (hollow) to A3 abliteration (filled)."""
    runs = [("BASE", AQUA), ("GC", BLUE), ("AM", ORANGE), ("AM_rand0", ORANGE), ("AM_rand1", ORANGE)]
    get = {(r["run"], r["attack"], r["metric"]): r for _, r in ci.iterrows()}
    fig, ax = plt.subplots(figsize=(COL_W, 2.15))
    offsets = {"BASE": (4, 2), "GC": (4, 3), "AM": (4, -1), "AM_rand0": (-30, -1), "AM_rand1": (4, 4)}
    for run, color in runs:
        pts = []
        for att in ("A0", "A3"):
            m, a = get.get((run, att, "mmlu")), get.get((run, att, "asr_wg"))
            if m is None or a is None:
                continue
            x, y = 100 * m["value"], 100 * a["value"]
            ax.errorbar(x, y, xerr=[[x - 100 * m["lo"]], [100 * m["hi"] - x]],
                        yerr=[[y - 100 * a["lo"]], [100 * a["hi"] - y]], fmt="D" if run == "AM_rand1" else "o", ms=4.5, color=color,
                        mfc="white" if att == "A0" else color, mec=color, elinewidth=0.7, alpha=0.95,
                        ls="none")
            pts.append((x, y))
        if len(pts) == 2:
            ax.annotate("", xy=pts[1], xytext=pts[0], arrowprops=dict(arrowstyle="-|>", color=color, lw=0.8,
                                                                       shrinkA=4, shrinkB=4, alpha=0.7))
            dx, dy = offsets[run]
            ax.annotate(run.replace("_", " "), pts[1], xytext=(dx, dy), textcoords="offset points", fontsize=6,
                        color=INK)
    ax.set_xlabel("MMLU accuracy, % (300 items)")
    ax.set_ylabel("Harmful-reply rate, % (WildGuard)")
    ax.set_ylim(-5, 100)
    ax.yaxis.grid(True, color=GRID, linewidth=0.5)
    ax.set_axisbelow(True)
    ax.text(0.98, 0.04, "better ↘", transform=ax.transAxes, ha="right", fontsize=6.5, color=INK2)
    ax.text(0.02, 0.97, "hollow: as released (A0)   filled: after abliteration (A3)", transform=ax.transAxes,
            fontsize=5.8, color=INK2, va="top")
    save(fig, "fig5_scatter_am")


def make_all(V: dict) -> None:
    from make_assets import REPO as R, rise_table, stage1_cells

    style()
    ci = stage1_cells()
    a0 = {r["run"]: 100 * r["value"] for _, r in ci[(ci["attack"] == "A0") & (ci["metric"] == "asr_wg")].iterrows()}
    rise = rise_table("asr_wg")
    fig_heatmap(rise, a0)
    rise_lg = rise_table("asr_lg")
    for judge, r in (("wg", rise), ("lg", rise_lg)):
        for att, g in r.groupby("attack"):
            V[f"rise.{judge}.{att}.maxabs"] = f"{100 * g['rise'].abs().max():.0f}"
    tables = R / "results" / "tables"
    ret_lg, ret_wg = pd.read_csv(tables / "retention.csv"), pd.read_csv(tables / "retention_wg.csv")
    fig_retention(ret_lg[ret_lg["stage"] == "test"], ret_wg[ret_wg["stage"] == "test"])
    fig_judges(pd.read_csv(tables / "human_agreement.csv"))
    fig_scatter_am(ci)
    dpath = R / "paper" / "derived" / "capability_delta.csv"
    if dpath.exists():
        from make_assets import capability_items

        items = capability_items()
        a0 = items[items["attack"] == "A0"].groupby(["run", "task"])["y"].mean().groupby("run").mean()
        fig_capability(pd.read_csv(dpath), {r: 100 * v for r, v in a0.items()})
    mpath = R / "paper" / "derived" / "ml_rates.csv"
    if mpath.exists():
        fig_multilingual(pd.read_csv(mpath))
