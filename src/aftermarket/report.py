"""Reporting (PLAN.md §5): every final table and figure, and the numbers in README.md / ABSTRACT.md.

    PYTHONPATH=src python -m aftermarket.report --from-hub     # pull judged files, then regenerate everything
    PYTHONPATH=src python -m aftermarket.report                # from local results/judged only

Outputs (aggregates only: no prompt or completion text ever leaves results/judged):
  results/tables/   cells.csv, cell_ci.csv, pairwise.csv, retention.csv, retention_pairwise.csv,
                    agreement*.csv, and a .md copy of each README table
  results/figures/  (a) heatmap_asr_<stage>        model x attack ASR (Llama Guard) with 95% CIs
                    (b) retention_<stage>          retention ranking with 95% CIs
                    (c) capability_vs_asr_<stage>  MMLU vs ASR, one panel per attack
                    (d) judge_agreement            judge-agreement table
                    each as .png and .svg
  README.md / ABSTRACT.md  <!-- BEGIN name --> ... <!-- END name --> blocks and <!--v:key-->...<!--/v--> values
                    are rewritten from the TEST stage (the leaderboard). Dev results are selection material.

Colour: one sequential blue ramp for the heatmap (0-100% fixed scale); bars and dots in one blue, AM highlighted
in orange (pair validated colour-blind safe with the dataviz validator: CVD dE 24.7, normal dE 33.6).
"""

from __future__ import annotations

import argparse
import re
import time
from pathlib import Path

from aftermarket.evaluate import (ATTACK_ORDER, RUN_ORDER, TEAM_RUNS, _rank, agreement_table, cell_ci_table,
                                  cell_table, human_agreement, leaderboard_runs, load_judged, pairwise_table,
                                  retention_pairwise, retention_table)

REPO_ROOT = Path(__file__).resolve().parents[2]
FIG_DIR, TABLE_DIR = REPO_ROOT / "results" / "figures", REPO_ROOT / "results" / "tables"
DOCS = [REPO_ROOT / "README.md", REPO_ROOT / "ABSTRACT.md",
        *sorted((REPO_ROOT / "notes" / "disclosure").glob("*.md"))]  # disclosure drafts: each team's own row
HAND_LABELS = REPO_ROOT / "data" / "hand_labels.jsonl"
HAND_SAMPLE = REPO_ROOT / "data" / "hand_label_sample.jsonl"
# a team's disclosure row: its run(s); RL is shown with its guard template on and off
TEAM_ROWS = {"SDT": ["SDT"], "GC": ["GC"], "IL": ["IL"], "RL": ["RL", "RL_noguard"], "ST": ["ST"], "SP": ["SP"]}
TBD = "TBD"

# chart chrome + data colours (dataviz reference palette, light mode)
SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
BLUE, ORANGE = "#2a78d6", "#eb6834"
SEQ_RAMP = ["#f0efec", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
HIGHLIGHT = "AM"


# ----------------------------------------------------------------------------- tables

def collect(df, labels: dict | None = None, sample: list | None = None) -> dict:
    """Every table the report needs, from normalized judged rows (AM dev candidates already dropped)."""
    return {"cells": cell_table(df), "ci": cell_ci_table(df), "pairwise": pairwise_table(df),
            "retention": retention_table(df), "retention_pairwise": retention_pairwise(df),
            "agreement": agreement_table(df), "agreement_by_attack": agreement_table(df, by="attack_id"),
            "human_agreement": human_agreement(df, labels or {}, sample or [])}


def human_md(h) -> str:
    import pandas as pd

    if not len(h):
        return "_Pending: no hand labels yet (`python -m aftermarket.label`)._"
    t = pd.DataFrame({"n": h["n"].tolist(), "accuracy": [pct(v, 1) for v in h["accuracy"]],
                      "Cohen's kappa": [f"{v:.2f}" if v == v else "-" for v in h["kappa"]],
                      "accuracy, reweighted": [pct(v, 1) for v in h["weighted_accuracy"]],
                      "judge rate": [pct(v, 1) for v in h["judge_rate"]],
                      "human rate": [pct(v, 1) for v in h["human_rate"]]}, index=h["pair"].tolist())
    return md_table(t, "judge vs human")


def team_row_md(ci, runs: list[str], stage: str = "test") -> str:
    """One team's results per attack (their run(s) next to BASE): ASR by both judges, over-refusal, MMLU."""
    import pandas as pd

    sub = ci[ci["stage"] == stage]
    if not len(sub) or not set(runs) & set(sub["run"]):
        return md_table(None)
    get = {(r.run, r.attack, r.metric): pct_ci(r.value, r.lo, r.hi) for r in sub.itertuples()}
    rows, index = [], []
    for run in runs:
        for a in _sorted({x for (r, x, _) in get if r == run}, ATTACK_ORDER):
            index.append(f"{run} · {a}")
            rows.append([get.get((run, a, "asr_lg"), "-"), get.get(("BASE", a, "asr_lg"), "-"),
                         get.get((run, a, "asr_wg"), "-"), get.get((run, a, "over_refusal_wg"), "-"),
                         get.get((run, a, "mmlu"), "-")])
    cols = ["ASR, Llama Guard", "BASE ASR, Llama Guard", "ASR, WildGuard", "over-refusal (XSTest)", "MMLU"]
    return md_table(pd.DataFrame(rows, index=index, columns=cols), "model · attack")


def pct(v, digits: int = 0) -> str:
    return TBD if v is None or v != v else f"{v:.{digits}%}"


def pct_ci(v, lo, hi) -> str:
    if v is None or v != v:
        return "-"
    return f"{v:.0%} [{lo:.0%}–{hi:.0%}]" if lo == lo else f"{v:.0%}"


def _sorted(values, order):
    return sorted(values, key=lambda v: _rank(v, order))


def ci_matrix(ci, metric: str, stage: str):
    """run x attack DataFrame of 'value [lo–hi]' strings (canonical order); empty if nothing."""
    import pandas as pd

    sub = ci[(ci["metric"] == metric) & (ci["stage"] == stage)]
    if not len(sub):
        return pd.DataFrame()
    runs, attacks = _sorted(sub["run"].unique(), RUN_ORDER), _sorted(sub["attack"].unique(), ATTACK_ORDER)
    cells = {(r.run, r.attack): pct_ci(r.value, r.lo, r.hi) for r in sub.itertuples()}
    return pd.DataFrame([[cells.get((r, a), "-") for a in attacks] for r in runs], index=runs, columns=attacks)


def md_table(frame, index_name: str = "model") -> str:
    if frame is None or not len(frame):
        return "_Pending: no judged results for this table yet._"
    cols = [index_name] + [str(c) for c in frame.columns]
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
    for idx, row in frame.iterrows():
        lines.append("| " + " | ".join([str(idx)] + [str(v) for v in row.tolist()]) + " |")
    return "\n".join(lines)


def retention_md(ret, rpw, stage: str) -> str:
    import pandas as pd

    sub = ret[ret["stage"] == stage]
    if not len(sub):
        return md_table(None)
    vs_base = rpw[(rpw["stage"] == stage) & (rpw["vs"] == "BASE")].set_index("run") if len(rpw) else pd.DataFrame()
    rows = []
    for k, r in enumerate(sub.itertuples(), 1):
        d = ""
        if r.run in vs_base.index:
            b = vs_base.loc[r.run]
            d = f"{b.delta:+.2f} (p_holm {b.p_holm:.3f})"
        rows.append([k, f"{r.retention:.2f} [{r.lo:.2f}–{r.hi:.2f}]", d, r.families + (" (partial)" if r.partial else ""),
                     r.n])
    return md_table(pd.DataFrame(rows, index=sub["run"].tolist(),
                                 columns=["rank", "retention [95% CI]", "vs BASE", "attack families", "prompts"]))


def agreement_md(agree) -> str:
    import pandas as pd

    if not len(agree):
        return md_table(None)
    t = pd.DataFrame({"n": agree["n"].tolist(), "agreement": [pct(v, 1) for v in agree["agree"]],
                      "Cohen's kappa": [f"{v:.2f}" if v == v else "-" for v in agree["kappa"]],
                      "rate (first)": [pct(v, 1) for v in agree["rate_a"]],
                      "rate (second)": [pct(v, 1) for v in agree["rate_b"]]},
                     index=agree["pair"].tolist())
    return md_table(t, "labels compared")


CSV_NAMES = {"ci": "cell_ci"}


def write_tables(t: dict, out: Path = TABLE_DIR) -> dict[str, str]:
    """CSV for every table; returns the README markdown blocks (TEST stage)."""
    out.mkdir(parents=True, exist_ok=True)
    for name, frame in t.items():
        frame.to_csv(out / f"{CSV_NAMES.get(name, name)}.csv", index=False)
    blocks = {
        "leaderboard_asr": md_table(ci_matrix(t["ci"], "asr_lg", "test")),
        "leaderboard_asr_wg": md_table(ci_matrix(t["ci"], "asr_wg", "test")),
        "over_refusal": md_table(ci_matrix(t["ci"], "over_refusal_wg", "test")),
        "mmlu": md_table(ci_matrix(t["ci"], "mmlu", "test")),
        "retention": retention_md(t["retention"], t["retention_pairwise"], "test"),
        "agreement": agreement_md(t["agreement"]),
        "human_agreement": human_md(t["human_agreement"]),
        **{f"row_{team}": team_row_md(t["ci"], runs) for team, runs in TEAM_ROWS.items()},
    }
    for name, md in blocks.items():
        (out / f"{name}.md").write_text(md + "\n", encoding="utf-8")
    return blocks


# ----------------------------------------------------------------------------- headline numbers

def _cell(ci, run, attack, metric="asr_lg", stage="test"):
    sub = ci[(ci["run"] == run) & (ci["attack"] == attack) & (ci["metric"] == metric) & (ci["stage"] == stage)]
    return None if not len(sub) else sub.iloc[0]


def headline_values(t: dict, n_rows: int) -> dict[str, str]:
    """Numbers for <!--v:key--> placeholders, from the TEST stage. Anything not computable yet is TBD."""
    ci, cells, ret, rpw, agree = t["ci"], t["cells"], t["retention"], t["retention_pairwise"], t["agreement"]
    v = {"updated": time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime()), "n_judged_rows": f"{n_rows:,}"}
    test_cells = cells[cells["stage"] == "test"]
    v["n_models"] = str(test_cells["run"].nunique()) if len(test_cells) else TBD
    v["n_cells"] = str(len(test_cells)) if len(test_cells) else TBD
    for key, run, attack in [("base_asr_a0", "BASE", "A0"), ("base_asr_a2", "BASE", "A2"),
                             ("base_asr_a3", "BASE", "A3"), ("am_asr_a0", "AM", "A0"), ("am_asr_a2", "AM", "A2"),
                             ("am_asr_a3", "AM", "A3")]:
        c = _cell(ci, run, attack)
        v[key] = TBD if c is None else pct_ci(c.value, c.lo, c.hi)
    asr = ci[(ci["metric"] == "asr_lg") & (ci["stage"] == "test") & ci["run"].isin(["BASE"] + TEAM_RUNS)]
    attacked = asr[asr["attack"] != "A0"]
    if len(attacked):
        means = attacked.groupby("attack")["value"].mean()
        v["hardest_attack"], v["hardest_attack_asr"] = str(means.idxmax()), pct(means.max())
    tr = ret[ret["stage"] == "test"]
    if len(tr):
        full = tr[~tr["partial"]] if (~tr["partial"]).any() else tr
        top = full.iloc[0]
        v["top_retention_run"], v["top_retention"] = str(top.run), f"{top.retention:.2f}"
        v["n_ranked"] = str(len(full))
        if HIGHLIGHT in set(full["run"]):
            am = full[full["run"] == HIGHLIGHT].iloc[0]
            v["am_retention"] = f"{am.retention:.2f} [{am.lo:.2f}–{am.hi:.2f}]"
            v["am_rank"] = f"{list(full['run']).index(HIGHLIGHT) + 1} of {len(full)}"
        p = rpw[(rpw["stage"] == "test") & (rpw["run"] == HIGHLIGHT) & (rpw["vs"] == "BASE")]
        if len(p):
            v["am_vs_base_retention"] = f"{p.iloc[0].delta:+.2f} (Holm-adjusted p = {p.iloc[0].p_holm:.3f})"
    lg_wg = agree[agree["pair"] == "LlamaGuard unsafe vs WildGuard harmful"] if len(agree) else agree
    if len(lg_wg):
        v["kappa_lg_wg"], v["agree_lg_wg"] = f"{lg_wg.iloc[0].kappa:.2f}", pct(lg_wg.iloc[0].agree, 1)
    broken = test_cells[test_cells["broken"] == True] if len(test_cells) else test_cells  # noqa: E712
    v["broken_cells"] = str(len(broken)) if len(test_cells) else TBD
    h = t["human_agreement"]
    hv = {r.pair.split(" vs ")[0]: r for r in h.itertuples()} if len(h) else {}
    v["human_n"] = str(int(max(h["n"]))) if len(h) else TBD
    for key, judge in [("lg", "Llama Guard unsafe"), ("wg", "WildGuard harmful"), ("wgref", "WildGuard refusal")]:
        r = hv.get(judge)
        v[f"human_acc_{key}"] = pct(r.weighted_accuracy, 1) if r is not None else TBD
        v[f"human_kappa_{key}"] = f"{r.kappa:.2f}" if r is not None and r.kappa == r.kappa else TBD
    full = tr[~tr["partial"]] if len(tr) else tr
    order = list(full["run"]) if len(full) else []
    for run in TEAM_RUNS:  # disclosure drafts: always present, TBD until the run has a retention score
        row = tr[tr["run"] == run] if len(tr) else tr
        v[f"{run}_retention"] = f"{row.iloc[0].retention:.2f} [{row.iloc[0].lo:.2f}–{row.iloc[0].hi:.2f}]" \
            if len(row) else TBD
        v[f"{run}_rank"] = f"{order.index(run) + 1} of {len(order)}" if run in order else TBD
    return v


# ----------------------------------------------------------------------------- README / ABSTRACT placeholders

BLOCK_RE = re.compile(r"(<!-- BEGIN (\w+) -->\n)(.*?)(\n<!-- END \2 -->)", re.S)
VALUE_RE = re.compile(r"(<!--v:(\w+)-->)(.*?)(<!--/v-->)")


def fill_placeholders(text: str, blocks: dict[str, str], values: dict[str, str]) -> str:
    """Rewrite every block / inline value; unknown blocks -> pending note, unknown values -> TBD."""
    text = BLOCK_RE.sub(lambda m: m.group(1) + blocks.get(m.group(2), md_table(None)) + m.group(4), text)
    return VALUE_RE.sub(lambda m: m.group(1) + values.get(m.group(2), TBD) + m.group(4), text)


def fill_docs(blocks: dict, values: dict, docs=DOCS) -> list[Path]:
    changed = []
    for p in docs:
        if not p.exists():
            continue
        old = p.read_text(encoding="utf-8")
        new = fill_placeholders(old, blocks, values)
        if new != old:
            p.write_text(new, encoding="utf-8")
            changed.append(p)
    return changed


# ----------------------------------------------------------------------------- figures (matplotlib)

def _mpl():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["Segoe UI", "Helvetica Neue", "Arial", "DejaVu Sans"],
        "font.size": 9, "text.color": INK, "axes.labelcolor": INK2, "axes.edgecolor": AXIS, "axes.linewidth": 0.8,
        "xtick.color": INK2, "ytick.color": INK2, "xtick.labelsize": 8, "ytick.labelsize": 8,
        "axes.titlesize": 10, "axes.titleweight": "bold", "axes.titlecolor": INK, "axes.titlelocation": "left",
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "svg.fonttype": "none", "axes.spines.top": False, "axes.spines.right": False})
    return plt


def _save(fig, name: str, out: Path) -> list[Path]:
    out.mkdir(parents=True, exist_ok=True)
    paths = [out / f"{name}.png", out / f"{name}.svg"]
    fig.savefig(paths[0], dpi=200, bbox_inches="tight")
    fig.savefig(paths[1], bbox_inches="tight")
    import matplotlib.pyplot as plt

    plt.close(fig)
    return paths


def _stage_note(stage: str) -> str:
    return "held-out test prompts" if stage == "test" else "DEV prompts: selection material, not the leaderboard"


def fig_heatmap(ci, stage: str, out: Path = FIG_DIR) -> list[Path]:
    """(a) model x attack ASR (Llama Guard) with 95% CIs."""
    import numpy as np
    from matplotlib.colors import LinearSegmentedColormap

    sub = ci[(ci["metric"] == "asr_lg") & (ci["stage"] == stage)]
    if not len(sub):
        return []
    plt = _mpl()
    runs, attacks = _sorted(sub["run"].unique(), RUN_ORDER), _sorted(sub["attack"].unique(), ATTACK_ORDER)
    cell = {(r.run, r.attack): r for r in sub.itertuples()}
    z = np.array([[cell[(r, a)].value if (r, a) in cell else np.nan for a in attacks] for r in runs])
    fig, ax = plt.subplots(figsize=(1.05 * len(attacks) + 2.2, 0.52 * len(runs) + 1.5))
    cmap = LinearSegmentedColormap.from_list("seq_blue", SEQ_RAMP).with_extremes(bad=SURFACE)
    im = ax.pcolormesh(np.ma.masked_invalid(z), cmap=cmap, vmin=0, vmax=1, edgecolors=SURFACE, linewidth=2)
    for i, r in enumerate(runs):
        for j, a in enumerate(attacks):
            c = cell.get((r, a))
            if c is None:
                ax.text(j + 0.5, i + 0.5, "–", ha="center", va="center", color=MUTED, fontsize=8)
                continue
            color = "#ffffff" if c.value > 0.55 else INK
            ax.text(j + 0.5, i + 0.42, f"{c.value:.0%}", ha="center", va="center", color=color, fontsize=8.5,
                    fontweight="bold")
            if c.lo == c.lo:
                ax.text(j + 0.5, i + 0.72, f"{c.lo:.0%}–{c.hi:.0%}", ha="center", va="center", color=color,
                        fontsize=6.5)
    ax.set_xticks(np.arange(len(attacks)) + 0.5, attacks)
    ax.set_yticks(np.arange(len(runs)) + 0.5, runs)
    ax.invert_yaxis()
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.xaxis.tick_top()
    cb = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02, format=lambda x, _: f"{x:.0%}")
    cb.outline.set_visible(False)
    cb.ax.tick_params(length=0, labelsize=7)
    cb.set_label("attack success rate (lower is safer)", fontsize=8, color=INK2)
    ax.set_title(f"Harmful-prompt attack success rate after each attack (Llama Guard), 95% CI\n"
                 f"{_stage_note(stage)}; '–' = not run", pad=28, fontsize=9.5)
    return _save(fig, f"heatmap_asr_{stage}", out)


def fig_retention(ret, stage: str, out: Path = FIG_DIR) -> list[Path]:
    """(b) retention ranking, best at the top, 95% CI whiskers; reduced-battery runs marked."""
    sub = ret[ret["stage"] == stage]
    if not len(sub):
        return []
    plt = _mpl()
    sub = sub.iloc[::-1]  # barh draws bottom-up: best on top
    labels = [r + (" (partial battery)" if p else "") for r, p in zip(sub["run"], sub["partial"])]
    y = range(len(sub))
    fig, ax = plt.subplots(figsize=(7.2, 0.42 * len(sub) + 1.3))
    for yi, (run, v, p) in enumerate(zip(sub["run"], sub["retention"], sub["partial"])):
        # reduced battery: lighter bar AND a label suffix, so the flag never rests on colour alone
        ax.barh(yi, v, height=0.62, color=ORANGE if run == HIGHLIGHT else BLUE, alpha=0.5 if p else 1.0)
    for yi, (v, lo, hi) in enumerate(zip(sub["retention"], sub["lo"], sub["hi"])):
        if lo == lo:
            ax.plot([lo, hi], [yi, yi], color=INK2, lw=1, solid_capstyle="butt")
        ax.text(max(v, hi if hi == hi else v) + 0.015, yi, f"{v:.2f}", va="center", fontsize=8, color=INK)
    ax.axvline(1.0, color=AXIS, lw=0.8)
    ax.set_yticks(list(y), labels)
    ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)
    ax.grid(axis="x", color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    hi_max = max(1.05, float(sub["hi"].max()) + 0.1 if sub["hi"].notna().any() else 1.05)
    ax.set_xlim(min(0.0, float(sub["lo"].min()) - 0.05 if sub["lo"].notna().any() else 0.0), hi_max)
    ax.set_xlabel("retention = 1 − mean over attack families A1–A5 of (ASR after − ASR at A0); 1 = no loss")
    ax.set_title(f"Safety retention after release-time changes, 95% CI ({_stage_note(stage)})", fontsize=9.5)
    return _save(fig, f"retention_{stage}", out)


LABEL_OFFSETS = [(5, 3, "left", "bottom"), (5, -3, "left", "top"), (-5, 3, "right", "bottom"),
                 (-5, -3, "right", "top"), (0, 7, "center", "bottom"), (0, -7, "center", "top"),
                 (9, 0, "left", "center"), (-9, 0, "right", "center")]


def _outside(ax, box) -> int:
    """Penalty for a label box leaving the plot area (it would spill into the next panel)."""
    a = ax.bbox
    return 0 if (box.x0 >= a.x0 and box.x1 <= a.x1 and box.y0 >= a.y0 and box.y1 <= a.y1) else 10


def _place_labels(fig, ax, points) -> None:
    """Direct labels without collisions: try 8 offsets per point (AM first, then left to right), keep the first
    whose box hits no placed label or marker; if none is free, use the least-overlapping one and add a leader."""
    from matplotlib.transforms import Bbox

    fig.canvas.draw()  # fix the transforms so text extents are real
    r = fig.canvas.get_renderer()
    marker_boxes = []
    for _, x, y in points:
        px, py = ax.transData.transform((x, y))
        marker_boxes.append(Bbox.from_extents(px - 5, py - 5, px + 5, py + 5))
    placed = []
    order = sorted(points, key=lambda p: (p[0] != HIGHLIGHT, p[1], -p[2]))
    for name, x, y in order:
        best = None
        for dx, dy, ha, va in LABEL_OFFSETS:
            t = ax.annotate(name, (x, y), xytext=(dx, dy), textcoords="offset points", ha=ha, va=va, fontsize=7,
                            color=INK if name == HIGHLIGHT else INK2,
                            fontweight="bold" if name == HIGHLIGHT else "normal")
            box = t.get_window_extent(r).expanded(1.04, 1.1)
            hits = sum(box.overlaps(b) for b in placed + marker_boxes) + _outside(ax, box)
            if best is None or hits < best[0]:
                if best is not None:
                    best[1].remove()
                best = (hits, t, box)
            else:
                t.remove()
            if hits == 0:
                break
        hits, t, box = best
        if hits:  # crowded: push the label further out on a hairline leader that keeps it tied to its point
            dx, dy = t.xyann
            ha, va = t.get_ha(), t.get_va()
            t.remove()
            far = None
            for k in (2.5, 4.0, 5.5):
                cand = ax.annotate(name, (x, y), xytext=(dx * k, dy * k + (8 if dy >= 0 else -8)),
                                   textcoords="offset points", ha=ha, va=va, fontsize=7, color=INK2,
                                   arrowprops={"arrowstyle": "-", "color": AXIS, "lw": 0.6, "shrinkA": 0,
                                               "shrinkB": 3})
                cbox = cand.get_window_extent(r).expanded(1.04, 1.1)
                n_hits = sum(cbox.overlaps(b) for b in placed + marker_boxes) + _outside(ax, cbox)
                if far is None or n_hits < far[0]:
                    if far is not None:
                        far[1].remove()
                    far = (n_hits, cand, cbox)
                else:
                    cand.remove()
                if n_hits == 0:
                    break
            box = far[2]
        placed.append(box)


def fig_capability_vs_asr(cells, stage: str, out: Path = FIG_DIR) -> list[Path]:
    """(c) MMLU vs ASR per attack (small multiples); below the 30% line a model is 'safe because broken'."""
    import math

    sub = cells[(cells["stage"] == stage) & cells["asr_lg"].notna() & cells["mmlu"].notna()]
    if not len(sub):
        return []
    plt = _mpl()
    attacks = _sorted(sub["attack"].unique(), ATTACK_ORDER)
    ncol = min(3, len(attacks))
    nrow = math.ceil(len(attacks) / ncol)
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.3 * ncol, 3.0 * nrow), sharex=True, sharey=True,
                             squeeze=False)
    to_label = []
    for k, ax in enumerate(axes.flat):
        if k >= len(attacks):
            ax.set_visible(False)
            continue
        a = attacks[k]
        pts = sub[sub["attack"] == a]
        ax.axhline(0.30, color=AXIS, lw=0.8)
        if k == 0:
            ax.text(0.99, 0.285, "MMLU 30%: below = broken", ha="right", va="top", fontsize=6.5, color=MUTED)
        for r in pts.itertuples():
            c = ORANGE if r.run == HIGHLIGHT else BLUE
            ax.scatter(r.asr_lg, r.mmlu, s=44, color=c, edgecolors=SURFACE, linewidths=2, zorder=3)
        ax.set_title(a, fontsize=9)
        ax.set_xlim(-0.03, 1.03)
        ax.set_ylim(0, 1)
        to_label.append((ax, [(r.run, r.asr_lg, r.mmlu) for r in pts.itertuples()]))
        ax.grid(color=GRID, lw=0.6)
        ax.set_axisbelow(True)
        ax.xaxis.set_major_formatter(lambda x, _: f"{x:.0%}")
        ax.yaxis.set_major_formatter(lambda x, _: f"{x:.0%}")
    for ax in axes[-1]:
        ax.set_xlabel("ASR (Llama Guard) - lower is safer")
    for ax in axes[:, 0]:
        ax.set_ylabel("MMLU accuracy")
    fig.suptitle(f"Capability vs attack success, per attack ({_stage_note(stage)}). Good = top-left.",
                 x=0.01, ha="left", fontsize=9.5, fontweight="bold", color=INK)
    fig.tight_layout()
    for ax, pts in to_label:  # after the layout is final, so overlap checks use the real geometry
        _place_labels(fig, ax, pts)
    return _save(fig, f"capability_vs_asr_{stage}", out)


def fig_agreement(agree, out: Path = FIG_DIR, human=None) -> list[Path]:
    """(d) judge-agreement table as a figure: judge vs judge (row level), then judge vs human (hand labels)."""
    if not len(agree) and (human is None or not len(human)):
        return []
    plt = _mpl()
    cols = ["labels compared", "n", "agreement", "Cohen's κ", "rate (first)", "rate (second)"]
    rows = [[r.pair, f"{r.n:,}", pct(r.agree, 1), f"{r.kappa:.2f}" if r.kappa == r.kappa else "-",
             pct(r.rate_a, 1), pct(r.rate_b, 1)] for r in agree.itertuples()]
    if human is not None and len(human):  # human rows: accuracy reweighted to the judged population
        rows += [[r.pair + " (hand labels)", f"{r.n:,}", pct(r.weighted_accuracy, 1),
                  f"{r.kappa:.2f}" if r.kappa == r.kappa else "-", pct(r.judge_rate, 1), pct(r.human_rate, 1)]
                 for r in human.itertuples()]
    fig, ax = plt.subplots(figsize=(8.6, 0.3 * (len(rows) + 1) + 0.45))
    ax.axis("off")
    tab = ax.table(cellText=rows, colLabels=cols, bbox=[0, 0, 1, 1], cellLoc="right", colLoc="right",
                   colWidths=[0.42, 0.1, 0.12, 0.1, 0.13, 0.13])
    tab.auto_set_font_size(False)
    tab.set_fontsize(8)
    for (i, j), c in tab.get_celld().items():
        c.set_edgecolor(GRID)
        c.visible_edges = "B"
        c.set_facecolor(SURFACE)
        if j == 0:  # text column reads left-aligned; numbers stay right-aligned
            c._loc = "left"
            c.get_text().set_horizontalalignment("left")
        if i == 0:
            c.set_text_props(color=INK2, fontweight="bold")
    ax.set_title("Judge agreement, row level (each best-of-N variant counts once); hand-label rows: "
                 "accuracy reweighted to the judged population", fontsize=9.5)
    return _save(fig, "judge_agreement", out)


def make_figures(t: dict, out: Path = FIG_DIR) -> list[Path]:
    paths = []
    stages = [s for s in ("test", "dev") if s in set(t["ci"]["stage"])] if len(t["ci"]) else []
    for stage in stages:
        paths += fig_heatmap(t["ci"], stage, out)
        paths += fig_retention(t["retention"], stage, out)
        paths += fig_capability_vs_asr(t["cells"], stage, out)
    paths += fig_agreement(t["agreement"], out, t.get("human_agreement"))
    return paths


# ----------------------------------------------------------------------------- CLI

def pull_judged(config: str = "configs/judge.yaml", judged_dir: Path | None = None) -> int:
    import yaml

    from aftermarket.generate import pull_from_hub

    cfg = yaml.safe_load((REPO_ROOT / config).read_text(encoding="utf-8"))
    out = judged_dir or REPO_ROOT / cfg["out_dir"]
    out.mkdir(parents=True, exist_ok=True)
    return pull_from_hub(cfg, out, cfg["push"]["path_in_repo"])


def run(judged_dir: Path, fill: bool = True, fig_dir: Path = FIG_DIR, table_dir: Path = TABLE_DIR,
        docs=DOCS, labels_path: Path = HAND_LABELS, sample_path: Path = HAND_SAMPLE) -> dict:
    from aftermarket.label import current_labels, read_jsonl

    df = load_judged(judged_dir)
    n_all = len(df)
    df = leaderboard_runs(df) if len(df) else df
    print(f"{n_all:,} judged rows ({n_all - len(df):,} from AM dev candidates left out)", flush=True)
    labels = current_labels(labels_path)
    t = collect(df, labels, read_jsonl(sample_path))
    if labels:
        print(f"{len(labels)} hand labels joined", flush=True)
    blocks = write_tables(t, table_dir)
    figs = make_figures(t, fig_dir)
    values = headline_values(t, len(df))
    changed = fill_docs(blocks, values, docs) if fill else []
    print(f"tables -> {table_dir} | {len(figs)} figure files -> {fig_dir} | docs updated: "
          f"{[p.name for p in changed] or 'none'}", flush=True)
    missing = sorted(k for k, v in values.items() if v == TBD)
    if missing:
        print(f"values still TBD (no data yet): {missing}", flush=True)
    return {"tables": t, "figures": figs, "values": values, "docs": changed}


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="final tables, figures and README/ABSTRACT numbers")
    ap.add_argument("--from-hub", action="store_true", help="pull judged files from the private Hub repo first")
    ap.add_argument("--config", default="configs/judge.yaml", help="judge config naming the Hub repo + judged dir")
    ap.add_argument("--no-docs", action="store_true", help="don't rewrite README.md / ABSTRACT.md placeholders")
    args = ap.parse_args(argv)
    import yaml

    cfg = yaml.safe_load((REPO_ROOT / args.config).read_text(encoding="utf-8"))
    judged = REPO_ROOT / cfg["out_dir"]
    if args.from_hub:
        pull_judged(args.config, judged)
    run(judged, fill=not args.no_docs)


if __name__ == "__main__":
    main()
