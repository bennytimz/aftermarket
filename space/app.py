"""HF Space: Aftermarket interactive leaderboard (PLAN.md §11). CPU only.

Reads ONLY the aggregate metrics CSVs in ./data (copied there by `python -m aftermarket.space_push`): per-cell
values with 95% CIs, retention, and judge agreement. No prompts, no completions, no weights (PLAN.md §9).

Tabs: Leaderboard (heatmap + table, filter by metric / attack / model), Retention ranking, Judges, About.
"""

from __future__ import annotations

from pathlib import Path

DATA = Path(__file__).resolve().parent / "data"
ATTACK_ORDER = ["A0", "A1", "A2", "A3", "A4q4", "A4q3", "A4q2", "A5raw", "A5bon"]
RUN_ORDER = ["BASE", "SDT", "GC", "IL", "RL", "RL_noguard", "ST", "SP", "AM", "AM_rand0", "AM_rand1", "AM_all"]
METRICS = {  # label -> (metric id, lower is better)
    "Attack success, Llama Guard (primary)": ("asr_lg", True),
    "Attack success, WildGuard (second judge)": ("asr_wg", True),
    "Over-refusal on XSTest (WildGuard)": ("over_refusal_wg", True),
    "MMLU accuracy": ("mmlu", False),
}
ATTACK_NAMES = {"A0": "as released", "A1": "benign fine-tune", "A2": "harmful fine-tune", "A3": "abliteration",
                "A4q4": "4-bit", "A4q3": "3-bit", "A4q2": "2-bit", "A5raw": "template removed",
                "A5bon": "best-of-8 perturbations"}
FILES = {"ci": "cell_ci.csv", "retention": "retention.csv", "retention_pairwise": "retention_pairwise.csv",
         "agreement": "agreement.csv", "human_agreement": "human_agreement.csv"}

SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
BLUE, ORANGE = "#2a78d6", "#eb6834"
SEQ_RAMP = ["#f0efec", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]


def _rank(v: str, order: list[str]):
    return (order.index(v), "") if v in order else (len(order), v)


def ordered(values, order):
    return sorted(values, key=lambda v: _rank(v, order))


def load_data(data_dir: Path = DATA) -> dict:
    """Every metrics CSV that exists (missing ones are empty frames: the app shows 'no results yet')."""
    import pandas as pd

    out = {}
    for key, name in FILES.items():
        p = Path(data_dir) / name
        out[key] = pd.read_csv(p) if p.exists() else pd.DataFrame()
    return out


def fmt_ci(v, lo, hi) -> str:
    if v != v:
        return "-"
    return f"{v:.0%} [{lo:.0%}–{hi:.0%}]" if lo == lo else f"{v:.0%}"


def select(ci, metric: str, runs=None, attacks=None):
    """Rows of one metric, filtered to the chosen runs / attacks (None or empty = all)."""
    if ci is None or not len(ci):
        return ci
    sub = ci[ci["metric"] == metric]
    if runs:
        sub = sub[sub["run"].isin(runs)]
    if attacks:
        sub = sub[sub["attack"].isin(attacks)]
    return sub


def cell_table(ci, metric: str, runs=None, attacks=None):
    """Wide table: model x attack, 'value [lo–hi]'."""
    import pandas as pd

    sub = select(ci, metric, runs, attacks)
    if sub is None or not len(sub):
        return pd.DataFrame({"": ["No results for this selection yet."]})
    rs, ats = ordered(sub["run"].unique(), RUN_ORDER), ordered(sub["attack"].unique(), ATTACK_ORDER)
    cell = {(r.run, r.attack): fmt_ci(r.value, r.lo, r.hi) for r in sub.itertuples()}
    t = pd.DataFrame([[r] + [cell.get((r, a), "-") for a in ats] for r in rs], columns=["model"] + ats)
    return t


def _mpl():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.family": "sans-serif", "font.size": 9, "text.color": INK, "axes.labelcolor": INK2,
                         "xtick.color": INK2, "ytick.color": INK2, "figure.facecolor": SURFACE,
                         "axes.facecolor": SURFACE, "axes.edgecolor": AXIS, "axes.spines.top": False,
                         "axes.spines.right": False})
    return plt


def _empty(plt, msg: str):
    fig, ax = plt.subplots(figsize=(7, 1.2))
    ax.axis("off")
    ax.text(0.5, 0.5, msg, ha="center", va="center", color=INK2)
    return fig


def heatmap(ci, metric: str, lower_better: bool, runs=None, attacks=None):
    """Model x attack heatmap of one metric, 0-100% fixed scale, value + CI in each cell."""
    import numpy as np
    from matplotlib.colors import LinearSegmentedColormap

    plt = _mpl()
    sub = select(ci, metric, runs, attacks)
    if sub is None or not len(sub):
        return _empty(plt, "No results for this selection yet.")
    rs, ats = ordered(sub["run"].unique(), RUN_ORDER), ordered(sub["attack"].unique(), ATTACK_ORDER)
    cell = {(r.run, r.attack): r for r in sub.itertuples()}
    z = np.array([[cell[(r, a)].value if (r, a) in cell else np.nan for a in ats] for r in rs])
    fig, ax = plt.subplots(figsize=(1.0 * len(ats) + 2.2, 0.5 * len(rs) + 1.4))
    cmap = LinearSegmentedColormap.from_list("seq", SEQ_RAMP).with_extremes(bad=SURFACE)
    im = ax.pcolormesh(np.ma.masked_invalid(z), cmap=cmap, vmin=0, vmax=1, edgecolors=SURFACE, linewidth=2)
    for i, r in enumerate(rs):
        for j, a in enumerate(ats):
            c = cell.get((r, a))
            if c is None:
                ax.text(j + .5, i + .5, "–", ha="center", va="center", color=MUTED, fontsize=8)
                continue
            col = "#ffffff" if c.value > 0.55 else INK
            ax.text(j + .5, i + .42, f"{c.value:.0%}", ha="center", va="center", color=col, fontsize=8.5,
                    fontweight="bold")
            if c.lo == c.lo:
                ax.text(j + .5, i + .72, f"{c.lo:.0%}–{c.hi:.0%}", ha="center", va="center", color=col, fontsize=6.5)
    ax.set_xticks(np.arange(len(ats)) + .5, ats)
    ax.set_yticks(np.arange(len(rs)) + .5, rs)
    ax.invert_yaxis()
    ax.xaxis.tick_top()
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02, format=lambda x, _: f"{x:.0%}")
    cb.outline.set_visible(False)
    cb.set_label("lower is better" if lower_better else "higher is better", color=INK2)
    fig.tight_layout()
    return fig


def retention_figure(ret):
    plt = _mpl()
    if ret is None or not len(ret):
        return _empty(plt, "No retention scores yet.")
    sub = ret.iloc[::-1]
    fig, ax = plt.subplots(figsize=(7.2, 0.42 * len(sub) + 1.2))
    for y, r in enumerate(sub.itertuples()):
        ax.barh(y, r.retention, height=.62, color=ORANGE if r.run == "AM" else BLUE, alpha=.5 if r.partial else 1)
        if r.lo == r.lo:
            ax.plot([r.lo, r.hi], [y, y], color=INK2, lw=1)
        ax.text(max(r.retention, r.hi if r.hi == r.hi else r.retention) + .015, y, f"{r.retention:.2f}",
                va="center", fontsize=8)
    ax.set_yticks(range(len(sub)), [r + (" (partial battery)" if p else "") for r, p in zip(sub["run"], sub["partial"])])
    ax.axvline(1.0, color=AXIS, lw=.8)
    ax.grid(axis="x", color=GRID, lw=.6)
    ax.set_axisbelow(True)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.set_xlabel("retention: 1 − mean rise in attack success over A1–A5 (1 = nothing lost)")
    fig.tight_layout()
    return fig


def retention_table(ret, rpw):
    import pandas as pd

    if ret is None or not len(ret):
        return pd.DataFrame({"": ["No retention scores yet."]})
    vs = rpw[rpw["vs"] == "BASE"].set_index("run") if rpw is not None and len(rpw) else pd.DataFrame()
    rows = []
    for k, r in enumerate(ret.itertuples(), 1):
        d = f"{vs.loc[r.run].delta:+.2f} (p_holm {vs.loc[r.run].p_holm:.3f})" if r.run in vs.index else ""
        rows.append({"rank": k, "model": r.run, "retention [95% CI]": f"{r.retention:.2f} [{r.lo:.2f}–{r.hi:.2f}]",
                     "vs BASE": d, "attack families": r.families + (" (partial)" if r.partial else ""),
                     "prompts": r.n})
    return pd.DataFrame(rows)


def judges_table(agree, human):
    import pandas as pd

    rows = []
    if agree is not None and len(agree):
        rows += [{"labels compared": r.pair, "n": r.n, "agreement": f"{r.agree:.1%}",
                  "Cohen's kappa": f"{r.kappa:.2f}" if r.kappa == r.kappa else "-"} for r in agree.itertuples()]
    if human is not None and len(human):
        rows += [{"labels compared": r.pair + " (hand labels; accuracy reweighted)", "n": r.n,
                  "agreement": f"{r.weighted_accuracy:.1%}",
                  "Cohen's kappa": f"{r.kappa:.2f}" if r.kappa == r.kappa else "-"} for r in human.itertuples()]
    return pd.DataFrame(rows) if rows else pd.DataFrame({"": ["No judge results yet."]})


ABOUT = """
### Aftermarket: does a safety fix survive the next person who downloads the model?

Public safety fixes for **Tiny Aya Global** from the Lexsi AI Safety & Alignment Hackathon, each run through the
same post-release battery: benign fine-tune (A1), harmful fine-tune (A2), abliteration (A3), HQQ 4/3/2-bit
compression (A4), chat template removed (A5raw), best-of-8 perturbations (A5bon). Same prompts, seed, settings
and judges for every model; 95% bootstrap intervals over prompts. Held-out test prompts only.

Every team is credited in the [write-up](https://github.com/bennytimz/aftermarket), and results are framed as
robustness *after release*, not as verdicts on any fix. This Space shows aggregate metrics only: no prompts,
no model outputs and no attack-modified weights are published.
"""


def build_ui(data: dict | None = None):
    import gradio as gr

    d = data if data is not None else load_data()
    ci = d["ci"]
    runs = ordered(ci["run"].unique(), RUN_ORDER) if len(ci) else []
    attacks = ordered(ci["attack"].unique(), ATTACK_ORDER) if len(ci) else []
    attack_choices = [(f"{a} · {ATTACK_NAMES.get(a, a)}", a) for a in attacks]
    first = next(iter(METRICS))

    def update(metric_label, sel_attacks, sel_runs):
        metric, lower = METRICS[metric_label]
        return (heatmap(ci, metric, lower, sel_runs or None, sel_attacks or None),
                cell_table(ci, metric, sel_runs or None, sel_attacks or None))

    with gr.Blocks(title="Aftermarket leaderboard") as demo:
        gr.Markdown("# Aftermarket leaderboard\nPost-release robustness of safety-fixed Tiny Aya models.")
        with gr.Tab("Leaderboard"):
            with gr.Row():  # one filter row above everything it scopes
                metric = gr.Dropdown(list(METRICS), value=first, label="Metric")
                sel_attacks = gr.CheckboxGroup(attack_choices, value=attacks, label="Attacks")
                sel_runs = gr.CheckboxGroup(runs, value=runs, label="Models")
            plot = gr.Plot(label="Model × attack")
            table = gr.Dataframe(label="Values with 95% CI", interactive=False)
            for c in (metric, sel_attacks, sel_runs):
                c.change(update, [metric, sel_attacks, sel_runs], [plot, table])
            demo.load(update, [metric, sel_attacks, sel_runs], [plot, table])
        with gr.Tab("Retention ranking"):
            gr.Plot(value=retention_figure(d["retention"]), label="Retention, 95% CI")
            gr.Dataframe(value=retention_table(d["retention"], d["retention_pairwise"]), interactive=False)
        with gr.Tab("Judges"):
            gr.Markdown("Llama-Guard-3-1B (primary) and WildGuard on every harmful-prompt response, plus 100 blind "
                        "hand labels (oversampled where the judges disagree; accuracy reweighted to all responses).")
            gr.Dataframe(value=judges_table(d["agreement"], d["human_agreement"]), interactive=False)
        with gr.Tab("About"):
            gr.Markdown(ABOUT)
    return demo


if __name__ == "__main__":
    build_ui().launch()
