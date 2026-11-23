"""Bundle the public metrics CSVs into space/data/ and (with --push) upload the leaderboard Space.

    PYTHONPATH=src python -m aftermarket.report --from-hub          # first: results/tables/*.csv
    PYTHONPATH=src python -m aftermarket.space_push                 # dry run: build + check the bundle, upload nothing
    PYTHONPATH=src python -m aftermarket.space_push --push          # create/update the PUBLIC static Space bennytimz/aftermarket

What can leave the machine is an allowlist: five aggregate CSVs, only the listed columns, only test-stage rows,
no AM dev candidates (AMc_*). Any column that looks like text (prompt, response, raw judge output, item ids)
stops the push. Report cards (results/cards/<run>.png + .md, aggregates only) are copied to space/cards/ for the
leaderboard runs only, with cards/index.json listing them. The Stage 2 tab (stage2/) is the README's "Stage 2
results" section (paper/make_readme.py, numbers from the paper's numbers.json) plus three paper figures, aggregates
only, and nothing else is allowed in that folder. The Space is static (sdk: static, no server, free
hosting): space/index.html + app.js + style.css + README.md + data/*.csv + cards/*.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
TABLE_DIR, SPACE_DIR = REPO_ROOT / "results" / "tables", REPO_ROOT / "space"
DEFAULT_REPO = "bennytimz/aftermarket"
ALLOW = {  # file -> columns that may be published
    "cell_ci.csv": ["stage", "run", "attack", "metric", "value", "lo", "hi", "n"],
    "retention.csv": ["stage", "run", "metric", "retention", "lo", "hi", "n", "families", "attacks", "partial",
                      "ranked"],
    "retention_pairwise.csv": ["stage", "metric", "attack", "run", "vs", "n", "baseline", "candidate", "delta",
                               "p_value", "p_holm", "significant_holm"],
    "agreement.csv": ["pair", "scope", "n", "agree", "kappa", "rate_a", "rate_b", "a_only", "b_only"],
    "human_agreement.csv": ["pair", "scope", "n", "accuracy", "kappa", "weighted_accuracy", "judge_rate", "human_rate"],
}
REQUIRED = "cell_ci.csv"
FORBIDDEN = ("prompt", "response", "model_input", "raw", "item", "label_id", "completion", "text")
SPACE_FILES = ["index.html", "app.js", "style.css", "README.md"]
CARD_INDEX = "index.json"
CARD_DIR = REPO_ROOT / "results" / "cards"
PAPER_DIR = REPO_ROOT / "paper"
STAGE2_FIGS = ["fig1_heatmap.png", "fig3_capability.png", "fig6_multilingual.png"]  # paper Figs 2, 4, 6
STAGE2_FILES = {"stage2.md", *STAGE2_FIGS}
PAPER_URL = "https://github.com/bennytimz/aftermarket/blob/main/docs/aftermarket_stage2.pdf"
VIDEO_URL = "https://youtu.be/dY9WLoC72Dk"
VIDEO_LINE = f"**[▶ Watch the 60-second video]({VIDEO_URL})**"


class UnsafeBundle(ValueError):
    pass


def build_bundle(table_dir: Path = TABLE_DIR, out_dir: Path = SPACE_DIR / "data") -> dict[str, int]:
    """Copy allowlisted columns of allowlisted CSVs, test stage only, AMc_* dropped. Returns rows per file."""
    import pandas as pd

    from aftermarket.evaluate import DEV_CANDIDATE_PREFIX

    if not (Path(table_dir) / REQUIRED).exists():
        raise FileNotFoundError(f"{table_dir / REQUIRED} missing: run `python -m aftermarket.report --from-hub` first")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for old in out_dir.glob("*.csv"):  # nothing stale or unlisted survives a rebuild
        old.unlink()
    counts = {}
    for name, cols in ALLOW.items():
        src = Path(table_dir) / name
        if not src.exists():
            continue
        df = pd.read_csv(src)
        if "stage" in df:
            df = df[df["stage"] == "test"]
        if "run" in df:
            df = df[~df["run"].astype(str).str.startswith(DEV_CANDIDATE_PREFIX)]
        df = df[[c for c in cols if c in df.columns]]
        df.to_csv(out_dir / name, index=False)
        counts[name] = len(df)
    check_bundle(out_dir)
    if not counts.get(REQUIRED):
        raise UnsafeBundle("no test-stage results in cell_ci.csv yet: nothing to publish")
    return counts


def build_cards(card_dir: Path = CARD_DIR, out_dir: Path = SPACE_DIR / "cards") -> int:
    """Copy the leaderboard runs' report cards (PNG + markdown). Returns the number of runs copied."""
    import shutil

    from aftermarket.evaluate import RUN_ORDER

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for old in out_dir.iterdir():
        old.unlink()
    runs = []
    for run in RUN_ORDER:
        src = [Path(card_dir) / f"{run}{ext}" for ext in (".png", ".md")]
        if all(p.exists() for p in src):
            for p in src:
                shutil.copy2(p, out_dir / p.name)
            runs.append(run)
    (out_dir / CARD_INDEX).write_text(json.dumps(runs), encoding="utf-8")
    check_cards(out_dir)
    return len(runs)


def check_cards(out_dir: Path) -> None:
    """Only <leaderboard run>.png / .md: never an audit's card or any other file."""
    from aftermarket.evaluate import RUN_ORDER

    allowed = {f"{r}{ext}" for r in RUN_ORDER for ext in (".png", ".md")} | {CARD_INDEX}
    bad = [p.name for p in Path(out_dir).iterdir() if p.name not in allowed]
    if bad:
        raise UnsafeBundle(f"unexpected files in the Space cards dir: {bad}")


def unwrap(md: str) -> str:
    """Join hard-wrapped lines so each paragraph, list item and note is one line (the page renders per line)."""
    out: list[str] = []
    for line in md.replace("\r", "").split("\n"):
        cont = out and out[-1].strip() and line.strip() and not out[-1].startswith(("|", "#", "![", "<!--"))
        starts = line.startswith(("|", "#", "![", "<!--")) or bool(re.match(r"\d+\.\s", line))
        if cont and not starts:
            out[-1] = out[-1].rstrip() + " " + (line[2:] if line.startswith("> ") else line).strip()
        else:
            out.append(line)
    return "\n".join(out)


def stage2_markdown(block: str) -> str:
    """README section -> Space page: video link on top, no heading or markers, figures as stage2/<file>, paper
    linked on GitHub."""
    lines = [ln for ln in block.split("\n") if not ln.startswith(("<!--", "## Stage 2 results"))]
    md = unwrap("\n".join(lines)).replace("](docs/aftermarket_stage2.pdf)", f"]({PAPER_URL})")
    md = re.sub(r"\]\(paper/figures/([\w.-]+\.png)\)", r"](stage2/\1)", md)
    return f"{VIDEO_LINE}\n\n{md.lstrip()}"


def build_stage2(paper_dir: Path = PAPER_DIR, out_dir: Path = SPACE_DIR / "stage2") -> int:
    """Write stage2/stage2.md and copy the three figures. Returns the number of files written."""
    import importlib.util
    import shutil

    spec = importlib.util.spec_from_file_location("make_readme", Path(paper_dir) / "make_readme.py")
    mr = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mr)
    numbers = json.loads((Path(paper_dir) / "derived" / "numbers.json").read_text(encoding="utf-8"))
    block = mr.render((Path(paper_dir) / "readme_stage2.md").read_text(encoding="utf-8"), numbers,
                      (Path(paper_dir) / "tables" / "cost.tex").read_text(encoding="utf-8"))
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for old in out_dir.iterdir():
        old.unlink()
    (out_dir / "stage2.md").write_text(stage2_markdown(block), encoding="utf-8")
    for f in STAGE2_FIGS:
        shutil.copy2(Path(paper_dir) / "figures" / f, out_dir / f)
    check_stage2(out_dir)
    return len(STAGE2_FILES)


def check_stage2(out_dir: Path) -> None:
    """Only stage2.md and the three listed figures; every image the page references must be one of them."""
    bad = [p.name for p in Path(out_dir).iterdir() if p.name not in STAGE2_FILES]
    if bad:
        raise UnsafeBundle(f"unexpected files in the Space stage2 dir: {bad}")
    md = (Path(out_dir) / "stage2.md").read_text(encoding="utf-8")
    refs = set(re.findall(r"!\[[^\]]*\]\(([^)]+)\)", md))
    if refs - {f"stage2/{f}" for f in STAGE2_FIGS}:
        raise UnsafeBundle(f"stage2.md references images outside the allowlist: {sorted(refs)}")


def check_bundle(out_dir: Path) -> None:
    """Refuse anything outside the allowlist: unknown files, unknown or text-like columns, dev rows, AMc_* runs."""
    import pandas as pd

    from aftermarket.evaluate import DEV_CANDIDATE_PREFIX

    for p in Path(out_dir).iterdir():
        if p.name not in ALLOW:
            raise UnsafeBundle(f"unexpected file in the Space data dir: {p.name}")
        df = pd.read_csv(p)
        bad = [c for c in df.columns if c not in ALLOW[p.name] or any(f in c.lower() for f in FORBIDDEN)]
        if bad:
            raise UnsafeBundle(f"{p.name}: columns not allowed in public data: {bad}")
        if "stage" in df and set(df["stage"]) - {"test"}:
            raise UnsafeBundle(f"{p.name}: non-test rows")
        if "run" in df and df["run"].astype(str).str.startswith(DEV_CANDIDATE_PREFIX).any():
            raise UnsafeBundle(f"{p.name}: AM dev candidates")


def push(repo_id: str = DEFAULT_REPO, space_dir: Path = SPACE_DIR) -> str:
    """Create (if needed) the PUBLIC static Space and upload page + metrics + cards. Returns the Space URL."""
    from huggingface_hub import HfApi

    check_bundle(space_dir / "data")
    if (space_dir / "cards").exists():
        check_cards(space_dir / "cards")
    if (space_dir / "stage2").exists():
        check_stage2(space_dir / "stage2")
    api = HfApi()
    api.create_repo(repo_id, repo_type="space", space_sdk="static", private=False, exist_ok=True)
    api.upload_folder(repo_id=repo_id, repo_type="space", folder_path=str(space_dir),
                      allow_patterns=SPACE_FILES + ["data/*.csv", "cards/*.png", "cards/*.md", f"cards/{CARD_INDEX}",
                                      "stage2/stage2.md"] + [f"stage2/{f}" for f in STAGE2_FIGS],
                      commit_message="Update leaderboard metrics, report cards and Stage 2 results")
    return f"https://huggingface.co/spaces/{repo_id}"


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="build the public leaderboard Space bundle; upload only with --push")
    ap.add_argument("--push", action="store_true", help="upload to the PUBLIC Space (default: dry run)")
    ap.add_argument("--repo", default=DEFAULT_REPO)
    args = ap.parse_args(argv)
    counts = build_bundle()
    print(f"bundle OK (test stage, allowlisted columns): {counts} -> {SPACE_DIR / 'data'}")
    print(f"report cards: {build_cards()} runs -> {SPACE_DIR / 'cards'}")
    print(f"stage 2 page: {build_stage2()} files -> {SPACE_DIR / 'stage2'}")
    if not args.push:
        print(f"dry run: nothing uploaded. Re-run with --push to publish https://huggingface.co/spaces/{args.repo}")
        return
    print(f"published {push(args.repo)}")


if __name__ == "__main__":
    main()
