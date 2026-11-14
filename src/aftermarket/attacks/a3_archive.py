"""Move stale A3 results (v1 layer selection) aside on the private runs dataset, so they are re-run.

    PYTHONPATH=src python -m aftermarket.attacks.a3_archive            # dry run: lists what would move
    PYTHONPATH=src python -m aftermarket.attacks.a3_archive --yes      # one Hub commit: copy to archive/, delete

A gen/<run>__A3__<split>.json manifest is stale when attack_info.select_by != abliterate.SELECT_BY (v1 results
have no select_by). For each stale cell, gen/ and judged/ .parquet + .json move to archive/A3_v1/<gen|judged>/.
Nothing is lost, and generate / judge then see the cells as missing and redo them. With --yes, stale local copies
under results/gen and results/judged are deleted too (a reused session may have pulled them).

The AM candidates' dev cells (AMc_*) are kept by default: they are the record of the frozen AM selection
(notes/limitations.md) and nothing re-runs them.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ARCHIVE = "archive/A3_v1"
KEEP_PREFIXES = ("AMc_",)


def is_stale(manifest: dict) -> bool:
    from aftermarket.attacks.abliterate import SELECT_BY

    return (manifest.get("attack_info") or {}).get("select_by") != SELECT_BY


def a3_cells(files: list[str]) -> list[str]:
    """Cell names (<run>__A3__<split>) that have a gen manifest in the repo listing."""
    return sorted(Path(f).stem for f in files if f.startswith("gen/") and f.endswith(".json") and "__A3__" in f
                  and f.count("/") == 1)


def plan_moves(stale: list[str], files: list[str], keep=KEEP_PREFIXES) -> list[tuple[str, str]]:
    """(src, dst) for every gen/judged file of a stale cell, except cells whose run starts with `keep`."""
    have = set(files)
    out = []
    for name in stale:
        if name.startswith(keep):
            continue
        for top in ("gen", "judged"):
            for ext in (".parquet", ".json"):
                src = f"{top}/{name}{ext}"
                if src in have:
                    out.append((src, f"{ARCHIVE}/{src}"))
    return out


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo", default="aftermarket-runs")
    ap.add_argument("--yes", action="store_true", help="do it (default: dry run)")
    args = ap.parse_args(argv)

    from huggingface_hub import CommitOperationAdd, CommitOperationDelete, HfApi, hf_hub_download

    from aftermarket.generate import resolve_repo_id

    repo = resolve_repo_id(args.repo)
    api = HfApi()
    files = api.list_repo_files(repo, repo_type="dataset")
    stale = []
    for name in a3_cells(files):
        m = json.loads(Path(hf_hub_download(repo, f"gen/{name}.json", repo_type="dataset")).read_text("utf-8"))
        if is_stale(m):
            stale.append(name)
    moves = plan_moves(stale, files)
    if args.yes:
        from aftermarket.data import REPO_ROOT

        stale_local = stale + [Path(f).stem for f in map(str, (REPO_ROOT / "results" / "gen").glob("*__A3__*.json"))
                               if is_stale(json.loads(Path(f).read_text("utf-8")))]
        gone = remove_local(sorted(set(stale_local)), REPO_ROOT)
        if gone:
            print(f"removed {len(gone)} stale local A3 files under results/")
    kept = [n for n in stale if n.startswith(KEEP_PREFIXES)]
    print(f"{repo}: {len(stale)} stale A3 cells, {len(moves)} files to move to {ARCHIVE}/"
          + (f" (kept as the AM selection record: {kept})" if kept else ""))
    for name in sorted({Path(s).stem for s, _ in moves}):
        print("  ", name)
    if not moves:
        return
    if not args.yes:
        print("dry run: pass --yes to move them")
        return
    ops = []
    for src, dst in moves:  # small files: download + re-add (works for LFS and non-LFS on every hub version)
        ops.append(CommitOperationAdd(dst, hf_hub_download(repo, src, repo_type="dataset")))
        ops.append(CommitOperationDelete(src))
    api.create_commit(repo, ops, repo_type="dataset",
                      commit_message=f"archive {len(stale) - len(kept)} stale A3 cells (v1 layer selection)")
    print(f"moved {len(moves)} files")


def remove_local(stale: list[str], root: Path, keep=KEEP_PREFIXES) -> list[Path]:
    """Delete local copies of stale cells (a reused session may have pulled them: generate would skip them)."""
    gone = []
    for name in stale:
        if name.startswith(keep):
            continue
        for top in ("gen", "judged"):
            for ext in (".parquet", ".json"):
                f = root / "results" / top / f"{name}{ext}"
                if f.exists():
                    f.unlink()
                    gone.append(f)
    return gone


if __name__ == "__main__":
    main()
