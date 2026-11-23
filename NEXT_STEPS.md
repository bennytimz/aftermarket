# Next steps (Stage 2), updated 23 Nov 2026

Deadline: Tue 24 Nov AoE (Wed 25 Nov 12:59 Lagos). Target: paper finished Tue 24 Nov night (Lagos).
Plan: `STAGE2_PLAN.md`. Page budget: `paper/PAGE_BUDGET.md`. **`stage2` was merged into main (fast-forward, 5fee697,
23 Nov). Commit further fixes on main directly.**

## Where we are
- **All experiments done.** 60 cells (12 runs x A0/A2/A3/A4q3/A1) judged; E5 seed check; E2 throughput on T4, L4
  and A100; E7 full audit of GC on one T4 (rerun job 6b03727e130a00b5aa8163cd: 1.1 h incl. setup, $0.68, the
  price of serving 7M tokens on an L4; its GC retention 0.74 matches the leaderboard's 0.74). Yoruba human checks
  committed.
- E7's first run (6b031f04...) finished every attack and judge, then crashed drawing the report card: matplotlib
  was missing from `requirements-stage2.txt`. Fixed there and in `pyproject.toml` (`dependencies`), and
  `paper/fetch_stage2.py` now downloads `stage2/audit/**` (it never did). The first run is not used.
- **Paper: complete.** `check_tex` passes (no TODOs, no eaten escapes, every `\V{key}` present). Main text
  5.65 / 6 pages, 13 pages total, appendix pp. 9-13 (about 4.5 pages), no overfull boxes. "+38% faster" is now
  "38% faster" (new unsigned key `cost.<gpu>.<run>.gain`).
- **PDF:** `docs/aftermarket_stage2.pdf` (copy of `paper/main.pdf`, which stays gitignored). Re-copy it after any
  rebuild.
- **README:** "Stage 2 results" section (findings, Figs 1, 3, 5 of the paper, cost table, audit cost) between the
  `<!-- stage2:start/end -->` markers, written by `python paper/make_readme.py` from the paper's `numbers.json` and
  `tables/cost.tex`. Do not edit that block by hand: edit `paper/readme_stage2.md` and rerun the script.
- **Space:** "Stage 2 results" tab (same text, the three figures), uploaded with `space_push --push`.

## Rebuild checklist (after any change to results or text)
From the repo root (`make` is not installed):
1. `python paper/fetch_stage2.py`
2. `PYTHONPATH="src;paper" python paper/make_assets.py`
3. `cd paper && python check_tex.py` -> "no eaten escapes, no TODOs"
4. `tectonic -X compile main.tex --keep-logs -Z search-path=template` (close main.pdf in any viewer first)
5. `python check_pages.py` (main text <= 6.0; appendix <= 6 pages after the references)
6. `cd .. && cp paper/main.pdf docs/aftermarket_stage2.pdf && python paper/make_readme.py`
7. `PYTHONPATH=src python -m aftermarket.space_push` (dry run), then `--push`

## Merge (done, 23 Nov)
- Fast-forward of `stage2` into main (`5fee697`), pushed. The README on main shows the Stage 2 section; the Space
  was re-uploaded and its paper link (`blob/main/docs/aftermarket_stage2.pdf`) opens.
- Ethics statement, appendix and README now say the same: machine translations of the public benchmark prompts
  (HarmBench/AdvBench/XSTest) are released for reproducibility, as MultiJail does; no completions to harmful prompts
  and no attacked weights.
- After any rebuild: commit on main, push, and re-upload the Space (step 7 above) if the README section changed.

## Optional, if time
- Kaggle quick audit (`notebooks/11_audit_demo.ipynb`) for a Kaggle T4 timing of the quick mode.
- `paper/sections/related.tex` and the intro were not reworked in the last review; one read-through.
- Local test env lacks `accelerate`: `tests/test_defense.py::test_training_kwargs_have_no_warmup_ratio` fails
  locally only (unrelated to Stage 2).
