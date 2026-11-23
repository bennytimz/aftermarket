# Next steps (Stage 2), updated 23 Nov 2026, ~01:00 UTC

Deadline: Tue 24 Nov AoE (Wed 25 Nov 12:59 Lagos). Target: paper finished Tue 24 Nov night (Lagos).
Plan: `STAGE2_PLAN.md`. Page budget: `paper/PAGE_BUDGET.md`. Branch `stage2`; **not merged to main yet** (waiting for
the user's go, see "Merge" below).

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

## Merge (prepared, not done: wait for the user)
- `stage2` is 39 commits ahead of `origin/main`, main has nothing stage2 lacks: a fast-forward, no conflicts.
- What it changes on main: 112 files, about 179k lines added, 8 removed. Mostly new: `paper/` (sources, template,
  derived numbers incl. the 155k-line `capability_items.csv`, tables, figures, `docs/aftermarket_stage2.pdf`),
  `src/aftermarket/stage2/` (jobs, cells, judging, throughput, audit job, Yoruba check), `configs/stage2/` and
  `configs/gpu_prices.yaml`, `data/stage2/translations.jsonl` and the Yoruba check files (prompts, ids, and replies
  to XSTest prompts only: no harmful completions), `notebooks/12_stage2_kaggle.ipynb`, `results/stage2/` (job
  log, overlap check), tests. Modified: README (Stage 2 section), Space page/app/style and `space_push.py`,
  `pyproject.toml` and `requirements-stage2.txt` (matplotlib), `.gitignore`.
- Links already point at main: README links are relative; the Space's paper link (`PAPER_URL` in
  `src/aftermarket/space_push.py`) points at `blob/main/docs/aftermarket_stage2.pdf`, so it 404s until the merge.
  The live Space was uploaded with the `blob/stage2` link and keeps working until then.
- To merge when told: `git checkout main && git merge --ff-only stage2 && git push origin main`, then
  `PYTHONPATH=src python -m aftermarket.space_push --push` (re-uploads the Space with the main link).

## Optional, if time
- Kaggle quick audit (`notebooks/11_audit_demo.ipynb`) for a Kaggle T4 timing of the quick mode.
- `paper/sections/related.tex` and the intro were not reworked in the last review; one read-through.
- Local test env lacks `accelerate`: `tests/test_defense.py::test_training_kwargs_have_no_warmup_ratio` fails
  locally only (unrelated to Stage 2).
