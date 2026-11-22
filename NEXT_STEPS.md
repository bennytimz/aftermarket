# Next steps (Stage 2), updated 22 Nov 2026, ~17:40 UTC

Deadline: Tue 24 Nov AoE (Wed 25 Nov 12:59 Lagos). Target: paper finished Tue 24 Nov night (Lagos).
Plan: `STAGE2_PLAN.md`. Page budget: `paper/PAGE_BUDGET.md`. Branch `stage2` only; nothing merged to main.

## Where we are
- **All experiments done except two runs** (below): 60 cells (12 runs x A0/A2/A3/A4q3/A1), all judged; E5 seed
  check (A2 seeds 1, 2 for BASE/GC/AM) judged; E2 throughput on L4 and A100; Yoruba human checks (translation
  60/60, judge verdicts 25/25) committed.
- **Paper: full draft.** Title "Aftermarket: Most Safety Fixes Break After Release, and the Tools That Measure Them
  Can Mislead". Trial build (pending numbers as placeholders): main text 5.70 / 6 pages, 13 pages total, appendix
  about 4.5 pages (second half of p. 9 to p. 13), no overfull boxes. All review fixes are in (commit 2b793b9 and later).
- `make pdf` now runs `paper/check_tex.py` first: it fails on any TODO, eaten escape (\r \t \f \b \a) or `\V{key}`
  missing from `derived/numbers.json`. **It fails right now only on the two pending runs' numbers** (by design).

## Running on HF Jobs (Aftermarket-lexsi), nothing to do but wait
| Run | Job id | Fills | Expected done (UTC) |
|---|---|---|---|
| E2 throughput on T4 (`s2-e2-t4-small`) | 6b02f9978d268c0fbd21ee81 | T4 column of Table 1, `cost.price.T4` in its caption | ~19:40 |
| E7 full audit of GC, timed (`s2-e7-audit-gc-full`, t4-medium, 6 h timeout) | 6b031f048d268c0fbd222c5e | `audit.full.hours/usd/mtok` in 4.7 and Takeaways | ~21:00 |

E7 pushes only `timing.json` + the report card to `stage2/audit/audit-gc-full/` (no completions). If it fails, its
log is `huggingface_hub.fetch_job_logs(job_id=..., namespace="Aftermarket-lexsi")`; the first launch failed on
`git` missing in the image and was fixed in `audit._git_sha` (commit after 42b3b25).

## When back
From the repo root (PowerShell or Git Bash; `make` is not installed, so run the steps directly):
1. Check both jobs: `PYTHONPATH=src python -m aftermarket.stage2.jobs status` (look for the two ids above: COMPLETED).
2. Fetch: `python paper/fetch_stage2.py`
3. Rebuild numbers, tables, figures: `PYTHONPATH="src;paper" python paper/make_assets.py`
4. Check: `cd paper && python check_tex.py` -> must print "no eaten escapes, no TODOs".
5. Compile: `tectonic -X compile main.tex --keep-logs -Z search-path=template` (close main.pdf in any viewer
   first: Windows locks it).
6. Page counts: `python check_pages.py` (main text must be <= 6.0; cut order in PAGE_BUDGET.md above 5.7) and
   check the appendix is <= 6 pages (it starts after the references).
7. Read 4.7 and Takeaways once with the real audit numbers; commit and push.

If E7 failed or is too late: replace the audit clause in 4.7 / Takeaways with the measured E2 numbers only (ask
the user first), so check_tex passes.

## Optional, if time
- Kaggle quick audit (`notebooks/11_audit_demo.ipynb`, unchanged) for a Kaggle T4 timing of the quick mode.
- `paper/sections/related.tex` and the intro were not reworked in the last review; one read-through.
