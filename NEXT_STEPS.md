# Next steps (Stage 2), updated 24 Nov 2026

Deadline: Tue 24 Nov AoE (Wed 25 Nov 12:59 Lagos). Target: paper finished Tue 24 Nov night (Lagos).
Plan: `STAGE2_PLAN.md`. Page budget: `paper/PAGE_BUDGET.md`. **`stage2` was merged into main (fast-forward, 5fee697,
23 Nov). Commit further fixes on main directly.**

## Now: re-upload the Overleaf zip, then submit
The organizer's feedback is applied (24 Nov): steps 1-5 and 7 below are done. Left: step 6 (re-upload
`paper/aftermarket_overleaf.zip`, replace all files, compile with pdfLaTeX), 8 (submit), 9 (release).
- Header: one centred icon-link line under the email (fontawesome5: Code, Leaderboard, Models, Datasets, Video);
  the old link footnote is gone.
- Lexsi kits cited from their official BibTeX (`paper/kits.bib`, hand-kept; `custom.bib` stays generated):
  CuratorKIT (Setup, App. B Data), SafeTune (Setup A5bon, §4.5 SafeGrad trainer, App. B BoN), AuditKIT (Setup,
  App. B: values computed by the identical local copy, now tested against AuditKIT in `tests/test_stats.py`),
  CircuitKIT (Limitations: attempted, OOM, used in no result).
- Main text 5.76 / 6 pages; 14 pages; the cost table moved earlier in the appendix source so it stays on p. 14.
- Final fixes (24 Nov): Fig. 1 is defined after the intro's first paragraph so it sits at the top of the right
  column, below the abstract (the taller header had pushed it above the abstract). CuratorKIT is cited once (arXiv
  paper only); SafeTune's note field dropped; AuditKIT/CircuitKIT carry "Software, v1.1.11 / v0.2.3" + GitHub URL;
  §4.5 has one merged citation. Tectonic: 5.76 / 6, 14 pp; Overleaf zip under pdflatex: 5.65 / 6, 14 pp, clean.
- The zip was compiled with pdflatex + bibtex in Docker `texlive/texlive:latest-small` (+ tlmgr fontawesome5
  courier helvetic ...): 14 pages, 0 errors, 0 BibTeX warnings.

Original checklist (from the repo root; `make` is not installed on Windows, the Python command after each target
does the same thing):
1. (done) `git apply paper/table2_am_public.patch`: Table 2's AM row points to the public
   `bennytimz/aftermarket-AM-frozen` @ `c014cf5` (same weights as private `aftermarket-am` @ `d11af3a`, checked by
   file hash), licence "cc-by-nc-4.0 / public". Changes `paper/appendix_tables.py` and `paper/tables/models.tex`;
   tested: compiles cleanly with pdflatex, no overfull box, 14 pages. Delete the patch file once applied.
2. The organizer's changes.
3. `make pdf` (= steps 3-5 of the rebuild checklist below): `check_tex`, tectonic, `check_pages` (everything before References <= 6.0, appendix <= 6.0).
4. `cp paper/main.pdf docs/aftermarket_stage2.pdf` (the release attaches this exact file, so it must be what you
   submit). Rerun `python paper/make_readme.py` and the Space push (checklist steps 6-7) if the README block changed.
5. `make overleaf` (`cd paper && python make_overleaf.py`) -> `paper/aftermarket_overleaf.zip` (untracked).
6. Re-upload the zip to the Overleaf project (replace all files) and check it compiles there with pdfLaTeX.
7. Commit and push on main.
8. Submit.
9. Right after submitting, run the release script on the submitted commit:
   `bash docs/release/create_v2.0-stage2.sh --dry-run`, check the commit and PDF sha256, then rerun without
   `--dry-run` (creates tag + release `v2.0-stage2` with the PDF attached).

## Done 24 Nov
- Overleaf zip: `make overleaf`; tested clean with pdflatex + bibtex (TeX Live 2026): 14 pages, no errors.
- Links: all 57 URLs in/around the PDF open logged out (HTTP 200), except `bennytimz/aftermarket-am` (401, private
  by design; step 1 above replaces it).
- HF: "Paper / Video / Code" section on the 5 public cards; both collection descriptions carry the three links
  (150-char limit), with the author credit in a note on the Space item.
- GitHub: About description, topics and the Space as website set.

## Where we are
- **All experiments done.** 60 cells (12 runs x A0/A2/A3/A4q3/A1) judged; E5 seed check; E2 throughput on T4, L4
  and A100; E7 full audit of GC on one T4 (rerun job 6b03727e130a00b5aa8163cd: 1.1 h incl. setup, $0.68, the
  price of serving 7M tokens on an L4; its GC retention 0.74 matches the leaderboard's 0.74). Yoruba human checks
  committed.
- E7's first run (6b031f04...) finished every attack and judge, then crashed drawing the report card: matplotlib
  was missing from `requirements-stage2.txt`. Fixed there and in `pyproject.toml` (`dependencies`), and
  `paper/fetch_stage2.py` now downloads `stage2/audit/**` (it never did). The first run is not used.
- **Paper: complete.** `check_tex` passes (no TODOs, no eaten escapes, every `\V{key}` present). Main text
  5.65 / 6 pages, 14 pages total, appendix pp. 9-14, no overfull boxes. "+38% faster" is now
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
5. `python check_pages.py` (everything before References, incl. Limitations and Ethics, <= 6.0; appendix <= 6.0)
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
