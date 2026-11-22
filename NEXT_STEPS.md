# Next steps (Stage 2), written 22 Nov 2026, 02:46 UTC

Deadline: Tue 24 Nov AoE (Wed 25 Nov 12:59 Lagos). Target: paper finished Tue 24 Nov night (Lagos).
Plan: `STAGE2_PLAN.md`. Page budget: `paper/PAGE_BUDGET.md`. Branch `stage2` only; nothing merged to main.

## Where we are
- **Smoke test passed** on Kaggle (2x T4), report at `stage2/smoke/smoke_report.json` in the private store
  `bennytimz/aftermarket-runs`:
  - vLLM fp16 matches HF fp16 (MMLU 52.7% both, 99.3% same answers);
  - A4q3 fake-quant matches HQQ 3-bit (100% same MMLU answers; gate was 97.3%), so A4q3 cells run;
  - smoke took about 35 min (translate 3, BASE A0 cell 15, equivalence 17).
- **Translations committed**: `data/stage2/translations.jsonl` (sha256 `35299d81...`, matches the cell manifests).
- **Yoruba check file ready for review**: `data/translation_check_yo.csv` (fill adequate / fluent / notes).
- **E4 overlap check done**: one near-paraphrase (advbench:033, cosine 0.92); A2 to be re-scored without it.
- **Paper skeleton** in `paper/`: builds (`make pdf`, tectonic), 3.4 of 6 main-text pages. All numbers come from
  `paper/make_assets.py`, there are no undefined values or citations, and a 10-number spot check passed.
  Drafted: abstract, intro, related work, setup, 4.1, 4.2, 4.4, 4.5. TODO: 4.3 capability, 4.6 multilingual,
  4.7 cost, takeaways, conclusion, limitations, ethics, appendix tables.

## What is running on Kaggle (session 1, started about 01:10 UTC)
- Notebook `notebooks/12_stage2_kaggle.ipynb`, driver plan `session1`: the smoke test (done), then the launch 2 cells
  on both T4s: 12 runs x [A2, A0, A3, A4q3], in that order, with a time guard (`--max-hours 11.3`), so it stops
  before Kaggle's 12 h limit, about 12:20 UTC.
- Each finished cell uploads to `stage2/cells/<RUN>__<ATTACK>/` (manifest, lm-eval results, generations). Weights
  are deleted after each cell and are never uploaded.
- At the end it writes `stage2/kaggle/kaggle_session1_*.json` (done, skipped and failed cells).
- At 02:46 UTC no real cell had finished yet (the first A2 cells take about 50 min each).
- It does not need the PC. Shutting the PC down does not affect it.

## What to do next
1. **Morning: check the Kaggle session.** Look in the store for cell manifests and the session status file, and compare
   cell times with the T4 estimates (A2 50, A0 30, A3 38, A4q3 35 min). If cells run over, stop and decide
   before launching more.
2. **Start Kaggle session 2** when session 1 ends: re-run the same notebook (Run all). The driver's `auto` plan picks
   session 2 and skips cells already on the store. Sessions are 12 h each; we have about 30 h of quota.
3. **Review the Yoruba check file** (user).
4. **Build next**: judging jobs (NLLB back-translation, GlotLID incoherence rule, Llama Guard + WildGuard, Llama Guard
   directly on es/fr), StrongREJECT rubric evaluator (E3), E2 throughput script (T4 on Kaggle; L4/A100 on HF Jobs), and
   the quick audit notebook.
5. **HF Jobs** (org `Aftermarket-lexsi`, $85 usable): still blocked (403) until Lexsi upgrades the role. When it
   works, move A1 cells, E2 on L4/A100, E5 seeds, judging and the full audit there. Ask before each launch.
6. **Paper**: as results land, run `make assets && make pdf`. Fill 4.3, 4.6, 4.7, takeaways and conclusion within
   `paper/PAGE_BUDGET.md`. Capability, multilingual and the cost table are protected. Then Phase 3: README section,
   `Aftermarket_paper.pdf` and the Space update.
