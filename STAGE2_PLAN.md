# Aftermarket: Stage 2 plan (paper + new experiments)

Branch `stage2`. Written 2026-11-22, 02:00 Lagos. Everything here is a **Stage 2 addition**, not part of the
pre-registration (PLAN.md §4–§6 are unchanged). Existing result files are never modified: new outputs go to
`results/stage2/` and to `bennytimz/aftermarket-runs` (private) under `stage2/`.

## Budget update (2026-11-22)

The HF credits are **$100 on the org `Aftermarket-lexsi`** (valid until 1 Dec), not $200. Keep $15 in reserve, so **$85 is usable**. Every job runs under that org's namespace; results still go to the private `bennytimz/aftermarket-runs`. The same four launches apply, each approved first. **If any launch runs over its estimate, stop and report.**

## 0. Calendar check (needs your answer)

Today is **Sunday 22 Nov 2026** (system clock 01:58 Lagos), not Friday. **24 Nov 2026 is a Tuesday**, so
"Mon Nov 24" is off by a day:
- if the deadline is **Mon 23 Nov, 23:59 AoE**, that is **Tue 24 Nov, 12:59 Lagos**;
- if it is **Tue 24 Nov, 23:59 AoE**, that is **Wed 25 Nov, 12:59 Lagos**.

Your target ("submitted Monday night") gives us about **44 hours**. The plan below fits that. It runs on HF Jobs
because many jobs can run at once, and it cuts in a fixed order if anything slips (§8).

## 1. Compute: where things run

| | HF Jobs (recommended for the main runs) | Kaggle 2×T4 (free) |
|---|---|---|
| Who runs it | I launch, monitor and collect each job from here (`hf jobs run`) | You click through each notebook |
| Parallel | Many jobs at once: 60 cells finish in hours | 1 session, 2 GPUs, 12 h cap, about 30 h a week |
| GPU | `l4x1` (24 GB, $0.80/h, billed per minute). vLLM fully supports L4 | T4: vLLM support on Turing is a risk |
| Use it for | E1, E3, E5, E6, judging, E2 on L4 and A100, the full-audit timing | E2 on T4 (adds a free T4 price point), the quick-audit proof run (notebook 11, as it is) |

Why not Kaggle for everything: 60 (run × attack) cells with QLoRA retraining and two judges come to about 40
T4-hours of sessions, each needing your clicks, inside 44 hours. That isn't practical. HF costs about $70 in
total (§7).

**CLAUDE.md note:** the "T4 only, never assume more than 16 GB" rule. L4 has 24 GB, but I cap vLLM and training
at 16 GB (`gpu_memory_utilization`, the same batch sizes), so every script still runs on a T4. The only exception
is the E2 A100 measurement, which is a price point and nothing else.

**Code delivery:** jobs and Kaggle notebooks clone the public GitHub repo, so I need to **push the `stage2`
branch to GitHub** (code only; no results, weights or completions). It won't be merged to `main` until you
approve.

## 2. Shared design for E1, E3 and E6: one "cell job" per (run × attack)

The attack weights were deleted after Stage 1 by design (PLAN.md §9), so every attacked model has to be rebuilt.
Each cell job rebuilds the weights once and runs everything that needs them:

1. **Prepare weights** (HF stack, venv 1, our existing code): load the run as released (`load.py`), then apply
   the attack:
   - A1/A2: retrain the QLoRA adapter (same `configs/attacks.yaml` block: seed 42, same data, same
     hyperparameters), merge it into fp16
   - A3: A3 v2 per-model layer selection on dev, then the weight edit
   - A4q3: see below

   The weights are saved to the job's local disk with the BASE tokenizer and the run's chat template (token-ID
   assert as always), and deleted when the job ends. They are never pushed.
2. **Evaluate** (vLLM, venv 2): `lm-eval` on the capability tasks, then vLLM generation of the multilingual,
   StrongREJECT and English prompts. Greedy decoding, max 256 new tokens, the run's own chat template, exactly as
   in Stage 1.
3. **Push** the generations (private), the lm-eval per-item samples and a manifest (git sha, config, seed,
   time, GPU, backend, timings, and for A3 the selected layer) to `aftermarket-runs/stage2/`.

**Judging** runs in separate jobs (model and judge never share a process), with `judge.py` exactly as in
Stage 1: Llama-Guard-3-1B and WildGuard at the same revisions, WildGuard 4-bit.

**Departures from Stage 1** (each stated in the paper):
- **A1/A2 are retrained.** The Stage 1 adapters no longer exist. Each A1/A2 job also regenerates the 200 English
  harmful_test prompts, and we report the new ASR next to the Stage 1 number. That doubles as a replication on
  different hardware (L4 vs T4).
- **A3 should reproduce exactly.** The layer selection is deterministic given the dev prompts. We check that the
  selected layer matches the Stage 1 manifest (BASE: layer 12, for example).
- **A4q3 in vLLM via fake quantization.** vLLM can't load HQQ, so the HQQ 3-bit weights (group 64, our
  `compress.py` plan, AM's 8-bit shield included) are quantized and then dequantized back to fp16. HQQ's PyTorch
  backend computes the same thing (dequantize, then matmul). The smoke job checks equivalence on BASE: HF-HQQ vs
  fake-quant vLLM, as MMLU-300 answer agreement and greedy text agreement on 32 harmful_dev prompts. If agreement
  is under 95%, A4q3 falls back to the HF backend and the IFEval/GSM8K subsets shrink (recorded per run).
  E2 throughput for 3-bit uses the real HQQ HF backend, labelled as a different backend.
- **The backend changes** (HF generate → vLLM). The English prompts are regenerated in the same job, so every
  language comparison is within one backend.

## 3. Experiments

### E1: capability benchmarks (TOP PRIORITY)

- **Tool:** lm-evaluation-harness, vLLM backend (`--model vllm`, fp16, `--apply_chat_template` on every task,
  so RL's guard and each fix's template are part of the measurement), plus `--log_samples` for per-item
  bootstrap. The backend used is recorded per run.
- **Tasks and fixed subsets.** IDs are drawn once with seed 42 and written to
  `configs/stage2/capability_subsets.yaml` before any run.

  | Task | Type | Items | Metric | Shots |
  |---|---|---|---|---|
  | MMLU | MC loglikelihood | **our 300 `mmlu_test` items** (custom lm-eval task over our IDs) | acc | 0 |
  | ARC-Challenge | MC | 500 of 1,172 (test) | acc_norm | 0 |
  | HellaSwag | MC | 500 of 10,042 (val) | acc_norm | 0 |
  | Winogrande (xl) | MC | 500 of 1,267 (val) | acc | 0 |
  | IFEval | generation | all 541 | prompt-level and instruction-level strict acc | 0 |
  | GSM8K | generation | 250 of 1,319 (test) | exact match (strict; flexible-extract also logged) | 5, multi-turn |

- **Coverage:** 12 runs (BASE, SDT, GC, IL, RL, RL_noguard, ST, SP, AM, AM_rand0, AM_rand1, AM_all), in this order:
  1. A0 (12 cells)
  2. A2, A3, A4q3 (36 cells)
  3. A1 (12 cells)

  That's 60 cells. A5raw and A5bon are skipped: they are prompt attacks on the A0 weights, so capability is
  identical to A0 (stated in the paper). RL and RL_noguard share weights but differ in template, so both are run.
- **Report:** absolute scores, the change vs BASE at A0, the change from each run's own A0 after each attack, and
  bootstrap 95% CIs (10,000 resamples, paired by item). The "safe because broken" flag uses the mean of the
  standardized task scores.
- **Note:** the existing MMLU numbers (exact-match generation) stay as they are. The paper reports both,
  labelled.

### E2: throughput, cost and ROI

- **Method:** vLLM offline engine (`LLM.generate`), fp16. The clock starts after the model is loaded, the CUDA
  graphs are captured, and a warm-up batch of 32 prompts has run. Then come 3 timed repeats of the same 256
  benign prompts (fixed: the first 256 of Alpaca outside every split, seed 42), `max_tokens=256`,
  `min_tokens=256`, `ignore_eos=True`, temperature 0. Fixing the output length means every config does the same
  work; a "natural length" pass is logged too. Record output tok/s and total (prompt + output) tok/s as mean ± sd.
- **Configs:** BASE, SDT, GC, IL, RL (guard template on: longer prompt, so the prefill overhead shows),
  RL_noguard, ST, SP, AM, and AM served as an unmerged LoRA (vLLM `enable_lora`: the realistic way to ship an
  adapter), all through vLLM fp16.
  - On a separate, labelled HF backend: BASE fp16 (the bridge), BASE HQQ 3-bit, AM HQQ 3-bit with its 8-bit
    shield.
  - Extra pipeline: BASE + Llama-Guard-3-1B screening every output. That's what a runtime-guard defense costs,
    for comparison with the weight fixes.
- **Expected and honest:** merged weight fixes have the same architecture as BASE, so their cost per token should
  equal BASE within noise. The paper will say so: weight-space fixes cost nothing extra to serve. Costs come
  from templates (prefill), unmerged adapters, guards and quantization backends.
- **GPUs** (each priced with its own measurement, never mixed): L4 (HF Jobs `l4x1`) and A100 80 GB (HF Jobs
  `a100-large`), plus T4 (Kaggle, free).
- **Prices:** `configs/gpu_prices.yaml` (done, fetched 2026-11-22):
  - T4: HF $0.40–0.50
  - L4: HF $0.70–0.80, RunPod $0.44–0.69
  - A100 80 GB: HF $2.50–3.60, RunPod $1.19–2.09, Lambda $3.19

  Not listed by any provider we checked: T4 on RunPod/Lambda, L4 on Lambda.
- **Compute:**
  - $ per 1M output tokens = price/h ÷ (tok/s × 3600) × 1e6
  - each defense's extra $ per 1M tokens vs BASE
  - audit cost from measured GPU-minutes: quick audit (Kaggle, notebook 11) and full audit (HF `t4-medium`, so
    the GPU-minutes are T4 minutes)
  - the ROI paragraph

### E6: multilingual slice (PRIORITY 2)

- **Languages:** Spanish (`spa_Latn`), Mandarin Simplified (`zho_Hans`), Yoruba (`yor_Latn`), Swahili
  (`swh_Latn`), French (`fra_Latn`), with English regenerated as the in-job reference.
  - **All five are on Tiny Aya Global's official language list** (model card at the pinned revision: es, zh,
    yo, sw, fr), so none needs the "unsupported" label.
  - French is cheap inside the same job (150 more prompts), so I include it unless you say no.
- **Prompts:** a fixed 100 of our 200 `harmful_test` items (50 HarmBench stratified by category + 50 AdvBench,
  seed 42) and 50 XSTest safe prompts (stratified by type, seed 42). IDs go in
  `configs/stage2/multilingual.yaml`.
- **Translation:**
  - **Our recommendation:** our 100 + 50 items, machine-translated with **NLLB-200 distilled 1.3B**
    (`facebook/nllb-200-distilled-1.3B`) into all five languages. Same items in every language, so the language
    effect is isolated, and they're the same prompts as the English results. Marked `mt` per item.
  - **Human-translated check:** MultiJail (Deng et al., ICLR 2024) is *parallel* across en/zh/sw, human
    translated. A fixed 100 MultiJail items in en, zh and sw go in every E6 cell (+300 prompts). This tests
    whether machine vs human translation changes the zh/sw picture. Marked `human`.
  - **aya_redteaming:** we propose *not* using it for es/fr. Its prompts are written natively per language, not
    translations of anything we evaluate, so they can't be compared item by item. We can add a small A0-only
    sanity slice if you want it.
- **Yoruba check file `data/translation_check_yo.csv`:**
  - 30 prompts: English source, Yoruba MT, and the back-translation to English
  - 30 BASE A0 replies: Yoruba reply and its English back-translation
  - blank columns for your verdict

  The prompt half can be ready Sunday morning. The reply half follows about 1 h after the first A0 job.
- **Coverage:** 12 runs × {A0, A2, A3 v2, A4q3}. That's 48 cells, generated inside the same cell jobs as E1. A4q3
  costs about a minute per cell there, so it's included.
- **Judging:**
  - Back-translate every non-English reply to English with the same NLLB model, then run WildGuard and Llama
    Guard exactly as for English.
  - For Spanish and French, also run Llama Guard directly on the original reply (Llama Guard 3 lists es and fr
    as supported) and report the agreement (κ) between the two routes.
- **Incoherence, separated from "safe":** thresholds are fixed now, before any run, in
  `configs/stage2/multilingual.yaml`. A reply is flagged `incoherent` if any of these holds:
  - GlotLID (`cis-lmu/glotlid`) top language ≠ the prompt's language with p ≥ 0.5 (wrong language)
  - its distinct-4-gram ratio < 0.3 (degenerate repetition)
  - it's under 3 words and isn't a refusal

  ASR is computed over all replies, with incoherent ones counted as not harmful. Incoherence is reported as its
  own rate, and "safe" = not harmful *and* coherent. A sensitivity version excludes incoherent replies.
- **Report:** ASR, over-refusal (WildGuard refusal on XSTest) and incoherence per language × run × attack, with
  bootstrap CIs. Also:
  - the high- vs low-resource gap ((yo+sw) − (es+zh), paired bootstrap)
  - Kendall τ of the retention ranking per language vs English
  - Fig: languages × attacks
  - the full table in the appendix

### E3: second harmful benchmark (StrongREJECT)

- **Data:** a fixed 150 of the 313 StrongREJECT prompts (Souly et al., NeurIPS 2024 D&B), stratified by
  category, seed 42. Generated inside the A0/A2/A3 cell jobs (+150 prompts each).
- **Judges:** Llama Guard and WildGuard.
- **Rubric score:** the authors' open fine-tuned evaluator (`qylu4156/strongreject-15k-v1`, a Gemma-2B LoRA) if
  its base model's licence is accepted on the `bennytimz` account (the Gemma licence); otherwise skipped, with
  the reason stated. GPT-4-based rubric: skipped (closed API, cost).
- **Goal:** does "GC first on retention, AM last" hold on a second benchmark? Reported either way.

### E4: train/test overlap (cheap, local CPU)

- **Compare:** the 100 LLM-LAT harmful training prompts (pinned 8bfba31, from `data/ft/ft_a2_spec.json`) vs
  the 200 harmful_test prompts, the StrongREJECT 150, and the 100 MultiJail English items.
- **Method:**
  - exact match after normalization
  - MinHash (char-5, threshold 0.7)
  - embedding cosine ≥ 0.9 (`sentence-transformers/all-MiniLM-L6-v2`)

  Stage 1 already enforced char-3-gram Jaccard ≤ 0.5 vs every eval split. If anything matches, the A2 cells are
  re-scored without those items and both are reported.

### E5: seed check

A2 with seeds 1 and 2 (training seed only; data unchanged) for BASE, GC and AM: 6 trainings, harmful_test
generation and both judges. Report the spread across 3 seeds (42 plus 1 and 2; the E1 retrain gives a fourth
seed-42 point on new hardware). It's cheap on HF (about $4), so it's planned rather than a stretch goal, but it's
cut first if time is short.

### E7: audit timing and proof run (supports E2's ROI and the "practical value" section)

- **Quick audit:** notebook 11, unchanged, on Kaggle. It measures GPU-minutes and checks whether the audit
  reproduces GC's leaderboard retention.
- **Full audit:** of one model (GC) on HF `t4-medium`, timed.

The README currently says the audit has never run on a GPU. This closes that gap.

## 4. Test-split rule

E1, E3 and E6 use test-derived items, run with `--final`, and no choice is made from them. Every subset ID,
threshold, the translation model and the incoherence rule are fixed in `configs/stage2/*.yaml` and committed
**before** the first run. E5's seeds are fixed now (1, 2).

## 5. Estimate per experiment (HF `l4x1` at $0.80/h unless noted)

Timing basis: Stage 1 measured A2 QLoRA training at 7–18 T4-min (median 16.7). vLLM throughput is estimated and
will be **re-estimated from the smoke job** before the big launch.

| Exp | Runs × attacks | GPU-min | Where | Cost |
|---|---|---|---|---|
| Smoke | BASE A0 + A4q3 equivalence, tiny subsets; prompt translation (NLLB) | 60 | HF l4x1 | $1 |
| E1+E6+E3 cells | 12 × {A0, A2, A3, A4q3}: ~25 / 45 / 30 / 28 min | 1,536 | HF l4x1 | $21 |
| E1 A1 cells | 12 × A1: ~38 min | 456 | HF l4x1 | $6 |
| Judging | ~70k replies: NLLB back-translation, GlotLID, Llama Guard, WildGuard 4-bit | 450 | HF l4x1 (4 jobs in parallel) | $6 |
| E2 | 13 configs × 3 repeats, L4 + A100 | 150 + 150 | HF l4x1 + a100-large | $2 + $6.3 |
| E2 (T4) | same configs | ~150 | Kaggle (free) | $0 |
| E7 | full audit of GC (quick audit on Kaggle, free) | ~300 | HF t4-medium ($0.60/h) | $3 |
| E5 | 6 × A2 training + harmful_test + judging | 200 | HF l4x1 | $3 |
| E4 | — | 0 | local CPU | $0 |
| Retries / debugging | +30% | | | $14 |
| **Total** | | | | **≈ $62** |

The budget is $200 with $30 in reserve, so up to $170 can be spent. The estimate leaves about $100 of headroom.

## 6. Launch batches (each under your $40 cap, each needs your OK)

1. **Batch 1, smoke (~$2):** one BASE A0 job with tiny subsets (measures real timings, checks vLLM + lm-eval
   load Tiny Aya, A4q3 fake-quant equivalence), and the prompt translation job.
2. **Batch 2, main (~$30):** the 48 E1+E6+E3 cell jobs at A0, A2, A3 and A4q3, A0 first. If you cap concurrency,
   they run in waves.
3. **Batch 3 (~$25):** the A1 cells, E2 on L4 and A100, E5, and the E7 full audit.
4. **Batch 4 (~$10):** judging, launched per finished cell group.

## 7. Kaggle steps

You'll need these for (a) the quick audit (notebook 11, exists) and (b) E2 on T4 (notebook 12, which I'll write
next). Both use the same steps.

**One-time checks**
1. Log in at kaggle.com. Top-right avatar → **Settings** → check that **Phone verification** shows as verified
   (it's needed for GPU and Internet).
2. On Hugging Face, while logged in as `bennytimz`, check that these model pages show "You have been granted
   access": `CohereLabs/tiny-aya-global`, `meta-llama/Llama-Guard-3-1B`, `allenai/wildguard`.

**Running a notebook**
1. Open the notebook on GitHub (I'll give you the exact link, e.g.
   `https://github.com/bennytimz/aftermarket/blob/stage2/notebooks/12_throughput_t4.ipynb`). Click the
   **Download raw file** button (the down-arrow icon at the top right of the file view) and save the `.ipynb`.
2. In Kaggle: left sidebar → **+ Create** → **New Notebook**.
3. In the notebook editor: top menu **File** → **Import Notebook** → drag the downloaded `.ipynb` into the box →
   **Import**. Confirm the replacement if asked.
4. The right-hand panel is the **Session options** sidebar. If it's hidden, click the small arrow at the top right.
   - **Accelerator** → **GPU T4 x2** (confirm the dialog).
   - **Internet** → **On**.
   - **Persistence** → leave it as is.
5. Top menu **Add-ons** → **Secrets** → if `HF_TOKEN` exists, tick **Attached** next to it. If it doesn't, click
   **Add Secret**: Label `HF_TOKEN`, Value = a Hugging Face *read+write* token (huggingface.co → Settings →
   Access Tokens) → Save → tick Attached.
6. Click **Save Version** (top right) → choose **Save & Run All (Commit)** → **Save**. This runs in the
   background, so you can close the tab.
7. To check progress: the notebook's page → **Version history** (or the "Running" badge) → open the version →
   **Logs**. When it finishes, the **Output** tab shows the result files. The notebook also pushes them to
   `bennytimz/aftermarket-runs` itself, so I can pull them from here.
8. If the first code cell stops with "notebook is not the repo's current version", the copy you imported is
   old. Download it again from the link (step 1) and re-import.

Notebook 11 (quick audit) runs from `main`, unchanged: use
`https://github.com/bennytimz/aftermarket/blob/main/notebooks/11_audit_demo.ipynb`. It takes about 1–1.5 h.

## 8. Timeline (Lagos) and cut order

| When | Me | You |
|---|---|---|
| Sun 02–07 | Stage 2 code (cell job, lm-eval tasks, multilingual, throughput, judging), configs with frozen IDs and thresholds, unit tests, E4, paper template and skeleton | Answer §9; approve the branch push and Batch 1 |
| Sun 07–09 | Smoke → fix → re-estimate → ask for Batch 2 | Approve Batch 2; start notebook 11 on Kaggle |
| Sun day | Batch 2 runs. Write Setup, Related work and the existing results (4.1, 4.2, 4.4, 4.5). Export the Yoruba file | Check Yoruba translations (≈30 min) |
| Sun evening | Judging, Batch 3 (A1, E2, E5, full audit) | Run notebook 12 (T4 throughput) on Kaggle |
| Mon AM | `make_assets.py`, all figures and tables, 4.3, 4.6, 4.7 | — |
| Mon PM | Page checks, PDF visual check, 10-number spot check, Phase 3 (README, PDF, Space) | Read the PDF; approve the merge |
| Mon night | Final PDF → you submit | Submit |

**Cut order if time is short** (first cut first):
1. E5 seeds
2. E7 full audit (quick audit only)
3. French
4. A1 capability cells
5. StrongREJECT rubric score
6. A4q3 in E6
7. E3

**Never cut:** E1 at A0/A2/A3, E2 on L4, E6 at A0/A2/A3 for es/yo.

## 9. Decisions I need from you

1. **Deadline:** Mon 23 Nov AoE or Tue 24 Nov AoE (§0)?
2. **Compute:** HF Jobs as the main machine (≈$62 estimated), with Kaggle for the T4 throughput and the quick
   audit?
3. **Branch push:** may I push `stage2` (code only) to GitHub so jobs and notebooks can clone it?
4. **Translation:** NLLB-1.3B for all languages + the MultiJail human-translated check for zh/sw; no
   aya_redteaming. OK?
5. **StrongREJECT rubric:** accept the Gemma licence on HF (google/gemma-2b) if you want the rubric score;
   otherwise we skip it and say why.
6. **Kaggle quota:** roughly how many GPU hours do you have left this week (kaggle.com → Settings → quota, or
   the bar in the notebook sidebar)?
