# Limitations and disclosures (for the write-up)

Choices made to fit the Kaggle GPU quota (~30 h/week on 2x T4) and T4 memory. Each one is stated in the
write-up as written here.

## AM recipe
- **SafeGrad (plain-loss variant, T4 memory).** AM is hardened with SafeTune's `SafeGradHFTrainer`
  (gradient surgery, Yi et al., arXiv:2508.07172), but its alignment gradient is the cross-entropy loss on
  refusal batches, not the paper's KL divergence to a frozen reference model: a second fp16 copy of Tiny Aya
  does not fit next to QLoRA training on a 16 GB T4. We do not cite the paper's KL-mode results as ours.
- **ReSta re-anchoring (PLAN.md §6, step 5) was dropped.** ReSta adds a safety vector (aligned - unaligned).
  No unaligned Tiny Aya exists, so with base = BASE and aligned = AM it would only extrapolate AM's own
  hardening delta (AM + 0.25 x (AM - BASE)), which is not the method's intent.
- **Only two candidates were compared** (k = 6 protected layers, spread-refusal off vs on). L* was still
  computed for k = 4, 6, 8, and the overlap between the three locate methods is reported for all three, but
  k itself was not selected on dev.
- **A2 (harmful fine-tune) is not in the selection loop.** Candidates were chosen on dev under A0, A3 and
  A4q3 only, because a fine-tune per candidate is the expensive part. The frozen AM is still attacked with A2
  on test, so the headline A2 result is not selected for; a weak A2 result is reported as is.
- **Two random-layer controls** (AM_rand0, AM_rand1) instead of three, plus AM_all. Controls match AM's data,
  steps, learning rate, seed and trainable parameter count.
- **EAP-IG (CircuitKIT) was dropped; L* is merged from two methods** (refusal-direction separation and SafeTune
  safety neurons). On Kaggle (notebook 05, 2026-11-13) it ran out of memory on both devices: on the T4 the
  TransformerLens per-head attention result (`(z * w).sum(-2)`) tried to allocate 12.2 GiB with 7.8 GB of fp16
  weights already resident, because the chat-templated calib prompts are ~380 tokens (batch 4); the CPU fallback
  (32 examples) was killed by the OOM killer (exit -9). Truncating the prompts to their last ~64 tokens at batch 1
  might fit, but it could not be verified off-GPU and the remaining quota went to the selection itself. Method
  overlap is therefore reported for one pair only.
- **The two locate methods pick disjoint layers (Jaccard 0.0 at k = 4, 6, 8), and that is real, not an indexing
  bug.** We checked: SafeTune scores decoder layer l by the MLP neurons inside layer l. The refusal-direction score
  was read at the residual stream *entering* layer l, i.e. one layer late, and its index 0 was the embedding
  output, whose harmful-vs-harmless direction is 0/0 = NaN and scrambled the mean-rank merge. Both are fixed (v2:
  scored at each layer's output; NaN ranks last) and L* was recomputed. The disagreement survives the fix: the
  refusal direction separates harmful from benign prompts best in early layers (about 2-9) while SafeTune's
  safety neurons peak in middle layers (about 12-20); across layers the two rankings are anti-correlated
  (Spearman about -0.43 on the v1 scores re-indexed). A linearly readable "harmful" feature forms early; the
  neurons whose activations differ most sit later.
- **L* is the refusal-direction top-k, not the planned mean-rank merge (changed after locate, before any training
  or evaluation).** PLAN.md §6 fixed a mean-rank merge of the locate methods in advance. Because the two remaining
  methods anti-correlate, the merge picked layers that are mediocre for both (e.g. 20, 21, 25 for k = 6). On
  2026-11-13, after seeing only the locate scores (calib_pairs / harmful_dev / benign_dev, never test) and BEFORE
  any candidate was trained or evaluated, L*_k became the top-k of the refusal-direction ranking alone: it is the
  method tied to what AM defends (A3 removes exactly this direction) and the one A3's own layer rule uses.
  safetune_neurons is still computed and reported as a diagnostic, and the disagreement itself is a finding for
  the write-up. The random-layer controls are drawn from outside this new L*. The choice was made by looking at
  locate output, so it is a post-hoc design change; no selection, retention or test number informed it.
- **The compression shield is part of AM's deployment recipe:** under A4, AM's protected layers stay 8-bit
  while the rest take the attack's bits. Other models get no shield, by design: the shield is what AM
  ships. The A4 rows of AM are reported with this marked.

- **AM_all has no compression shield; AM, AM_rand0 and AM_rand1 do** (each keeps its own 6 trained layers at
  8-bit under A4). Shielding "all layers" would just be an 8-bit model, so AM_all ships without one. Any A4
  comparison of placement must therefore be AM vs AM_rand, never AM vs AM_all, which would mix up layer choice
  with the shield. (In the current battery the controls face no A4 at all; see "Controls run a reduced battery".)
  AM_all also matches AM's trainable parameters by rank (rank 3 on 24 layers, 2 on 12) with lora_alpha fixed at 4,
  so its LoRA scale is 1.33 on rank-3 layers and 2.0 on rank-2 layers (AM: 2.0).

## A3 (abliteration): v1 layer selection was broken, re-run with causal selection
- **What was wrong.** A3 v1 took the diff-in-means direction at the layer with the largest Cohen's d separation of
  harmful_dev vs benign_dev projections (within 20-80% depth). The weight edit was correct (o_proj, down_proj and
  the tied embeddings orthogonalised, fp16 weights used at generation, same chat template for calibration and
  generation, projection residual ~1e-4 = fp16 rounding). But on Tiny Aya the separation peaks at layers 3-4 and
  falls with depth, so **every model got the window floor, layer 7**. That direction encodes harmful *content*, not
  refusal: removing it changed nothing. On harmful_test, A0 -> A3 keyword refusal was 0.82 -> 0.78 (BASE),
  1.00 -> 1.00 (GC), 0.78 -> 0.77 (IL), 0.42 -> 0.50 (RL), 0.82 -> 0.78 (RL_noguard), 0.52 -> 0.51 (SDT),
  0.83 -> 0.79 (SP), 0.32 -> 0.38 (ST); judged ASR BASE 0.09 -> 0.09 (Llama Guard) / 0.09 -> 0.07 (WildGuard),
  GC 0.01 -> 0.01 / 0.00 -> 0.00. The 20-80% window was not the cause (the metric, not the window, chose the floor).
- **Fix (v2, Arditi et al. 2024, App. C).** For each layer in the window, ablate its direction everywhere (forward
  hooks, verified equal to the weight edit on Cohere2) and measure keyword refusal of 32-token greedy replies on the
  first 32 harmful_dev prompts; drop layers whose ablation moves the benign_dev next-token distribution by KL > 0.1;
  lowest refusal wins (ties: higher separation). Simplifications vs the paper: the last post-template token only
  (not every post-instruction position), keyword refusal instead of refusal-token log-odds, no induce-refusal
  filter. Selection uses dev prompts only; the dev A3 cells share 32 prompts with selection (slightly optimistic on
  dev; test cells are clean). v1 results were moved to `archive/A3_v1/` on the private runs dataset, never mixed in.
- **If A3 v2 is still weak** (best layer's refusal near the no-ablation rate), that is a finding about Tiny Aya,
  reported as such.
- **What this means for the frozen AM.** AM's selection used v1 A3 retention on dev, but the choice did not depend
  on it: AMc_k6 was ineligible on over-refusal (0.46 vs BASE 0.38 + 3 pts) whatever its A3 score, so AMc_k6s was
  the only eligible candidate. AM's spread-refusal step, however, ablated the v1 direction (locate's layer-7
  direction, same rule) during training, so spread-refusal targeted a direction that turned out not to mediate
  refusal. AM stays frozen as is (no re-hardening after the fact); its test A3 cells use v2, like every other model,
  and the write-up states that spread-refusal was trained against the v1 direction. The two AMc_* dev A3 cells are
  kept as the record of what selection saw.
- **Where it ran.** All A3 cells (team models on test and dev, AM and controls on test) were re-run on a free Colab
  T4 (notebook 09), off the Kaggle quota.

## Test battery (quota)
- **A4 is reported at 3-bit (A4q3) for every model.** Notebook 03 v2 (team models x A3/A4 on test) was stopped
  after 6.3 h because HQQ generation is about 6x slower than fp16 A0 (measured: ~7.1 s vs ~1.2 s per harmful
  prompt), to keep quota for notebook 05 and the AM battery. A Hub check of the test cells it left:
  every team model (and BASE) has A3 and A4q3 on harmful_test, xstest_test and mmlu_test, so no fill-in cells
  were needed. A4q4 exists only for BASE, GC and IL (harmful_test and mmlu_test; no XSTest under A4q4 by
  design) and A4q2 for no model on test. The leaderboard's A4 column and the A4 term of every retention score
  are therefore A4q3 alone, for team models and AM alike, so A4 is like for like across rows. A4q4 / A4q2 cells
  are shown as extra data where they exist (last columns, marked partial) and never enter retention or
  rankings. We make no claim about any model at 4-bit or 2-bit beyond those extra cells. This changes PLAN.md
  §4's A4 (4/3/2-bit) to 3-bit only on test; 3-bit is also the A4 setting AM was selected on (dev A4q3).
- **AM's battery skips A4q4 and A4q2** for the same reason: AM faces A0, A1, A2, A3, A4q3, A5raw and A5bon.
- **Part of the AM battery ran on Colab, not Kaggle.** With 6 h 28 min of Kaggle quota left and no reset before
  the deadline, AM under A1 and A5bon and the three controls under A2 ran on a free Colab T4 (notebook 08, one GPU)
  and the rest on Kaggle 2x T4 (notebook 07). With 5.2 h left after notebook 05's first run, the controls' A2
  fine-tunes moved to Colab too, so the Kaggle part fits. Same GPU model, seed, configs, batch size and greedy decoding; only the run order across GPUs
  differs. Colab's image ships its own torch / transformers builds (bitsandbytes, peft and accelerate are pinned
  as in requirements.txt), so these cells can differ from a Kaggle run in floating-point detail; each manifest
  records the git sha, config and model revision. Judging also runs on Colab (notebook 06).
- **Controls run a reduced battery.** AM_rand0, AM_rand1 and AM_all face only A0, A2 and A3 (no A4): enough
  to show whether placing the hardening on L* matters for the weight-level attacks AM targets, not a full
  leaderboard row for each control. Whether L* matters under quantization is therefore not tested on test;
  under A4 only AM itself is reported (with its 8-bit shield).
- **No XSTest under A4q4, A4q2 or A5raw, for every model** (team models and AM alike, so cells stay
  comparable). These cells report harmful-prompt ASR and MMLU only. Over-refusal is mainly an A0 question;
  it is still measured at A0, A1, A2, A3 and A4q3. Taken because the measured-rate estimate of the AM battery
  (notebook 02 rates: ~0.6-0.8 prompts/s on harmful/XSTest, ~5 prompts/s on MMLU) was above 6 quota-hours.
