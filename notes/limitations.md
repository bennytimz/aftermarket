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
- **EAP-IG** (CircuitKIT) uses the first-token logit difference "I" (refusal) vs "Here" (compliance) on
  length-unmatched calib pairs, scores a layer by its kept nodes, and cannot pin the model revision
  (notes/upstream_issues.md). If it fails on GPU and CPU, L* is merged from the other two methods and the
  failure is recorded in locate.json.
- **The compression shield is part of AM's deployment recipe:** under A4, AM's protected layers stay 8-bit
  while the rest take the attack's bits. Other models get no shield, by design: the shield is what AM
  ships. The A4 rows of AM are reported with this marked.
