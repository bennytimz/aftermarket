# Abstract

**Aftermarket: does a safety fix survive the next person who downloads the model?**

Safety fixes for open-weight models are evaluated on the checkpoint their authors release. But a released model
rarely stays as released: users fine-tune it, compress it for a phone, serve it without its chat template, or
remove its refusals on purpose. We ask which of this hackathon's public safety fixes for Tiny Aya Global survive
those everyday changes.

We collected the public fixes (one-shot weight repair, Lisa and SafeGrad hardening, a prompt-guard template,
rank-one repair, and a SafeTune safety-vector repair) and ran each, plus the base model, through one shared attack
battery: a benign fine-tune, a fine-tune with 100 harmful examples, abliteration of the refusal direction, HQQ
compression to 4, 3 and 2 bits, a stripped chat template, and best-of-8 prompt perturbations. Every cell uses the
same seed, settings and held-out prompts: 200 HarmBench/AdvBench harmful prompts (attack success judged by
Llama-Guard-3-1B, with WildGuard as a second judge), 250 XSTest safe prompts for over-refusal, and 300 MMLU
questions for capability, so a model cannot look safe by being broken. Uncertainty comes from bootstrap
confidence intervals over prompts and paired tests against the base model. All choices were made on separate
dev prompts.

Across <!--v:n_models-->TBD<!--/v--> model runs and <!--v:n_cells-->TBD<!--/v--> model × attack cells, the hardest
change was <!--v:hardest_attack-->TBD<!--/v--> (mean attack success <!--v:hardest_attack_asr-->TBD<!--/v-->). The
base model's attack success went from <!--v:base_asr_a0-->TBD<!--/v--> as released to
<!--v:base_asr_a2-->TBD<!--/v--> after the harmful fine-tune and <!--v:base_asr_a3-->TBD<!--/v--> after abliteration.
The most robust public fix by retention (1 − mean rise in attack success over the five attack families) was
<!--v:top_retention_run-->TBD<!--/v--> at <!--v:top_retention-->TBD<!--/v-->. The two judges agreed on
<!--v:agree_lg_wg-->TBD<!--/v--> of responses (Cohen's κ <!--v:kappa_lg_wg-->TBD<!--/v-->).

We then built AM, a defense combining what the leaderboard rewards: safety layers located with SafeTune and
CircuitKIT, SafeGrad hardening restricted to those layers, training against an ablated refusal direction, and an
8-bit shield for those layers under compression. Against random-layer and all-layer controls, AM reached retention
<!--v:am_retention-->TBD<!--/v--> (rank <!--v:am_rank-->TBD<!--/v-->; vs base <!--v:am_vs_base_retention-->TBD<!--/v-->),
with attack success <!--v:am_asr_a2-->TBD<!--/v--> after the harmful fine-tune, reported as measured.

We release the battery, judges and statistics as a reusable suite built on CuratorKIT, SafeTune, CircuitKIT and
AuditKIT, along with metrics and verdicts only: no attack-tuned weights and no harmful completions. Every team is
credited and sent its own results before publication.
