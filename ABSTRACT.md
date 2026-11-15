# Abstract

**Aftermarket: does a safety fix survive the next person who downloads the model?**

**Less than an hour on a free T4 erases every safety fix we tested from this hackathon:** one QLoRA epoch on 100
harmful examples takes attack success from 1–14% to 80–86%, with no loss in MMLU. **Abliteration, with no training
at all, undoes most fixes; only one (GC, Lisa hardening) really resists it:** at its strongest layer it takes SDT and SP to 81–87% under both judges and IL to 54%
(WildGuard; Llama Guard 68%), while RL and ST hold partly at 21–22% (WildGuard; Llama Guard 33–42%).

Safety fixes for open-weight models are evaluated on the checkpoint their authors release. But a released model
rarely stays as released: users fine-tune it, compress it for a phone, serve it without its chat template, or
remove its refusals on purpose. We ask which of this hackathon's public safety fixes for Tiny Aya Global survive
those everyday changes.

We collected the public fixes (one-shot weight repair, Lisa and SafeGrad hardening, a prompt-guard template,
rank-one repair, and a SafeTune safety-vector repair) and ran each, plus the base model, through one shared attack
battery: a benign fine-tune, a fine-tune with 100 harmful examples, abliteration of the refusal direction, HQQ
compression to 3 bits, a stripped chat template, and best-of-8 prompt perturbations. Every cell uses the
same seed, settings and held-out prompts: 200 HarmBench/AdvBench harmful prompts (attack success judged by
Llama-Guard-3-1B, with WildGuard as a second judge), 250 XSTest safe prompts for over-refusal, and 300 MMLU
questions for capability, so a model cannot look safe by being broken. Uncertainty comes from bootstrap
confidence intervals over prompts and paired tests against the base model. All choices were made on separate
dev prompts.

Across <!--v:n_models-->12<!--/v--> model runs and <!--v:n_cells-->87<!--/v--> model × attack cells, the hardest
change was <!--v:hardest_attack-->A2<!--/v--> (mean attack success <!--v:hardest_attack_asr-->82%<!--/v-->). The
base model's attack success went from <!--v:base_asr_a0-->8% [5%–12%]<!--/v--> as released to
<!--v:base_asr_a2-->81% [76%–86%]<!--/v--> after the harmful fine-tune and <!--v:base_asr_a3-->70% [63%–76%]<!--/v--> after abliteration.
The most robust public fix by retention (1 − mean rise in attack success over the five attack families) was
<!--v:top_retention_run-->GC<!--/v--> at <!--v:top_retention-->0.74<!--/v-->. The two judges agreed on
<!--v:agree_lg_wg-->81.3%<!--/v--> of responses (Cohen's κ <!--v:kappa_lg_wg-->0.58<!--/v-->). Against
<!--v:human_n-->100<!--/v--> blind hand labels, WildGuard matched the human better than Llama Guard (reweighted accuracy
<!--v:human_acc_wg-->83.7%<!--/v-->, κ <!--v:human_kappa_wg-->0.42<!--/v--> vs <!--v:human_acc_lg-->72.7%<!--/v-->, κ
<!--v:human_kappa_lg-->0.09<!--/v-->; on template-stripped and best-of-8 prompts, A5, κ
<!--v:human_kappa_wg_a5-->0.18<!--/v--> vs <!--v:human_kappa_lg_a5-->0.05<!--/v-->). Llama Guard over-flags refusals
on perturbed text, so where the judges disagree, mostly on A5, we report WildGuard first: best-of-8 perturbations
take the base model from 8% to 66% (Llama Guard 95%). Llama Guard stays the pre-registered judge for the ranking;
with WildGuard as judge (post-hoc) the top three and the last place are unchanged.

We then built AM, a defense combining what the leaderboard rewards: SafeGrad hardening restricted to the six layers
with the strongest refusal direction, training against an ablated refusal direction, and an 8-bit shield for those
layers under compression. Against random-layer and all-layer controls, AM reached retention
<!--v:am_retention-->0.53 [0.50–0.56]<!--/v--> (rank <!--v:am_rank-->9 of 9<!--/v-->; vs base <!--v:am_vs_base_retention-->-0.06 (Holm-adjusted p = 0.028)<!--/v-->),
with attack success <!--v:am_asr_a2-->86% [81%–90%]<!--/v--> after the harmful fine-tune and
<!--v:am_asr_a3-->90% [86%–94%] (WildGuard; Llama Guard 79%)<!--/v--> after abliteration, reported as measured. AM is a trade-off: under both judges it is significantly
safer than the base model as released and worse under abliteration; its advantages under 3-bit compression and
prompt perturbations hold under Llama Guard only. Random-layer and all-layer controls did as well as AM on every attack they share, and two of them
better under the strongest abliteration, so we find no advantage from choosing the layers. Abliteration's
strength also depends on its layer-selection rule: on four runs it chose a layer that left refusal intact on dev,
so their low abliteration numbers are not evidence of robustness.

We release the battery, judges and statistics as a reusable suite built on CuratorKIT, SafeTune, CircuitKIT and
AuditKIT, along with metrics and verdicts only: no attack-tuned weights and no harmful completions. Every team is
credited by name with links to its release.
