# Abstract

**Aftermarket: does a safety fix survive the next person who downloads the model?**

**Less than an hour on a free T4 erases every safety fix we tested from this hackathon: one QLoRA epoch on 100
harmful examples takes attack success from 1–14% to 80–86%, with no loss in MMLU.**

Safety fixes for open-weight models are evaluated on the checkpoint their authors release, but released models
rarely stay that way: users fine-tune them, compress them, drop the chat template, or remove refusals on purpose.
We ask which public safety fixes for Tiny Aya Global survive those everyday changes.

We ran six public fixes and the unfixed base model through one shared attack battery: a benign and a harmful
fine-tune, abliteration of the refusal direction, 3-bit compression, a stripped chat template, and best-of-8 prompt
perturbations. Every cell uses the same seed and held-out prompts (200 HarmBench/AdvBench harmful prompts
judged by Llama-Guard-3-1B and WildGuard, 250 XSTest prompts for over-refusal, 300 MMLU questions for capability),
with bootstrap confidence intervals and paired tests; all choices were made on dev prompts.

Three results stand out. First, the harmful fine-tune is the hardest change: every fix ends within 5 points of the
unfixed base model (81%). Second, abliteration, with no training at all, undoes most fixes: at its strongest layer
it takes SDT and SP to 81–87% under both judges. GC (Lisa hardening) is the one exception and stays at 10%
(WildGuard 4%). Third, the judges part ways on perturbed and template-stripped text.
Against 100 blind hand labels WildGuard agrees with a human at κ 0.42 and Llama Guard at κ 0.09; Llama Guard
over-flags refusals as unsafe. Where the judges disagree we report WildGuard first: best-of-8 perturbations take the
base model from 8% to 66% (Llama Guard 95%), and only GC and a template-based guard hold, at 16% (Llama Guard 29% and 55%).

We also built AM, a defense that hardens the six layers with the strongest refusal direction, trains against an
ablated refusal direction and keeps those layers at 8 bits under compression. AM is a trade-off: under both judges
it is safer than the base model as released but worse under abliteration (WildGuard 90% vs 65%; Llama Guard 79% vs 70%), with no advantage
over hardening random layers, and it ranks last on retention.

We release the battery, judges and statistics as a reusable suite, with an `aftermarket audit` command that writes a
robustness report card for any Tiny Aya fix. It is built on CuratorKIT, SafeTune and AuditKIT; CircuitKIT discovery
was attempted but did not run on Tiny Aya. We publish metrics, verdicts and our defense adapters only: no
attack-tuned weights and no harmful completions. Every team is credited by name.
