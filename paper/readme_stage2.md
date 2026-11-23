## Stage 2 results

> **Paper:** [Aftermarket: Most Safety Fixes Break After Release, and the Tools That Measure Them Can Mislead](docs/aftermarket_stage2.pdf)
> (EMNLP format, 6 pages + appendix). Every number below comes from the same `paper/derived/numbers.json` as the
> paper (this section is written by `python paper/make_readme.py`). Stage 2 experiments were not pre-registered.

Stage 2 adds what Stage 1 lacked: capability beyond MMLU (six benchmarks), six languages with a native-speaker check,
a judge audit against blind hand labels, serving cost on three GPUs, and a seed check of the harmful fine-tune.

**Key findings**

1. **Most fixes do not survive.** One QLoRA epoch on {{cfg.a2.harmful}} harmful examples takes every fix from
   {{lg.A0.fixes.min}}–{{lg.A0.fixes.max}}% to {{lg.A2.min}}–{{lg.A2.max}}% harmful replies (Llama Guard; WildGuard
   {{wg.A2.min}}–{{wg.A2.max}}%), and abliteration undoes most of them. Retraining that fine-tune with two other seeds
   moves BASE, GC and AM by at most {{seed.spread.wg.max}} points (WildGuard).
2. **Safety at release does not predict durability** (Spearman ρ = {{rho.a0ret.lg}}, Llama Guard). GC ranks first
   (retention {{ret.GC}} {{ret.GC.ci}}), RL second ({{ret.RL}}), BASE scores {{ret.BASE}}.
3. **Attacked models are unsafe, not broken.** BASE averages {{cap.comp.BASE.A0}}% over six benchmarks as released;
   the harmful fine-tune, abliteration and the benign fine-tune barely move capability. No cell comes near the 30%
   "safe because broken" flag (lowest MMLU {{cap.mmlu.min}}%). Only 3-bit compression costs capability for every run.
4. **The judges can mislead.** Against {{hum.nlabels}} blind hand labels, Llama Guard agrees with a human at
   κ = {{kappa.lg.all}} (WildGuard κ = {{kappa.wg.all}}); both fall to near chance on template-stripped and perturbed text.
5. **Back-translated judging overstates harm in Yoruba.** After 3-bit compression WildGuard rates
   {{mlmean.A4q3.yo.wg}}% of Yoruba replies harmful (English {{mlmean.A4q3.en.wg}}%), yet the author, a native Yoruba
   speaker, confirms only {{humjudge.flag.yes}} of {{humjudge.flag.n}} flagged replies.
6. **Our layer-targeted defense (AM) is safer at release but less durable** (retention {{ret.AM}} {{ret.AM.ci}}, last),
   and no better than hardening random layers.
7. **Safety in the weights is free to serve; a prompt guard is not.** RL's guard cuts L4 throughput by
   {{cost.L4.RL.loss}}%: serving 1B output tokens a month on one L4 costs ${{roi.L4.weights.min}}–{{roi.L4.weights.max}}
   with BASE or a weight-only fix and ${{roi.L4.RL}} with RL's guard, ${{roi.L4.RL.extra}} more every month.

![Rise in harmful-reply rate after each post-release change](paper/figures/fig1_heatmap.png)
*Fig. 1 of the paper. Most fixes do not survive: rise in harmful-reply rate (WildGuard, percentage points) after each
post-release change, relative to the same run as released; row labels give the as-released rate. Bold: paired 95%
bootstrap CI excludes 0.*

![Capability change on six benchmarks](paper/figures/fig3_capability.png)
*Fig. 3 of the paper. Fixes and attacks barely move capability, except 3-bit: change in accuracy (points) on six
benchmarks and their mean. Left: each fix as released vs BASE. Right: after the harmful fine-tune vs the same run as
released, then the mean change after abliteration, 3-bit and the benign fine-tune.*

![Judged harmful-reply rate by language](paper/figures/fig6_multilingual.png)
*Fig. 5 of the paper. Judged harmful-reply rate by language (WildGuard on NLLB back-translations): bar = mean over the
12 runs, dots = single runs; 100 harmful prompts per cell. The native-speaker check finds the Yoruba bars overstate harm.*

**Serving cost** (Table 1 of the paper): output tokens/s (vLLM fp16, 256 prompts × 256 tokens, each fix with its own
chat template; mean of 3 repeats, sd < 2%) and $ per 1M output tokens at Hugging Face on-demand prices
(T4 ${{cost.price.T4}}/h, L4 ${{cost.price.L4}}/h, A100 ${{cost.price.A100}}/h).

{{COST_TABLE}}

GC's template drops the default system prompt, so it serves {{cost.L4.GC.gain}}% faster on an L4, a property of its
template rather than its fix. Serving AM as an unmerged LoRA costs {{cost.L4.lora.overhead}}%, so merge it;
screening every output with Llama Guard 3 1B costs {{cost.L4.guard.overhead}}%.

**Audit cost.** A full `aftermarket audit` of one fix (all seven attacks, both judges) took {{audit.full.hours}} hours
on one T4 and cost ${{audit.full.usd}} once, the price of serving {{audit.full.mtok}}M tokens on an L4. Its retention
for GC ({{audit.full.ret}}) matches the leaderboard's ({{audit.full.retref}}).
