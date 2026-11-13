<!-- DRAFT, not posted. Post manually. Placeholders are filled by `python -m aftermarket.report`;
check that no TBD remains before posting. -->
**Where to post:** GitHub issue on https://github.com/DhruvParmar051/safety-doesnt-translate
**Title:** Thanks for Safety Doesn't Translate: post-release robustness results from the Aftermarket project

---

Hi! Thank you for releasing Safety Doesn't Translate openly. Being able to download and run it is exactly what made this
project possible.

We're Aftermarket, another team in the Lexsi AI Safety & Alignment Hackathon. We wanted to know how the public
safety fixes for Tiny Aya hold up after release, when people fine-tune, compress, or re-serve a model. So we ran
every public fix we could find, plus Tiny Aya itself, through the same battery, with the same prompts, judges
and settings for everyone.

**What we tested:** the published one-shot repair checkpoint (`DhruvParmar051/tiny-aya-global-oneshot-repair`, revision `2c6dc02`) with its own chat template.

Your README's headline numbers are for the 2% setting, but the published checkpoint is the 1% repair, so that is what we tested (your README reports 8.6% for it). We say this next to your row so nobody compares it with the 2% numbers.

**The battery:** a benign fine-tune (A1), a fine-tune with 100 harmful examples (A2), refusal-direction ablation
(A3), HQQ compression to 4/3/2 bits (A4), the chat template removed (A5raw), and best-of-8 prompt perturbations
(A5bon). Measured on 200 held-out HarmBench/AdvBench prompts (attack success judged by Llama-Guard-3-1B, with
WildGuard as a second judge), 250 XSTest safe prompts (over-refusal) and 300 MMLU questions (capability).
Brackets are 95% bootstrap intervals.

**Your row (held-out test prompts):**

<!-- BEGIN row_SDT -->
_Pending: no judged results for this table yet._
<!-- END row_SDT -->

**Retention** (1 − the mean rise in attack success across the five attack families; 1 = nothing lost):
`SDT` <!--v:SDT_retention-->TBD<!--/v--> (rank <!--v:SDT_rank-->TBD<!--/v-->).

We see this as information about robustness after release, not as a verdict on your fix: every fix we tested,
including our own, has attacks it resists and attacks it doesn't, and some of these attacks were never what your
method set out to address. The full leaderboard, method and code are at https://github.com/bennytimz/aftermarket.

Please tell us if anything about how we loaded or ran your model doesn't match how you intended it to be used.
We're happy to correct it, re-run, and note your response next to your row before the write-up is final. We're
also happy to share more details, per-attack breakdowns or redacted examples privately. We don't publish any
attack-modified weights or full harmful outputs.

Thanks again, and congratulations on shipping!
