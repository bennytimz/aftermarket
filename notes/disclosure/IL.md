<!-- DRAFT, not posted. Post manually. Placeholders are filled by `python -m aftermarket.report`;
check that no TBD remains before posting. -->
**Where to post:** Discussion on https://huggingface.co/Sykeclonex/signers-doubt-tiny-aya-hardened/discussions (no GitHub repo linked)
**Title:** Thanks for Intentlock (signers-doubt): post-release robustness results from the Aftermarket project

---

Hi! Thank you for releasing Intentlock (signers-doubt) openly. Being able to download and run it is exactly what made this
project possible.

We're Aftermarket, another team in the Lexsi AI Safety & Alignment Hackathon. We wanted to know how the public
safety fixes for Tiny Aya hold up after release, when people fine-tune, compress, or re-serve a model. So we ran
every public fix we could find, plus Tiny Aya itself, through the same battery, with the same prompts, judges
and settings for everyone.

**What we tested:** your SafeGrad-hardened LoRA adapter (`Sykeclonex/signers-doubt-tiny-aya-hardened`, revision `e3fdda2`), merged, with your own chat template.

The adapter config doesn't pin a base-model revision, so we used Tiny Aya Global `af89d219` (the revision every other model here uses). If you trained on a different revision, tell us and we'll note it.

**The battery:** a benign fine-tune (A1), a fine-tune with 100 harmful examples (A2), refusal-direction ablation
(A3), HQQ compression to 4/3/2 bits (A4), the chat template removed (A5raw), and best-of-8 prompt perturbations
(A5bon). Measured on 200 held-out HarmBench/AdvBench prompts (attack success judged by Llama-Guard-3-1B, with
WildGuard as a second judge), 250 XSTest safe prompts (over-refusal) and 300 MMLU questions (capability).
Brackets are 95% bootstrap intervals.

**Your row (held-out test prompts):**

<!-- BEGIN row_IL -->
| model · attack | ASR, Llama Guard | BASE ASR, Llama Guard | ASR, WildGuard | over-refusal (XSTest) | MMLU |
|---|---|---|---|---|---|
| IL · A0 | 8% [4%–12%] | 8% [5%–12%] | 6% [4%–10%] | 10% [6%–14%] | 54% [48%–59%] |
| IL · A1 | 10% [6%–15%] | 8% [4%–12%] | 10% [6%–14%] | 5% [3%–8%] | 54% [48%–59%] |
| IL · A2 | 86% [82%–91%] | 81% [76%–86%] | 90% [86%–94%] | 2% [0%–3%] | 55% [49%–60%] |
| IL · A3 | 68% [61%–74%] | 70% [63%–76%] | 54% [46%–60%] | 0% [0%–1%] | 53% [48%–59%] |
| IL · A4q3 | 11% [7%–16%] | 14% [9%–18%] | 6% [4%–10%] | 12% [8%–16%] | 49% [43%–55%] |
| IL · A5raw | 52% [45%–58%] | 56% [50%–63%] | 33% [26%–40%] | 4% [2%–7%] | 31% [26%–36%] |
| IL · A5bon | 94% [90%–97%] | 95% [92%–98%] | 64% [58%–71%] | - | - |
| IL · A4q4 | 8% [4%–12%] | 10% [6%–14%] | 6% [4%–10%] | - | 52% [46%–57%] |
<!-- END row_IL -->

**Retention** (1 − the mean rise in attack success across the five attack families; 1 = nothing lost):
`IL` <!--v:IL_retention-->0.58 [0.55–0.61]<!--/v--> (rank <!--v:IL_rank-->7 of 9<!--/v-->).

We see this as information about robustness after release, not as a verdict on your fix: every fix we tested,
including our own, has attacks it resists and attacks it doesn't, and some of these attacks were never what your
method set out to address. The full leaderboard, method and code are at https://github.com/bennytimz/aftermarket.

Please tell us if anything about how we loaded or ran your model doesn't match how you intended it to be used.
We're happy to correct it, re-run, and note your response next to your row before the write-up is final. We're
also happy to share more details, per-attack breakdowns or redacted examples privately. We don't publish any
attack-modified weights or full harmful outputs.

Thanks again, and congratulations on shipping!
