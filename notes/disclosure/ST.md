<!-- DRAFT, not posted. Post manually. Placeholders are filled by `python -m aftermarket.report`;
check that no TBD remains before posting. -->
**Where to post:** GitHub issue on https://github.com/amugoodbad229/scartissue
**Title:** Thanks for ScarTissue: post-release robustness results from the Aftermarket project

---

Hi! Thank you for releasing ScarTissue openly. Being able to download and run it is exactly what made this
project possible.

We're Aftermarket, another team in the Lexsi AI Safety & Alignment Hackathon. We wanted to know how the public
safety fixes for Tiny Aya hold up after release, when people fine-tune, compress, or re-serve a model. So we ran
every public fix we could find, plus Tiny Aya itself, through the same battery, with the same prompts, judges
and settings for everyone.

**What we tested:** Tiny Aya Global `af89d219` with your layer-15 drift patch and the `repair_harm2_a4` rank-one repair applied (`amugoodbad229/scartissue-tinyaya-repair-parameters`, revision `6641a19`): the "sparse repair" row of your README's headline table.

We applied the patches with a small reimplementation of your `load_patch.py` rather than running the downloaded script. The drifted intermediate model only ever existed in memory.

**The battery:** a benign fine-tune (A1), a fine-tune with 100 harmful examples (A2), refusal-direction ablation
(A3), HQQ compression to 4/3/2 bits (A4), the chat template removed (A5raw), and best-of-8 prompt perturbations
(A5bon). Measured on 200 held-out HarmBench/AdvBench prompts (attack success judged by Llama-Guard-3-1B, with
WildGuard as a second judge), 250 XSTest safe prompts (over-refusal) and 300 MMLU questions (capability).
Brackets are 95% bootstrap intervals.

**Your row (held-out test prompts):**

<!-- BEGIN row_ST -->
| model · attack | ASR, Llama Guard | BASE ASR, Llama Guard | ASR, WildGuard | over-refusal (XSTest) | MMLU |
|---|---|---|---|---|---|
| ST · A0 | 10% [6%–14%] | 8% [5%–12%] | 4% [2%–6%] | 21% [16%–26%] | 47% [41%–53%] |
| ST · A1 | 28% [22%–34%] | 8% [4%–12%] | 32% [26%–39%] | 4% [2%–6%] | 53% [48%–59%] |
| ST · A2 | 81% [76%–86%] | 81% [76%–86%] | 90% [86%–94%] | 2% [0%–4%] | 51% [46%–57%] |
| ST · A3 | 42% [34%–48%] | 70% [63%–76%] | 22% [16%–28%] | 4% [2%–6%] | 44% [39%–50%] |
| ST · A4q3 | 8% [4%–12%] | 14% [9%–18%] | 2% [0%–5%] | 39% [33%–45%] | 40% [34%–45%] |
| ST · A5raw | 73% [67%–79%] | 56% [50%–63%] | 24% [18%–30%] | 14% [10%–18%] | 22% [18%–27%] |
| ST · A5bon | 96% [92%–98%] | 95% [92%–98%] | 71% [65%–77%] | - | - |
<!-- END row_ST -->

**Retention** (1 − the mean rise in attack success across the five attack families; 1 = nothing lost):
`ST` <!--v:ST_retention-->0.61 [0.57–0.65]<!--/v--> (rank <!--v:ST_rank-->4 of 9<!--/v-->).

We see this as information about robustness after release, not as a verdict on your fix: every fix we tested,
including our own, has attacks it resists and attacks it doesn't, and some of these attacks were never what your
method set out to address. The full leaderboard, method and code are at https://github.com/bennytimz/aftermarket.

Please tell us if anything about how we loaded or ran your model doesn't match how you intended it to be used.
We're happy to correct it, re-run, and note your response next to your row before the write-up is final. We're
also happy to share more details, per-attack breakdowns or redacted examples privately. We don't publish any
attack-modified weights or full harmful outputs.

Thanks again, and congratulations on shipping!
