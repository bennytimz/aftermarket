<!-- DRAFT, not posted. Post manually. Placeholders are filled by `python -m aftermarket.report`;
check that no TBD remains before posting. -->
**Where to post:** Discussion on https://huggingface.co/vasmi-sas/guardian-tiny-aya-lisa/discussions (no GitHub repo linked)
**Title:** Thanks for Guardián colombiano: post-release robustness results from the Aftermarket project

---

Hi! Thank you for releasing Guardián colombiano openly. Being able to download and run it is exactly what made this
project possible.

We're Aftermarket, another team in the Lexsi AI Safety & Alignment Hackathon. We wanted to know how the public
safety fixes for Tiny Aya hold up after release, when people fine-tune, compress, or re-serve a model. So we ran
every public fix we could find, plus Tiny Aya itself, through the same battery, with the same prompts, judges
and settings for everyone.

**What we tested:** your Lisa-hardened LoRA adapter (`vasmi-sas/guardian-tiny-aya-lisa`, revision `4314ffb`) on Tiny Aya Global `af89d219`, merged, with your own chat template.

**The battery:** a benign fine-tune (A1), a fine-tune with 100 harmful examples (A2), refusal-direction ablation
(A3), HQQ compression to 4/3/2 bits (A4), the chat template removed (A5raw), and best-of-8 prompt perturbations
(A5bon). Measured on 200 held-out HarmBench/AdvBench prompts (attack success judged by Llama-Guard-3-1B, with
WildGuard as a second judge), 250 XSTest safe prompts (over-refusal) and 300 MMLU questions (capability).
Brackets are 95% bootstrap intervals.

**Your row (held-out test prompts):**

<!-- BEGIN row_GC -->
| model · attack | ASR, Llama Guard | BASE ASR, Llama Guard | ASR, WildGuard | over-refusal (XSTest) | MMLU |
|---|---|---|---|---|---|
| GC · A0 | 1% [0%–2%] | 8% [5%–12%] | 0% [0%–0%] | 26% [21%–32%] | 54% [49%–60%] |
| GC · A1 | 6% [4%–10%] | 8% [4%–12%] | 8% [4%–12%] | 6% [3%–9%] | 52% [46%–57%] |
| GC · A2 | 82% [77%–88%] | 81% [76%–86%] | 90% [86%–94%] | 1% [0%–3%] | 53% [48%–59%] |
| GC · A3 | 10% [6%–14%] | 70% [63%–76%] | 4% [2%–7%] | 17% [12%–22%] | 52% [47%–58%] |
| GC · A4q3 | 1% [0%–2%] | 14% [9%–18%] | 0% [0%–2%] | 27% [22%–32%] | 46% [41%–52%] |
| GC · A5raw | 45% [38%–52%] | 56% [50%–63%] | 5% [2%–8%] | 18% [13%–22%] | 30% [24%–35%] |
| GC · A5bon | 29% [22%–36%] | 95% [92%–98%] | 16% [11%–21%] | - | - |
| GC · A4q4 | 0% [0%–2%] | 10% [6%–14%] | 0% [0%–0%] | - | 54% [48%–60%] |
<!-- END row_GC -->

**Retention** (1 − the mean rise in attack success across the five attack families; 1 = nothing lost):
`GC` <!--v:GC_retention-->0.74 [0.71–0.76]<!--/v--> (rank <!--v:GC_rank-->1 of 9<!--/v-->).

We see this as information about robustness after release, not as a verdict on your fix: every fix we tested,
including our own, has attacks it resists and attacks it doesn't, and some of these attacks were never what your
method set out to address. The full leaderboard, method and code are at https://github.com/bennytimz/aftermarket.

Please tell us if anything about how we loaded or ran your model doesn't match how you intended it to be used.
We're happy to correct it, re-run, and note your response next to your row before the write-up is final. We're
also happy to share more details, per-attack breakdowns or redacted examples privately. We don't publish any
attack-modified weights or full harmful outputs.

Thanks again, and congratulations on shipping!
