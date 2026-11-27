# Page budget (limit 6.0 pages before References, Limitations and Ethics included; appendix <= 6.0)

Measured with `python check_pages.py` after every build (`make pdf` runs it and fails if References start after 6.0 pages or the appendix exceeds 6.0; the organizer counts everything before References, 27 Nov).
Target is 5.5 pages, leaving 0.5 for float placement and late edits. A section over its cap is cut
or moved to the appendix immediately, not on Tuesday.

| Part | Cap (pp) | Main-text content | Status (22 Nov) |
|---|---|---|---|
| Title, abstract, intro | 1.00 | 4 contributions, headline paragraph | drafted, ~1.0 |
| 2 Related work | 0.40 | 5 short paragraphs | drafted, ~0.4 |
| 3 Setup | 0.65 | models, battery (one line per attack), evaluation, Stage 2 one paragraph | drafted, ~0.6 |
| 4.1 Survive + Fig 1 (full width) | 0.55 | heatmap, one paragraph | drafted |
| 4.2 Retention + Fig 2 (1 col) | 0.35 | ranking, release-vs-retention rho | drafted |
| 4.3 Capability + Fig 3 (1 col) | 0.40 | composite / per-benchmark change after A2, A3 (A0 levels in text) | waits on E1 |
| 4.4 Judges + Fig 4 (1 col) | 0.35 | kappa by scope, reporting rule | drafted |
| 4.5 Defense + main scatter (1 col) | 0.45 | negative result, controls | drafted, scatter not placed |
| 4.6 Multilingual + 1 fig (1 col) | 0.45 | ASR by language x attack, incoherence share | waits on E6 |
| 4.7 Cost (small inline table, no fig) | 0.30 | $/1M tokens per fix, defense overhead, audit cost, ROI | waits on E2 |
| 5 Practical takeaways | 0.25 | 4 bullets + `aftermarket audit` | stub |
| 6 Conclusion | 0.15 | 3 sentences | stub |
| **Total** | **5.30** | | measured 3.4 with 3 figures placed |

Main text has at most 5 figures (Fig 1-4 + the AM scatter) and 1 small table. No other tables in the main text.

## Already in, or going to, the appendix (not main text)
- Attack hyperparameters, abliteration layer rule, BoN settings, data construction, capability subset sizes (App. B, moved 22 Nov).
- Models table with revisions, teams and licences (App. A).
- Full cell tables: ASR (both judges), over-refusal, MMLU, with CIs; Llama Guard version of Fig 1; 4/2-bit results; retention without A3.
- Per-benchmark capability table with CIs and change vs BASE / vs A0.
- Per-language x attack x run table (E6), MultiJail human-translation check, es/fr direct-judging agreement, Yoruba translation check.
- StrongREJECT (E3) table: one sentence in 4.4 at most.
- E5 seed spread: one sentence in 4.1 at most, table in appendix.
- E4 overlap check and A2 re-scored without the flagged prompt.
- Cost table per GPU (T4/L4/A100) with price sources; audit GPU-minutes.
- Hand-label protocol, judge agreement table, prompt templates, BoN examples (redacted), compute, reproducibility checklist.

## Cut order if the main text still exceeds 5.7 pp
1. Fig 4 -> appendix (keep the two kappas in the text).
2. AM scatter -> appendix.
3. Related work -> 0.3 pp.
4. Cost table -> one sentence + appendix table (last resort only).

## Protected (never cut or shrunk below their caps)
- 4.3 Capability + Fig 3, 4.6 Multilingual + its figure: the organizer's main requests.
- 4.7 Cost table (tokens/s, $/1M tokens, ROI): explicitly requested by the organizer; only step 4 above touches it.
