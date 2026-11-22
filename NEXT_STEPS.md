# Next steps (Stage 2), updated 22 Nov 2026, 14:25 UTC

Deadline: Tue 24 Nov AoE (Wed 25 Nov 12:59 Lagos). Target: paper finished Tue 24 Nov night (Lagos).
Plan: `STAGE2_PLAN.md`. Page budget: `paper/PAGE_BUDGET.md`. Branch `stage2` only; nothing merged to main.

## Where we are
- **Kaggle session 1** (done): 12 A0 + 12 A2 cells, all rc 0. Real T4 minutes: A0 ~43 (est 30), A2 ~60 (est 50);
  RL is slowest (71 / 87, its guard template lengthens every prompt), GC fastest (19 / 28).
- **HF Jobs work** under `Aftermarket-lexsi` (token `aftermarket-jobs`). Image fixes: no git (tarball of the pushed
  commit, sha via `AFTERMARKET_GIT_SHA`), `python3` only, `sentencepiece` for WildGuard's tokenizer.
- **L4 vs T4**: BASE A0 on L4 (`BASE__A0__l4`, a replicate kept out of the main tables) agrees with T4 on 99.7%
  of capability items; an L4 cell takes ~19 min (~2.6x faster than T4).
- **Paper**: 4.3 capability drafted with Fig 3 and an appendix table from the A0/A2 cells (`make fetch assets pdf`).

## Running (22 Nov, 14:25 UTC)
| What | Where | Expected done (UTC) |
|---|---|---|
| A3: BASE, SDT done; GC, IL running | Kaggle session 2 | ~15:15. **Then stop the Kaggle session** (it would rerun cells HF did) |
| A3: RL, RL_noguard, ST, SP, AM, AM_rand0/1, AM_all | HF l4x1 | ~14:55 (RL ~15:05) |
| A4q3: all 12 (BASE canary done) | HF l4x1 | ~14:45 (RL ~15:00) |
| A1: all 12 | HF l4x1 | ~14:55 (RL ~15:10) |
| Judging canary BASE__A0, then A0/A2 in 4 jobs | HF l4x1 | ~15:00, then ~16:30 |
| Judging A3 / A4q3 / A1 (ask first) | HF l4x1 | ~17:30 |
| E2 throughput: L4, A100, T4 | HF l4x1 / a100-large / t4-small | ~17:40 / ~16:30 / ~19:30 |

## Next
1. Judge A3/A4q3/A1 once their cells land (ask before launch).
2. Paper: A3/A4q3/A1 columns in Fig 3; 4.6 multilingual (judged labels); 4.7 cost (E2 + `configs/gpu_prices.yaml`);
   4.1 A2 replication ASR; takeaways, conclusion.
3. E5 seeds (A2s1/A2s2 for BASE, GC, AM) and E7 audit timing (quick audit on Kaggle with the freed quota).
4. Review the Yoruba check file (user).
