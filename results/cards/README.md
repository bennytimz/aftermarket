# Report cards

One page per leaderboard run (held-out test prompts). Make one for your own model with `aftermarket audit --model <hf id or path>`.

| model | retention | rank | worst attack | over-refusal | MMLU |
|---|---|---|---|---|---|
| [BASE](BASE.md) | 0.59 | 5 of 9 | A5bon 95% | 9% | 53% |
| [SDT](SDT.md) | 0.70 | 3 of 9 | A5bon 96% | 2% | 54% |
| [GC](GC.md) | 0.74 | 1 of 9 | A2 82% | 26% | 54% |
| [IL](IL.md) | 0.58 | 7 of 9 | A5bon 94% | 10% | 54% |
| [RL](RL.md) | 0.73 | 2 of 9 | A2 81% | 18% | 53% |
| [RL_noguard](RL_noguard.md) | 0.59 | 5 of 9 | A5bon 95% | 9% | 53% |
| [ST](ST.md) | 0.61 | 4 of 9 | A5bon 96% | 21% | 47% |
| [SP](SP.md) | 0.54 | 8 of 9 | A5bon 96% | 10% | 53% |
| [AM](AM.md) | 0.53 | 9 of 9 | A2 86% | 7% | 52% |
| [AM_rand0](AM_rand0.md) | 0.19 | not ranked | A2 88% | 11% | 53% |
| [AM_rand1](AM_rand1.md) | 0.56 | not ranked | A2 88% | 9% | 54% |
| [AM_all](AM_all.md) | 0.56 | not ranked | A2 88% | 11% | 54% |
