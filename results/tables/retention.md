| model | rank | retention on A1, A2, A3, A4, A5 [95% CI] | WildGuard judge (post-hoc sensitivity, not ranked) | vs BASE (paired, shared attacks) | note | prompts |
|---|---|---|---|---|---|---|
| GC | 1 | 0.74 [0.71–0.76] | 0.78 [0.76–0.79] | +0.14 on A1, A2, A3, A4, A5 (p_holm 0.001) |  | 200 |
| RL | 2 | 0.73 [0.71–0.76] | 0.77 [0.76–0.79] | +0.14 on A1, A2, A3, A4, A5 (p_holm 0.001) |  | 200 |
| SDT | 3 | 0.70 [0.67–0.73] | 0.71 [0.68–0.74] | +0.11 on A1, A2, A3, A4, A5 (p_holm 0.001) |  | 200 |
| ST | 4 | 0.61 [0.57–0.65] | 0.64 [0.61–0.68] | +0.02 on A1, A2, A3, A4, A5 (p_holm 1.000) |  | 200 |
| BASE | 5 | 0.59 [0.56–0.63] | 0.64 [0.61–0.68] |  |  | 200 |
| RL_noguard | 6 | 0.59 [0.56–0.63] | 0.64 [0.61–0.68] | +0.00 on A1, A2, A3, A4, A5 (p_holm 1.000) |  | 200 |
| IL | 7 | 0.58 [0.55–0.61] | 0.65 [0.62–0.68] | -0.01 on A1, A2, A3, A4, A5 (p_holm 1.000) |  | 200 |
| SP | 8 | 0.54 [0.51–0.57] | 0.59 [0.56–0.62] | -0.05 on A1, A2, A3, A4, A5 (p_holm 0.006) |  | 200 |
| AM | 9 | 0.53 [0.50–0.56] | 0.53 [0.51–0.55] | -0.06 on A1, A2, A3, A4, A5 (p_holm 0.028) |  | 200 |
| AM_all | not ranked | 0.56 [0.54–0.58] | 0.54 [0.52–0.56] | +0.23 on A2, A3 (p_holm 0.001) | own: A2, A3 (control, partial) | 200 |
| AM_rand1 | not ranked | 0.56 [0.54–0.59] | 0.54 [0.52–0.56] | +0.23 on A2, A3 (p_holm 0.001) | own: A2, A3 (control, partial) | 200 |
| AM_rand0 | not ranked | 0.19 [0.15–0.23] | 0.10 [0.07–0.13] | -0.14 on A2, A3 (p_holm 0.001) | own: A2, A3 (control, partial) | 200 |
