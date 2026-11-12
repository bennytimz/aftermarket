"""A3 abliteration (PLAN.md §4).

Computes the refusal direction (SafeTune `steer` / difference of means over 64
contrast pairs) and projects it out of the write matrices. No training.
Abliterated checkpoints are never published (§9).
"""
