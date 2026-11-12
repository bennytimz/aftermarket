"""Judging step (PLAN.md §5).

Llama-Guard-3-1B as primary judge, WildGuard on a subset if VRAM allows, plus a
keyword refusal check. Runs in its own process, reading generation parquet.
"""
