"""Generation step (PLAN.md §5).

Runs model x attack x prompts with greedy decoding, max 256 new tokens and each
model's official chat template (except A5), writing outputs to parquet under
results/<run_id>/ with a manifest. Judging happens later in a separate process.
"""
