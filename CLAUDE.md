# Aftermarket — rules for Claude Code
- Read PLAN.md first. Don't change the experiment design (§4–§6) without asking.
- Python 3.12, pinned requirements. Base: CohereLabs/tiny-aya-global @ af89d219b53ed9b13b8a4645f9c8028973510324.
- Kaggle 2xT4 / Colab T4 only: never assume >16 GB per GPU. Model and judge run in separate processes.
- Every run: --config yaml, output to results/<run_id>/ with manifest (git sha, config, seed, time). Push parquet to HF after each run.
- Same seed, hyperparameters and data for every model x attack cell. Never tune on data/splits/test; it's touched once with --final.
- Never push attack-tuned weights or unredacted harmful completions anywhere public.
- Pin the HF tokenizer (Kaggle's Tiny Aya copy differs) and assert token IDs match.
- When a Lexsi API is unclear, read its source in site-packages. Log real gaps/bugs in notes/upstream_issues.md.
- Small functions, fast unit tests for loaders, splits and metrics.
