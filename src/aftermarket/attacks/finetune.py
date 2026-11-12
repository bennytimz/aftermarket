"""A1 benign fine-tune and A2 small harmful fine-tune (PLAN.md §4).

LoRA r=16, 1 epoch, fixed seed. A1: 1,000 Alpaca examples. A2: 100 harmful +
900 benign. Adapters are saved to a private local dir only and never published (§9).
"""
