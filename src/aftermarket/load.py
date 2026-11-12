"""Unified loader for every model in configs/models.yaml (PLAN.md §3, §8).

Will handle full HF models, LoRA adapters on BASE (GC, IL), parameter deltas
(ST, via the team's loader) and GGUF files via llama-cpp-python (RL, with and
without its template). Pins BASE revision and tokenizer and asserts token IDs match.
"""
