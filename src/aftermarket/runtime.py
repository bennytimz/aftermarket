"""Where models run. GPU (CUDA) everywhere by default; `AFTERMARKET_DEVICE=cpu` (set by `aftermarket audit --smoke`)
runs the same code on CPU in float32, so the whole pipeline can be exercised locally before a Kaggle run.
Memory statistics are CUDA-only and read as 0 on CPU."""

from __future__ import annotations

import os

DEVICE_ENV = "AFTERMARKET_DEVICE"


def on_cpu() -> bool:
    return os.environ.get(DEVICE_ENV, "").lower() == "cpu"


def device() -> str:
    return "cpu" if on_cpu() else "cuda"


def _cuda():
    """torch.cuda on GPU runs (exactly the calls the code made before smoke mode existed), None on CPU runs."""
    if on_cpu():
        return None
    import torch

    return torch.cuda


def reset_peak() -> None:
    c = _cuda()
    if c:
        c.reset_peak_memory_stats()


def allocated() -> int:
    c = _cuda()
    return c.memory_allocated() if c else 0


def peak() -> int:
    c = _cuda()
    return c.max_memory_allocated() if c else 0


def empty_cache() -> None:
    c = _cuda()
    if c:
        c.empty_cache()
