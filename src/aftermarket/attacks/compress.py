"""A4 compression (PLAN.md §4): HQQ 4/3/2-bit, group size 64.

Every model is loaded in fp16 on the GPU with `load_model` (LoRA merged, deltas applied, template set), then
each decoder-layer nn.Linear is swapped for an hqq HQQLinear with BaseQuantizeConfig(axis=1, as transformers'
HqqConfig used). lm_head stays unquantized, as with HqqConfig's default skip_modules.

Why not `from_pretrained(quantization_config=HqqConfig(...))`: transformers 5.x raises
`NotImplementedError: QuantizationMethod.HQQ is not available yet` from its new loader (Kaggle run of
notebook 03, 2026-11). The per-layer swap gives the same quantization and needs no transformers support.
Merged LoRA / delta weights stay in GPU memory only; nothing is written to disk or pushed.

Records real memory: torch.cuda.memory_allocated after load and model.get_memory_footprint().

Compression shield (AM's deployment recipe, PLAN.md §6.4): a model whose spec has `quant_shield`
{bits, group_size, layers} keeps those decoder layers at `bits` (8) while every other layer follows the
attack's bits.
"""

from __future__ import annotations

import re


def layer_bits(name: str, nbits: int, shield: dict | None = None) -> int | None:
    """Bits for one nn.Linear: shielded decoder layers -> shield bits, other decoder layers -> attack bits,
    anything outside the decoder layers (lm_head) -> None (unquantized)."""
    m = re.search(r"(?:^|\.)layers\.(\d+)\.", name)
    if not m:
        return None
    if shield and int(m.group(1)) in set(shield["layers"]):
        return shield["bits"]
    return nbits


def quant_plan(linear_names: list[str], nbits: int, group_size: int = 64,
               shield: dict | None = None) -> dict[str, tuple[int, int]]:
    """{linear name: (bits, group_size)} for every nn.Linear to quantize; names left out stay fp16."""
    plan = {}
    for name in linear_names:
        bits = layer_bits(name, nbits, shield)
        if bits is None:
            continue
        gs = shield.get("group_size", group_size) if shield and bits == shield["bits"] else group_size
        plan[name] = (bits, gs)
    return plan


def load_hqq(model_id: str, template: str, nbits: int, group_size: int = 64):
    """Return (model, tok, info) for `model_id` quantized with HQQ (per-layer HQQLinear swap)."""
    import torch
    from hqq.core.quantize import BaseQuantizeConfig, HQQLinear

    from aftermarket.load import get_spec, load_model

    spec = get_spec(model_id)
    shield = spec.get("quant_shield")
    torch.cuda.reset_peak_memory_stats()
    before = torch.cuda.memory_allocated()
    model, tok = load_model(model_id, precision="fp16", template=template)
    names = [n for n, m in model.named_modules() if isinstance(m, torch.nn.Linear)]
    plan = quant_plan(names, nbits, group_size, shield)
    counts: dict[int, int] = {}
    for name, (bits, gs) in plan.items():
        parent_name, child = name.rsplit(".", 1)
        parent = model.get_submodule(parent_name)
        cfg = BaseQuantizeConfig(nbits=bits, group_size=gs, view_as_float=False, axis=1)
        setattr(parent, child, HQQLinear(getattr(parent, child), quant_config=cfg, compute_dtype=torch.float16,
                                         device="cuda", del_orig=True))
        counts[bits] = counts.get(bits, 0) + 1
    torch.cuda.empty_cache()
    info = {"attack": f"A4q{nbits}", "method": "hqq", "nbits": nbits, "group_size": group_size,
            "route": "fp16-then-hqq-per-layer" + (" (quant shield)" if shield else ""),
            "linear_counts_by_bits": {str(k): v for k, v in sorted(counts.items())},
            "vram_gb_model": round((torch.cuda.memory_allocated() - before) / 1e9, 3),
            "vram_gb_peak_load": round(torch.cuda.max_memory_allocated() / 1e9, 3),
            "footprint_gb": round(model.get_memory_footprint() / 1e9, 3)}
    if shield:
        info["quant_shield"] = shield
    return model, tok, info
