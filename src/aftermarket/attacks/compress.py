"""A4 compression (PLAN.md §4): HQQ 4/3/2-bit, group size 64, via transformers HqqConfig.

Full / template models load straight from the Hub with HqqConfig. LoRA (GC, IL) and deltas (ST)
are first built in fp16 on CPU (adapter merged / patches applied), written to a local temp dir,
reloaded with HqqConfig, and the temp dir is deleted. Those fp16 weights are the teams' own
released fixes, kept local and short-lived; nothing is pushed.

Records real memory: torch.cuda.memory_allocated after load and model.get_memory_footprint().

Compression shield (AM's deployment recipe, PLAN.md §6.4): a model whose spec has `quant_shield`
{bits, group_size, layers} keeps those decoder layers at `bits` (8) while every other layer follows the
attack's bits. transformers' HqqConfig only takes per-module-TYPE configs, so this path loads fp16 on the GPU
and swaps each nn.Linear for an hqq HQQLinear itself, with the same BaseQuantizeConfig (axis=1, as
transformers uses). lm_head stays unquantized, as with HqqConfig's default skip_modules.
"""

from __future__ import annotations

import re

def load_hqq(model_id: str, template: str, nbits: int, group_size: int = 64):
    """Return (model, tok, info) for `model_id` quantized with HQQ."""
    import torch
    from transformers import AutoModelForCausalLM, HqqConfig

    from aftermarket.load import finalize_model, get_spec, load_base_tokenizer, released_weights

    spec = get_spec(model_id)
    if spec.get("quant_shield"):
        return load_hqq_shielded(model_id, template, nbits, group_size, spec["quant_shield"])
    tok = load_base_tokenizer()
    q = HqqConfig(nbits=nbits, group_size=group_size)
    torch.cuda.reset_peak_memory_stats()
    before = torch.cuda.memory_allocated()
    with released_weights(model_id) as (src, rev):
        model = AutoModelForCausalLM.from_pretrained(src, revision=rev, dtype=torch.float16, quantization_config=q,
                                                     device_map={"": 0})
    route = "hub" if spec["format"] in ("full", "template") else "fp16-merged-then-quantized"
    model = finalize_model(model, model_id, spec, template, tok)
    info = {"attack": f"A4q{nbits}", "method": "hqq", "nbits": nbits, "group_size": group_size, "route": route,
            "vram_gb_model": round((torch.cuda.memory_allocated() - before) / 1e9, 3),
            "vram_gb_peak_load": round(torch.cuda.max_memory_allocated() / 1e9, 3),
            "footprint_gb": round(model.get_memory_footprint() / 1e9, 3)}
    return model, tok, info


def layer_bits(name: str, nbits: int, shield: dict) -> int | None:
    """Bits for one nn.Linear: shielded decoder layers -> shield bits, other decoder layers -> attack bits,
    anything outside the decoder layers (lm_head) -> None (unquantized)."""
    m = re.search(r"(?:^|\.)layers\.(\d+)\.", name)
    if not m:
        return None
    return shield["bits"] if int(m.group(1)) in set(shield["layers"]) else nbits


def load_hqq_shielded(model_id: str, template: str, nbits: int, group_size: int, shield: dict):
    import torch
    from hqq.core.quantize import BaseQuantizeConfig, HQQLinear

    from aftermarket.load import load_model

    torch.cuda.reset_peak_memory_stats()
    before = torch.cuda.memory_allocated()
    model, tok = load_model(model_id, precision="fp16", template=template)
    counts: dict[int, int] = {}
    for name, mod in list(model.named_modules()):
        if not isinstance(mod, torch.nn.Linear):
            continue
        bits = layer_bits(name, nbits, shield)
        if bits is None:
            continue
        cfg = BaseQuantizeConfig(nbits=bits, group_size=shield.get("group_size", group_size) if bits == shield["bits"]
                                 else group_size, view_as_float=False, axis=1)
        parent = model.get_submodule(name.rsplit(".", 1)[0])
        setattr(parent, name.rsplit(".", 1)[1],
                HQQLinear(mod, quant_config=cfg, compute_dtype=torch.float16, device="cuda", del_orig=True))
        counts[bits] = counts.get(bits, 0) + 1
    torch.cuda.empty_cache()
    info = {"attack": f"A4q{nbits}", "method": "hqq", "nbits": nbits, "group_size": group_size,
            "route": "fp16-then-hqq-per-layer (quant shield)", "quant_shield": shield,
            "linear_counts_by_bits": {str(k): v for k, v in sorted(counts.items())},
            "vram_gb_model": round((torch.cuda.memory_allocated() - before) / 1e9, 3),
            "vram_gb_peak_load": round(torch.cuda.max_memory_allocated() / 1e9, 3),
            "footprint_gb": round(model.get_memory_footprint() / 1e9, 3)}
    return model, tok, info
