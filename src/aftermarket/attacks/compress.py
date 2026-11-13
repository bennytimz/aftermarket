"""A4 compression (PLAN.md §4): HQQ 4/3/2-bit, group size 64, via transformers HqqConfig.

Full / template models load straight from the Hub with HqqConfig. LoRA (GC, IL) and deltas (ST)
are first built in fp16 on CPU (adapter merged / patches applied), written to a local temp dir,
reloaded with HqqConfig, and the temp dir is deleted. Those fp16 weights are the teams' own
released fixes, kept local and short-lived; nothing is pushed.

Records real memory: torch.cuda.memory_allocated after load and model.get_memory_footprint().
"""

from __future__ import annotations

def load_hqq(model_id: str, template: str, nbits: int, group_size: int = 64):
    """Return (model, tok, info) for `model_id` quantized with HQQ."""
    import torch
    from transformers import AutoModelForCausalLM, HqqConfig

    from aftermarket.load import finalize_model, get_spec, load_base_tokenizer, released_weights

    spec = get_spec(model_id)
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
