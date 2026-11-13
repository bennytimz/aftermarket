"""A4 compression (PLAN.md §4): HQQ 4/3/2-bit, group size 64, via transformers HqqConfig.

Full / template models load straight from the Hub with HqqConfig. LoRA (GC, IL) and deltas (ST)
are first built in fp16 on CPU (adapter merged / patches applied), written to a local temp dir,
reloaded with HqqConfig, and the temp dir is deleted. Those fp16 weights are the teams' own
released fixes, kept local and short-lived; nothing is pushed.

Records real memory: torch.cuda.memory_allocated after load and model.get_memory_footprint().
"""

from __future__ import annotations

import os
import shutil
import tempfile


def load_hqq(model_id: str, template: str, nbits: int, group_size: int = 64):
    """Return (model, tok, info) for `model_id` quantized with HQQ."""
    import torch
    from transformers import AutoModelForCausalLM, HqqConfig

    from aftermarket.load import (_check_repo_tokenizer, finalize_model, get_spec, load_base_tokenizer, load_model,
                                  load_registry)

    spec = get_spec(model_id)
    tok = load_base_tokenizer()
    q = HqqConfig(nbits=nbits, group_size=group_size)
    kw = {"dtype": torch.float16, "quantization_config": q, "device_map": {"": 0}}
    torch.cuda.reset_peak_memory_stats()
    before = torch.cuda.memory_allocated()
    if spec["format"] in ("full", "template"):
        _check_repo_tokenizer(spec, tok)
        if spec["format"] == "template":
            reg = load_registry()
            src, rev = reg["base_model"], reg["base_revision"]
        else:
            src, rev = spec["hf_path"], spec["revision"]
        model = AutoModelForCausalLM.from_pretrained(src, revision=rev, **kw)
        route = "hub"
    else:
        dense, _ = load_model(model_id, precision="fp16", template=template, device="cpu")
        tmp = tempfile.mkdtemp(prefix="am_hqq_", dir=os.environ.get("AFTERMARKET_TMP"))
        try:
            dense.save_pretrained(tmp)
            del dense
            model = AutoModelForCausalLM.from_pretrained(tmp, **kw)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
        route = "fp16-merged-then-quantized"
    model = finalize_model(model, model_id, spec, template, tok)
    info = {"attack": f"A4q{nbits}", "method": "hqq", "nbits": nbits, "group_size": group_size, "route": route,
            "vram_gb_model": round((torch.cuda.memory_allocated() - before) / 1e9, 3),
            "vram_gb_peak_load": round(torch.cuda.max_memory_allocated() / 1e9, 3),
            "footprint_gb": round(model.get_memory_footprint() / 1e9, 3)}
    return model, tok, info
