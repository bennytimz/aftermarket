"""`aftermarket audit --smoke`: the whole audit pipeline on CPU in minutes, so errors show up locally, not after an
hour on Kaggle.

What is real: every process boundary (audit -> generate -> worker -> judge -> card), the registry hook, the test-split
guard, every attack's code path (A1/A2 LoRA training + merge, A3 direction fit + layer selection + weight edit,
A4q3 HQQ, A5raw / A5bon prompts), parquet + manifest writing, tables, the card and validate().
What is not: the model is a tiny random-weight Cohere2 (Tiny Aya's architecture, its real tokenizer and chat
template) standing in for BASE ($AFTERMARKET_BASE_OVERRIDE), the "fix" is a random LoRA adapter on it (GC's format:
adapter + tokenizer.json + chat_template.jinja), training uses 8 rows in float32 instead of 4-bit QLoRA (bitsandbytes
needs a GPU), each test split has 4 items, and the judges are keyword stubs. The numbers mean nothing.
"""

from __future__ import annotations

import copy
from pathlib import Path

import yaml

from aftermarket.data import SMOKE_SPLITS

SMOKE = {"attacks": ["A0", "A1", "A2", "A3", "A4q3", "A5raw", "A5bon"],
         "splits": ["harmful_s4_test", "mmlu_s4_test"], "refusal_split": "xstest_s4_test",
         "scope": "SMOKE TEST: tiny random model, 4 prompts per split, stub judges (numbers are meaningless)"}
assert set(SMOKE["splits"] + [SMOKE["refusal_split"]]) == set(SMOKE_SPLITS)
SMOKE_GEN = {"batch_size": 4, "max_new_tokens": {"default": 12, "mmlu": 4}}
TINY = {"num_hidden_layers": 5, "hidden_size": 64, "intermediate_size": 128, "num_attention_heads": 4,
        "num_key_value_heads": 2, "head_dim": 16, "sliding_window": 64}


def smoke_attacks(attacks_yaml: Path, out: Path) -> Path:
    """configs/attacks.yaml with every cost knob shrunk (same kinds, data, directions and selection rules)."""
    doc = yaml.safe_load(Path(attacks_yaml).read_text(encoding="utf-8"))
    doc = copy.deepcopy(doc)
    for a in doc["attacks"].values():
        if a["kind"] == "finetune":
            # max_len kept: shorter cuts every example inside Tiny Aya's long template preamble (no response tokens)
            a["train"] = {**a["train"], "max_rows": 8, "batch_size": 2, "grad_accum": 2,
                          "gradient_checkpointing": False}
        elif a["kind"] == "abliterate":
            a["select"] = {**a["select"], "n_harmful": 4, "n_benign": 4, "max_new_tokens": 4, "batch_size": 4}
        elif a.get("mode") == "bon":
            a["n"] = 2
    path = Path(out) / "attacks_smoke.yaml"
    path.write_text(yaml.safe_dump(doc, sort_keys=False), encoding="utf-8")
    return path


def tiny_config(base_cfg):
    """Tiny Aya's config, shrunk: same class, vocab, tokenizer ids, attention pattern (sliding / full)."""
    cfg = copy.deepcopy(base_cfg)
    for k, v in TINY.items():
        setattr(cfg, k, v)
    period = len(base_cfg.layer_types) and base_cfg.layer_types.index("full_attention") + 1
    cfg.layer_types = ["full_attention" if (i + 1) % period == 0 else "sliding_attention"
                       for i in range(TINY["num_hidden_layers"])]
    return cfg


def build_models(root: Path, base_id: str, base_revision: str, seed: int = 0) -> tuple[Path, Path]:
    """(tiny base folder, LoRA adapter folder) under root; rebuilt only if missing. Random weights, real tokenizer."""
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    base_dir, adapter_dir = Path(root) / "tiny-base", Path(root) / "tiny-fix-lora"
    if (base_dir / "config.json").exists() and (adapter_dir / "adapter_config.json").exists():
        return base_dir, adapter_dir
    tok = AutoTokenizer.from_pretrained(base_id, revision=base_revision)
    cfg = tiny_config(AutoConfig.from_pretrained(base_id, revision=base_revision))
    torch.manual_seed(seed)
    model = AutoModelForCausalLM.from_config(cfg, dtype=torch.float32)
    model.save_pretrained(base_dir)
    tok.save_pretrained(base_dir)
    lora = get_peft_model(model, LoraConfig(r=4, lora_alpha=8, target_modules=["q_proj", "v_proj"],
                                            task_type="CAUSAL_LM"))
    with torch.no_grad():  # PEFT starts B at zero (a no-op adapter): make the merge change something
        for n, p in lora.named_parameters():
            if "lora_B" in n:
                p.normal_(0, 0.02)
    lora.save_pretrained(adapter_dir)
    tok.save_pretrained(adapter_dir)  # like GC: the adapter ships tokenizer.json + chat_template.jinja (checked)
    return base_dir, adapter_dir
