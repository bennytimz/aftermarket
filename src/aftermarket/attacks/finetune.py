"""A1 benign fine-tune and A2 small harmful fine-tune (PLAN.md §4, §13.5).

Per (model run, attack), once:
  1. QLoRA-train a LoRA adapter on the model AS RELEASED (load.released_weights: GC/IL adapter merged,
     ST deltas applied, RL = BASE weights) loaded in 4-bit nf4, gradient checkpointing, fp16 autocast.
     Hyperparameters come only from configs/attacks.yaml (`train`, shared by A1 and A2), so they are
     identical for every model. Loss on response tokens only.
  2. Save the adapter to a LOCAL temp dir (ADAPTER_ROOT, outside the repo and results/), load the released
     model in fp16, merge the adapter -> eval model (same precision as A0).
  3. The worker generates every split, then calls the returned cleanup(), which deletes the adapter dir.
Adapters are never pushed: generate.push_files refuses anything but .parquet/.json and anything under
ADAPTER_ROOT (PLAN.md §9).

Chat template for training: the model's own (native), except RL, whose guard is a prompt-level fix the
attacker would not train with: RL trains with BASE's template and is evaluated with its guard template.
"""

from __future__ import annotations

import json
import math
import os
import random
import shutil
import tempfile
import time
from pathlib import Path

ADAPTER_ROOT = Path(os.environ.get("AFTERMARKET_ADAPTERS", Path(tempfile.gettempdir()) / "aftermarket_adapters"))


# ----------------------------------------------------------------------------- config (no torch)

TRAIN_KEYS = ["quant", "lora_r", "lora_alpha", "lora_dropout", "target_modules", "lr", "epochs", "max_len",
              "batch_size", "grad_accum", "warmup_ratio", "scheduler", "max_grad_norm", "weight_decay",
              "gradient_checkpointing", "loss_on", "seed"]


def train_config(attack: dict) -> dict:
    """Training hyperparameters: a function of the attack config only (never of the model)."""
    t = attack["train"]
    missing = [k for k in TRAIN_KEYS if k not in t]
    if missing:
        raise KeyError(f"train config missing {missing}")
    return {k: t[k] for k in TRAIN_KEYS}


def config_hash(tc: dict) -> str:
    import hashlib

    return hashlib.sha256(json.dumps(tc, sort_keys=True).encode()).hexdigest()[:16]


def training_plan(run_id: str, attack: dict, registry: dict | None = None) -> dict:
    """What gets trained for this run: identical hparams + data for every run; only the chat template differs."""
    from aftermarket.load import load_registry, resolve_run

    reg = registry or load_registry()
    model_id, _ = resolve_run(run_id, reg)
    fmt = reg["models"][model_id]["format"]
    return {"model_id": model_id, "hparams": train_config(attack), "data": attack["data"],
            "train_template": "base" if fmt == "template" else "native"}


def adapter_dir(run_id: str, attack_id: str) -> Path:
    from aftermarket.generate import REPO_ROOT

    d = ADAPTER_ROOT / f"{run_id}__{attack_id}"
    if (REPO_ROOT / "results") in d.resolve().parents:
        raise RuntimeError("adapter dir must never be inside results/ (pushed area)")
    return d


# ----------------------------------------------------------------------------- data

def encode_example(tok, chat_template: str | None, prompt: str, response: str, max_len: int):
    """(input_ids, labels): prompt tokens masked (-100), response tokens learned; truncated to max_len."""
    user = [{"role": "user", "content": prompt}]
    p = tok.apply_chat_template(user, chat_template=chat_template, tokenize=False, add_generation_prompt=True)
    full = tok.apply_chat_template(user + [{"role": "assistant", "content": response}], chat_template=chat_template,
                                   tokenize=False)
    pi = tok(p, add_special_tokens=False)["input_ids"]
    fi = tok(full, add_special_tokens=False)["input_ids"]
    n = 0  # common prefix: robust to templates that render the generation prompt slightly differently
    while n < min(len(pi), len(fi)) and pi[n] == fi[n]:
        n += 1
    labels = [-100] * n + fi[n:]
    return fi[:max_len], labels[:max_len], len(fi) > max_len


# ----------------------------------------------------------------------------- training (torch)

def train_adapter(model_id: str, train_template: str, rows: list[dict], tc: dict, out_dir: Path) -> dict:
    import numpy as np
    import torch
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig, get_linear_schedule_with_warmup

    from aftermarket.load import _chat_template, get_spec, load_base_tokenizer, released_weights

    random.seed(tc["seed"]); np.random.seed(tc["seed"]); torch.manual_seed(tc["seed"])
    t0 = time.time()
    tok = load_base_tokenizer()
    ct = _chat_template(get_spec(model_id), train_template)
    q = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type=tc["quant"], bnb_4bit_compute_dtype=torch.float16)
    with released_weights(model_id) as (src, rev):
        model = AutoModelForCausalLM.from_pretrained(src, revision=rev, quantization_config=q, dtype=torch.float16,
                                                     device_map={"": 0})
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=tc["gradient_checkpointing"])
    model = get_peft_model(model, LoraConfig(r=tc["lora_r"], lora_alpha=tc["lora_alpha"], lora_dropout=tc["lora_dropout"],
                                             target_modules=tc["target_modules"], task_type="CAUSAL_LM"))
    enc = [encode_example(tok, ct, r["prompt"], r["response"], tc["max_len"]) for r in rows]
    truncated = sum(e[2] for e in enc)
    bs, acc = tc["batch_size"], tc["grad_accum"]
    order = list(range(len(enc)))
    random.Random(tc["seed"]).shuffle(order)
    n_steps = math.ceil(len(enc) * tc["epochs"] / (bs * acc))
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=tc["lr"], weight_decay=tc["weight_decay"])
    sched = get_linear_schedule_with_warmup(opt, int(tc["warmup_ratio"] * n_steps), n_steps)
    scaler = torch.amp.GradScaler("cuda")
    pad = tok.pad_token_id
    model.train()
    losses, micro, n_tok = [], 0, 0
    for _ in range(tc["epochs"]):
        for i in range(0, len(order), bs):
            batch = [enc[k] for k in order[i:i + bs]]
            L = max(len(b[0]) for b in batch)
            ids = torch.tensor([b[0] + [pad] * (L - len(b[0])) for b in batch], device=model.device)
            lab = torch.tensor([b[1] + [-100] * (L - len(b[1])) for b in batch], device=model.device)
            att = torch.tensor([[1] * len(b[0]) + [0] * (L - len(b[0])) for b in batch], device=model.device)
            with torch.autocast("cuda", dtype=torch.float16):
                loss = model(input_ids=ids, attention_mask=att, labels=lab).loss
            scaler.scale(loss / acc).backward()
            losses.append(float(loss))
            n_tok += int((lab != -100).sum())
            micro += 1
            if micro % acc == 0 or i + bs >= len(order):
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(params, tc["max_grad_norm"])
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)
                sched.step()
    model.save_pretrained(out_dir)  # adapter weights only, local temp dir
    secs = time.time() - t0
    info = {"train_seconds": round(secs, 1), "n_examples": len(enc), "optimizer_steps": n_steps,
            "truncated_examples": truncated, "response_tokens_trained": n_tok,
            "loss_first10": round(sum(losses[:10]) / max(1, len(losses[:10])), 4),
            "loss_last10": round(sum(losses[-10:]) / max(1, len(losses[-10:])), 4),
            "peak_vram_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2)}
    del model, opt
    return info


def finetune_and_load(run_id: str, attack_id: str, attack: dict, cfg: dict):
    """Train (once per run x attack), then return (fp16 merged eval model, tok, info, cleanup)."""
    import gc

    import torch
    from peft import PeftModel

    from aftermarket.ft_data import load_ft
    from aftermarket.load import finalize_model, get_spec, load_model, resolve_run

    plan = training_plan(run_id, attack)
    model_id, template = resolve_run(run_id)
    tc = plan["hparams"]
    d = adapter_dir(run_id, attack_id)
    shutil.rmtree(d, ignore_errors=True)
    d.mkdir(parents=True)

    def cleanup():
        shutil.rmtree(d, ignore_errors=True)

    try:
        torch.cuda.reset_peak_memory_stats()
        info = train_adapter(model_id, plan["train_template"], load_ft(plan["data"]), tc, d)
        gc.collect(); torch.cuda.empty_cache()
        print(f"[{run_id}/{attack_id}] trained in {info['train_seconds']}s, loss {info['loss_first10']} -> "
              f"{info['loss_last10']}, peak {info['peak_vram_gb']} GB", flush=True)
        model, tok = load_model(model_id, precision=cfg["precision"], template=template)
        model = PeftModel.from_pretrained(model, str(d)).merge_and_unload()
        model = finalize_model(model, model_id, get_spec(model_id), template, tok)
    except BaseException:
        cleanup()
        raise
    info.update({"attack": attack_id, "kind": "finetune", "data": plan["data"], "train_template": plan["train_template"],
                 "train_config": tc, "train_config_hash": config_hash(tc), "adapter_location": "local temp, deleted"})
    return model, tok, info, cleanup
