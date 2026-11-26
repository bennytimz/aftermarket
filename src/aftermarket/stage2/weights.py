"""Build one (run, attack) model and save it as an fp16 HF checkpoint that vLLM / lm-eval can load.

    PYTHONPATH=src python -m aftermarket.stage2.weights --run BASE --attack A2 --out w/BASE__A2

Reuses the Stage 1 code path for every attack (aftermarket.generate.prepare_model): A1/A2 retrain the QLoRA
adapter with the one shared training block (seed / data / hyperparameters from configs/attacks.yaml) and merge it
into fp16; A3 runs the v2 per-model layer selection on dev and edits the weights.

A4q3 ("fake quantization"): the model is quantized with exactly Stage 1's HQQ plan (compress.load_hqq, 3-bit,
group 64, AM's 8-bit shield included), then every HQQLinear is replaced by an nn.Linear holding
HQQLinear.dequantize(). HQQ's PyTorch backend computes matmul(x, dequantize().T) + bias, so the saved fp16 model
computes the same function; it is what vLLM (which cannot load HQQ) evaluates. Throughput of real 3-bit
inference is measured separately on the HQQ backend (E2).

The saved directory holds attacked weights: it lives on the job's local disk only, is never pushed, and the cell
job deletes it when it ends (PLAN.md §9). Only the info JSON (no weights) leaves the machine.
"""

from __future__ import annotations

import argparse
import copy
import json
import time
from pathlib import Path


def stage2_attack(attack_id: str) -> tuple[str, dict]:
    """(Stage 1 attack id, attack dict) for a Stage 2 attack id. 'A2s1' = A2 with training seed 1 (E5)."""
    from aftermarket.generate import load_attacks

    attacks = load_attacks()
    if attack_id in attacks:
        return attack_id, attacks[attack_id]
    if attack_id.startswith("A2s") and attack_id[3:].isdigit():
        a = copy.deepcopy(attacks["A2"])
        a["train"]["seed"] = int(attack_id[3:])
        return "A2", a
    raise KeyError(f"unknown attack {attack_id!r}")


def dequantize_hqq_(model) -> int:
    """Replace every HQQLinear in `model` with an fp16 nn.Linear holding its dequantized weight. Returns count."""
    import torch
    from hqq.core.quantize import HQQLinear

    names = [n for n, m in model.named_modules() if isinstance(m, HQQLinear)]
    for name in names:
        parent_name, child = name.rsplit(".", 1)
        parent = model.get_submodule(parent_name)
        q = getattr(parent, child)
        W = q.dequantize().to(torch.float16)
        lin = torch.nn.Linear(W.shape[1], W.shape[0], bias=q.bias is not None, device=W.device, dtype=torch.float16)
        with torch.no_grad():
            lin.weight.copy_(W)
            if q.bias is not None:
                lin.bias.copy_(q.bias.to(torch.float16))
        setattr(parent, child, lin)
        del q
    return len(names)


def save_for_vllm(model, tok, out: Path) -> None:
    """fp16 safetensors + the BASE tokenizer carrying this run's chat template (what vLLM and lm-eval apply)."""
    import torch

    out.mkdir(parents=True, exist_ok=True)
    model = model.to(torch.float16)
    model.save_pretrained(out, safe_serialization=True)
    tmpl = getattr(model, "aftermarket_chat_template", None)
    if tmpl:
        tok.chat_template = tmpl
    tok.save_pretrained(out)


def prepare(run_id: str, attack_id: str, out: Path, cfg: dict | None = None) -> dict:
    from aftermarket import runtime
    from aftermarket.generate import prepare_model

    cfg = cfg or {"precision": "fp16", "seed": 42}
    base_attack, attack = stage2_attack(attack_id)
    t0 = time.time()
    runtime.reset_peak()
    model, tok, info = prepare_model(run_id, base_attack, attack, cfg)
    cleanup = info.pop("_cleanup", None)
    try:
        if attack["kind"] == "hqq":
            info["fake_quant_linears"] = dequantize_hqq_(model)
            info["route"] = info.get("route", "") + " -> dequantized to fp16 (fake quant, for vLLM)"
        info["prepare_seconds"] = round(time.time() - t0, 1)
        t1 = time.time()
        save_for_vllm(model, tok, out)
        info["save_seconds"] = round(time.time() - t1, 1)
    finally:
        if cleanup:
            cleanup()
    info.update({"run_id": run_id, "attack_id": attack_id, "base_attack": base_attack,
                 "peak_vram_gb": round(runtime.peak() / 1e9, 2)})
    return {k: v for k, v in info.items() if not k.startswith("_")}


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", required=True)
    ap.add_argument("--attack", required=True)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--info", type=Path, help="write the attack info JSON here")
    args = ap.parse_args(argv)
    info = prepare(args.run, args.attack, args.out)
    text = json.dumps(info, indent=1, default=str)
    print(text)
    if args.info:
        args.info.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
