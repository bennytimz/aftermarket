"""Build AM candidates and controls (PLAN.md §6.2-6.5, steps b-e).

    PYTHONPATH=src python -m aftermarket.defense.harden --config configs/defense.yaml --stage candidates [--shard 0/2]
    PYTHONPATH=src python -m aftermarket.defense.harden --config configs/defense.yaml --stage resta
    PYTHONPATH=src python -m aftermarket.defense.harden --config configs/defense.yaml --stage controls [--shard 0/2]

(b) harden: SafeTune SafeGradHFTrainer (gradient surgery, Yi et al. 2508.07172) on BASE in 4-bit (QLoRA),
    LoRA restricted to the protected layers. User task = 300 benign Alpaca pairs; alignment batches = 300
    harmful -> refusal pairs (SFT-loss alignment signal: a frozen fp16 reference does not fit next to the
    model on a T4). Adapter merged into fp16 at eval, like the A1/A2 attacks.
(c) spread-refusal: on every `every`-th alignment batch (half), BASE's top refusal direction is projected out
    of the residual stream at every decoder layer (input and output), so refusal is learned without it.
(d) shield: recorded as `quant_shield` (L* stay 8-bit under the A4 attack; attacks/compress.py applies it).
(e) ReSta: SafeTune ReStaTrainer with base=BASE, aligned=candidate, on the candidate itself, i.e. the
    hardening delta re-applied at alpha (DARE on the zero fine-tune delta is a no-op). Kept only if it helps
    on dev (select.py stage 2). Output is dense fp16 weights.
Controls use the frozen recipe with the same data, steps, lr, seed and trainable parameter count:
AM_rand{s} = k random layers outside L*; AM_all = every layer with the same total LoRA rank spread over layers.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import re
import time
from contextlib import contextmanager
from pathlib import Path

import yaml

# ----------------------------------------------------------------------------- ids and plans (pure)

def candidate_id(k: int, spread: bool) -> str:
    return f"AMc_k{k}" + ("s" if spread else "")


def candidate_grid(dcfg: dict) -> list[dict]:
    c = dcfg["candidates"]
    return [{"id": candidate_id(k, s), "k": k, "spread": s, "shield": c["shield"]} for k in c["k"] for s in c["spread"]]


def rank_plan(layers: list[int] | None, n_layers: int, r: int, k: int | None = None) -> dict[int, int]:
    """LoRA rank per layer. Restricted: rank r on `layers`. All layers (layers=None): the same total rank r*k
    spread over every layer (floor + 1 on evenly spaced layers), so trainable parameters match exactly."""
    if layers is not None:
        return {layer: r for layer in sorted(layers)}
    budget = r * k
    base, extra = divmod(budget, n_layers)
    if base < 1:
        raise ValueError(f"rank budget {budget} < {n_layers} layers: cannot cover every layer")
    bump = {int(i * n_layers / extra) for i in range(extra)}
    return {layer: base + (layer in bump) for layer in range(n_layers)}


def lora_param_count(plan: dict[int, int], module_shapes: dict[str, tuple[int, int]]) -> int:
    """Trainable LoRA parameters: rank * (in + out) per target module, summed over planned layers."""
    per_rank = sum(i + o for i, o in module_shapes.values())
    return sum(r * per_rank for r in plan.values())


def lora_kwargs(plan: dict[int, int], hc: dict) -> dict:
    """LoraConfig kwargs for a rank plan (alpha/r ratio fixed, so every layer has the same LoRA scaling)."""
    ratio = hc["lora_alpha"] / hc["lora_r"]
    ranks = set(plan.values())
    kw = {"target_modules": hc["target_modules"], "lora_dropout": hc["lora_dropout"], "task_type": "CAUSAL_LM",
          "layers_to_transform": sorted(plan)}
    if len(ranks) == 1:
        r = ranks.pop()
        return {**kw, "r": r, "lora_alpha": ratio * r}
    r0 = min(plan.values())
    pat = {rf"layers\.{layer}\.(self_attn|mlp)\.\w+": rr for layer, rr in plan.items() if rr != r0}
    return {**kw, "r": r0, "lora_alpha": ratio * r0, "rank_pattern": pat,
            "alpha_pattern": {key: ratio * rr for key, rr in pat.items()}}


# ----------------------------------------------------------------------------- spread-refusal hook

def project_out(h, d):
    """Remove the component of h along unit vector d (last axis)."""
    if hasattr(d, "to") and hasattr(h, "dtype") and hasattr(h, "device"):
        d = d.to(device=h.device, dtype=h.dtype)
    return h - (h @ d)[..., None] * d


def _ablate_first(x, d):
    if isinstance(x, tuple):
        return (project_out(x[0], d),) + tuple(x[1:])
    return project_out(x, d)


@contextmanager
def ablate_direction(layers, d):
    """Directional ablation (Arditi et al.): project d out of every decoder layer's input and output."""
    def pre(_m, args, kwargs):
        if args:
            return (_ablate_first(args[0], d),) + tuple(args[1:]), kwargs
        if "hidden_states" in kwargs:
            return args, {**kwargs, "hidden_states": project_out(kwargs["hidden_states"], d)}
        return None

    def post(_m, _args, out):
        return _ablate_first(out, d)

    handles = []
    try:
        for layer in layers:
            handles.append(layer.register_forward_pre_hook(pre, with_kwargs=True))
            handles.append(layer.register_forward_hook(post))
        yield
    finally:
        for h in handles:
            h.remove()


def spread_schedule(n_batches: int, every: int) -> list[bool]:
    """Which alignment batches run with the direction ablated (1-based every-th batch)."""
    return [(i + 1) % every == 0 for i in range(n_batches)]


# ----------------------------------------------------------------------------- training (torch)

def _decoder_layers(model):
    import torch

    for _, m in model.named_modules():
        if isinstance(getattr(m, "layers", None), torch.nn.ModuleList):
            return list(m.layers)
    raise RuntimeError("decoder layers not found")


def _collate(pad_id: int):
    import torch

    def fn(batch):
        L = max(len(b["input_ids"]) for b in batch)
        return {"input_ids": torch.tensor([b["input_ids"] + [pad_id] * (L - len(b["input_ids"])) for b in batch]),
                "labels": torch.tensor([b["labels"] + [-100] * (L - len(b["labels"])) for b in batch]),
                "attention_mask": torch.tensor([[1] * len(b["input_ids"]) + [0] * (L - len(b["input_ids"]))
                                                for b in batch])}
    return fn


def train_defense(model_id: str, plan: dict[int, int], spread: bool, dcfg: dict, meta_extra: dict) -> dict:
    """Train one defense adapter into the local store folder `model_id`. Returns its metadata."""
    import numpy as np
    import torch
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from safetune.interventions.harden.safegrad import SafeGradHFTrainer
    from torch.utils.data import DataLoader
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig, TrainingArguments

    from aftermarket.attacks.finetune import encode_example
    from aftermarket.defense import store
    from aftermarket.ft_data import load_ft
    from aftermarket.load import load_base_tokenizer, load_registry

    hc = dcfg["harden"]
    seed = hc["seed"]
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    t0 = time.time()
    reg = load_registry()
    tok = load_base_tokenizer()
    q = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type=hc["quant"], bnb_4bit_compute_dtype=torch.float16)
    model = AutoModelForCausalLM.from_pretrained(reg["base_model"], revision=reg["base_revision"], quantization_config=q,
                                                 dtype=torch.float16, device_map={"": 0})
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=hc["gradient_checkpointing"],
                                            gradient_checkpointing_kwargs={"use_reentrant": False})
    model = get_peft_model(model, LoraConfig(**lora_kwargs(plan, hc)))
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    shapes = {m: v for m, v in module_shapes(model).items() if m in hc["target_modules"]}
    expected = lora_param_count(plan, shapes)
    if n_train != expected:
        raise AssertionError(f"{model_id}: {n_train} trainable params != planned {expected}")

    rows = load_ft(hc["data"])
    enc = lambda r: dict(zip(("input_ids", "labels"), encode_example(tok, None, r["prompt"], r["response"], hc["max_len"])[:2]))
    benign = [enc(r) for r in rows if r["kind"] == "benign"]
    refusals = [enc(r) for r in rows if r["kind"] == "refusal"]
    collate = _collate(tok.pad_token_id)
    g = torch.Generator().manual_seed(seed)
    safety_loader = DataLoader(refusals, batch_size=hc["batch_size"], shuffle=True, generator=g, collate_fn=collate)

    direction = None
    if spread:
        direction = torch.from_numpy(np.load(store.defense_dir() / "locate" / "direction.npy"))
    layers = _decoder_layers(model)

    class SpreadSafeGrad(SafeGradHFTrainer):
        n_align = n_ablated = n_skipped = 0

        def _alignment_gradient(self, m, batch):
            type(self).n_align += 1
            if direction is not None and type(self).n_align % hc["spread_refusal"]["every"] == 0:
                type(self).n_ablated += 1
                with ablate_direction(layers, direction):
                    out = super()._alignment_gradient(m, batch)
            else:
                out = super()._alignment_gradient(m, batch)
            if out is None:
                type(self).n_skipped += 1
            return out

    out_dir = store.model_dir(model_id)
    out_dir.mkdir(parents=True, exist_ok=True)
    args = TrainingArguments(
        output_dir=str(out_dir / "_trainer"), per_device_train_batch_size=hc["batch_size"],
        gradient_accumulation_steps=hc["grad_accum"], num_train_epochs=hc["epochs"], learning_rate=hc["lr"],
        warmup_ratio=hc["warmup_ratio"], lr_scheduler_type="linear", max_grad_norm=hc["max_grad_norm"], fp16=True,
        logging_steps=10, save_strategy="no", report_to=[], seed=seed, data_seed=seed, remove_unused_columns=False,
        dataloader_num_workers=0)
    tr = SpreadSafeGrad(model=model, args=args, train_dataset=benign, data_collator=collate,
                        safety_dataset=safety_loader, reference_model=None, rho=hc["rho"])
    res = tr.train()
    model.save_pretrained(out_dir)
    import shutil

    shutil.rmtree(out_dir / "_trainer", ignore_errors=True)
    losses = [h["loss"] for h in tr.state.log_history if "loss" in h]
    meta = {
        "format": "lora", "layers": sorted(plan), "rank_plan": {str(k): v for k, v in plan.items()},
        "spread": spread, "trainable_params": n_train, "train_seconds": round(time.time() - t0, 1),
        "steps": res.global_step, "loss_first": losses[0] if losses else None, "loss_last": losses[-1] if losses else None,
        "align_batches": SpreadSafeGrad.n_align, "align_ablated": SpreadSafeGrad.n_ablated,
        "align_skipped": SpreadSafeGrad.n_skipped, "peak_vram_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2),
        "harden": hc, **meta_extra,
    }
    store.write_meta(model_id, meta)
    del tr, model
    return meta


def module_shapes(model) -> dict[str, tuple[int, int]]:
    """(in, out) of each LoRA target module in one decoder layer (all layers share shapes)."""
    out = {}
    for name, m in model.named_modules():
        mm = re.search(r"layers\.0\.(self_attn|mlp)\.(\w+)$", name)
        if mm and hasattr(m, "in_features") and mm.group(2) not in out and not name.endswith(("lora_A", "lora_B")):
            out[mm.group(2)] = (m.in_features, m.out_features)
    return out


def apply_resta(src_id: str, dst_id: str, dcfg: dict) -> dict:
    """(e) SafeTune ReStaTrainer re-anchor on candidate `src_id` -> dense weights in `dst_id`."""
    import torch
    from safetune.runner.recover._layer import ReStaTrainer

    from aftermarket.defense import store
    from aftermarket.load import load_model

    rc = dcfg["harden"]["resta"]
    t0 = time.time()
    model, _ = load_model(src_id, precision="fp16")              # finetuned (GPU)
    aligned, _ = load_model(src_id, precision="fp16", device="cpu")
    base, _ = load_model("BASE", precision="fp16", device="cpu")
    ReStaTrainer(model, base_model=base, aligned_model=aligned, alpha=rc["alpha"], dare=True,
                 dare_drop_rate=rc["dare_drop_rate"], dare_seed=0, device="cpu").apply()
    del aligned, base
    out = store.model_dir(dst_id)
    out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out, safe_serialization=True)
    src = store.read_meta(src_id)
    meta = {**{k: v for k, v in src.items() if k != "id"}, "format": "full", "resta": rc, "resta_from": src_id,
            "resta_seconds": round(time.time() - t0, 1)}
    store.write_meta(dst_id, meta)
    del model
    torch.cuda.empty_cache()
    return meta


# ----------------------------------------------------------------------------- stages

def _locate() -> dict:
    from aftermarket.defense import store

    p = store.defense_dir() / "locate" / "locate.json"
    if not p.exists():
        raise FileNotFoundError("run aftermarket.defense.locate first")
    return json.loads(p.read_text(encoding="utf-8"))


def shield_spec(layers: list[int] | None, dcfg: dict) -> dict | None:
    s = dcfg["shield"]
    return None if layers is None else {"bits": s["bits"], "group_size": s["group_size"], "layers": sorted(layers)}


def control_specs(dcfg: dict, recipe: dict, loc: dict) -> list[dict]:
    """AM_rand{s} (k random layers outside L*) and AM_all (every layer, same parameter count)."""
    from aftermarket.defense.locate import random_layers

    k, lstar, n = recipe["k"], loc["L_star"][str(recipe["k"])], loc["n_layers"]
    r = dcfg["harden"]["lora_r"]
    out = []
    for s in dcfg["controls"]["rand_seeds"]:
        layers = random_layers(n, lstar, k, s)
        out.append({"id": f"AM_rand{s}", "plan": rank_plan(layers, n, r), "layers": layers, "seed": s,
                    "shield": recipe["shield"]})
    if dcfg["controls"]["all_layers"]:
        out.append({"id": "AM_all", "plan": rank_plan(None, n, r, k), "layers": None, "seed": None,
                    "shield": False})
    return out


def stage_candidates(dcfg: dict, shard: tuple[int, int]) -> None:
    from aftermarket.defense import store

    loc = _locate()
    for i, c in enumerate(candidate_grid(dcfg)):
        if i % shard[1] != shard[0] or store.read_meta(c["id"]):
            continue
        layers = loc["L_star"][str(c["k"])]
        print(f"[{c['id']}] L*={layers} spread={c['spread']}", flush=True)
        meta = train_defense(c["id"], rank_plan(layers, loc["n_layers"], dcfg["harden"]["lora_r"]), c["spread"], dcfg,
                             {"kind": "candidate", "k": c["k"], "quant_shield": shield_spec(layers, dcfg) if c["shield"] else None,
                              "locate_methods": loc["methods_used"]})
        print(f"[{c['id']}] {meta['trainable_params']:,} params, {meta['train_seconds']}s, loss "
              f"{meta['loss_first']} -> {meta['loss_last']}", flush=True)
        store.push_model(c["id"], dcfg)


def stage_resta(dcfg: dict) -> None:
    from aftermarket.defense import select, store

    chosen = select.load_selection()["stage1"]["chosen"]
    dst = f"{chosen}_r"
    if not store.read_meta(dst):
        apply_resta(chosen, dst, dcfg)
        store.push_model(dst, dcfg)
    print(f"ReSta candidate {dst} ready")


def stage_controls(dcfg: dict, shard: tuple[int, int]) -> None:
    """Controls with the selected recipe (after selection, before freeze)."""
    from aftermarket.defense import select, store

    recipe = select.load_selection()["recipe"]
    loc = _locate()
    for i, c in enumerate(control_specs(dcfg, recipe, loc)):
        if i % shard[1] != shard[0] or store.read_meta(c["id"]):
            continue
        cid = c["id"] + ("_lora" if recipe["resta"] else "")
        if not store.read_meta(cid):
            train_defense(cid, c["plan"], recipe["spread"], dcfg,
                          {"kind": "control", "k": recipe["k"], "control_seed": c["seed"],
                           "quant_shield": shield_spec(c["layers"], dcfg) if c["shield"] else None})
            store.push_model(cid, dcfg)
        if recipe["resta"]:
            apply_resta(cid, c["id"], dcfg)
            store.push_model(c["id"], dcfg)
        print(f"[{c['id']}] done ({store.read_meta(c['id'])['trainable_params']:,} trainable params)", flush=True)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="build AM candidates / controls")
    ap.add_argument("--config", required=True)
    ap.add_argument("--stage", required=True, choices=["candidates", "resta", "controls"])
    ap.add_argument("--shard", default="0/1", help="i/n: this process trains every n-th item (one per GPU)")
    ap.add_argument("--no-pull", action="store_true")
    args = ap.parse_args(argv)
    dcfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    shard = tuple(int(x) for x in args.shard.split("/"))
    from aftermarket.data import verify_checksums
    from aftermarket.defense import store

    verify_checksums()
    if not args.no_pull:
        store.pull_all(dcfg)
    {"candidates": lambda: stage_candidates(dcfg, shard), "resta": lambda: stage_resta(dcfg),
     "controls": lambda: stage_controls(dcfg, shard)}[args.stage]()


if __name__ == "__main__":
    main()
