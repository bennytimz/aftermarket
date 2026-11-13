"""Locate safety layers L* (PLAN.md §6.1, step a).

    PYTHONPATH=src python -m aftermarket.defense.locate --config configs/defense.yaml

Three per-layer scores on BASE, all from calib_pairs / dev splits (never test):
  refusal_direction  Cohen's d separation of harmful_dev vs benign_dev projected on each layer's
                     diff-in-means direction (calib_pairs): exactly A3's layer-selection statistic.
  safetune_neurons   safetune.identify_safety_neurons (activation contrast, calib_pairs): a layer's score is
                     the mean |score| of its top-k neurons.
  eap_ig             circuitkit.discover_circuit (EAP-IG, node level) on the calib refusal contrast
                     (harmful -> "I", harmless -> "Here"): a layer's score is the number of its nodes (heads + MLP)
                     kept at the target sparsity. Own process; GPU first, CPU fallback (fewer examples).
Merged by mean rank (ties -> lower layer index), so L*_k is deterministic. Also saves the top refusal
direction (for spread-refusal) and the pairwise top-k overlap of the three methods (for the write-up).

Output: $AFTERMARKET_DEFENSE_DIR/locate/locate.json (+ direction.npy), mirrored to the private store.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import yaml

from aftermarket.data import REPO_ROOT

LOCATE = "locate"


# ----------------------------------------------------------------------------- pure (unit-tested)

def average_ranks(scores: list[float]) -> list[float]:
    """Rank 1 = highest score; tied scores share their average rank (order-independent)."""
    order = sorted(range(len(scores)), key=lambda i: -scores[i])
    ranks = [0.0] * len(scores)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and scores[order[j + 1]] == scores[order[i]]:
            j += 1
        for t in range(i, j + 1):
            ranks[order[t]] = (i + j) / 2 + 1
        i = j + 1
    return ranks


def merge_rankings(method_scores: dict[str, list[float]]) -> tuple[list[int], list[float]]:
    """Layers ordered by mean rank across methods (best first); ties broken by lower layer index."""
    if not method_scores:
        raise ValueError("no locate method produced scores")
    n = {len(v) for v in method_scores.values()}
    if len(n) != 1:
        raise ValueError(f"methods disagree on the layer count: {n}")
    ranks = [average_ranks(list(map(float, method_scores[m]))) for m in sorted(method_scores)]
    mean = [float(np.mean([r[layer] for r in ranks])) for layer in range(n.pop())]
    return sorted(range(len(mean)), key=lambda layer: (mean[layer], layer)), mean


def top_k(order: list[int], k: int) -> list[int]:
    if not 0 < k <= len(order):
        raise ValueError(f"k={k} out of range for {len(order)} layers")
    return sorted(order[:k])


def method_overlap(method_scores: dict[str, list[float]], ks: list[int]) -> dict:
    """Per k: each method's own top-k layers and the pairwise Jaccard overlap (for the write-up)."""
    out = {}
    for k in ks:
        tops = {m: top_k(sorted(range(len(s)), key=lambda layer: (average_ranks(list(map(float, s)))[layer], layer)), k)
                for m, s in sorted(method_scores.items())}
        names = sorted(tops)
        pairs = {f"{a}~{b}": round(len(set(tops[a]) & set(tops[b])) / len(set(tops[a]) | set(tops[b])), 3)
                 for i, a in enumerate(names) for b in names[i + 1:]}
        common = sorted(set.intersection(*map(set, tops.values()))) if tops else []
        out[str(k)] = {"top_k": tops, "jaccard": pairs, "in_all": common}
    return out


def random_layers(n_layers: int, exclude: list[int], k: int, seed: int) -> list[int]:
    """AM_rand control: k layers drawn uniformly (seeded) from the layers NOT in L*."""
    pool = sorted(set(range(n_layers)) - set(exclude))
    if k > len(pool):
        raise ValueError("not enough layers outside L*")
    return sorted(random.Random(seed).sample(pool, k))


def eap_layer_scores(nodes, n_layers: int) -> list[float]:
    """Count kept EAP nodes ('a12.h3', 'm12', ...) per layer; input/logits nodes are ignored."""
    counts = [0.0] * n_layers
    for name in nodes:
        m = re.match(r"^[am](\d+)", str(name))
        if m and int(m.group(1)) < n_layers:
            counts[int(m.group(1))] += 1
    return counts


# ----------------------------------------------------------------------------- model side

def _render(tok, prompt: str) -> str:
    """Chat-rendered prompt text without a leading BOS string (the libraries add special tokens themselves)."""
    s = tok.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True)
    return s[len(tok.bos_token):] if tok.bos_token and s.startswith(tok.bos_token) else s


def refusal_direction_scores(model, tok, cfg: dict) -> tuple[list[float], "np.ndarray", int]:
    from aftermarket.attacks import abliterate as ab
    from aftermarket.data import load_split

    calib = load_split(cfg["fit_split"])
    dirs = ab.mean_diff_directions(ab.last_token_acts(model, tok, [r["harmful"] for r in calib]),
                                   ab.last_token_acts(model, tok, [r["harmless"] for r in calib]))
    h_split, b_split = cfg["select_on"]
    sep = ab.layer_separation(dirs, ab.last_token_acts(model, tok, [r["prompt"] for r in load_split(h_split)]),
                              ab.last_token_acts(model, tok, [r["prompt"] for r in load_split(b_split)]))
    attacks = yaml.safe_load((REPO_ROOT / "configs" / "attacks.yaml").read_text(encoding="utf-8"))["attacks"]
    layer = ab.pick_layer(sep, tuple(attacks["A3"]["layer_window"]))  # same rule as the A3 attack
    return [float(s) for s in sep], dirs[layer].numpy().astype(np.float32), layer


def safetune_scores(model, tok, cfg: dict, n_layers: int) -> list[float]:
    from safetune.instrumentation.interpret.safety_neurons import SafetyNeuronConfig, identify_safety_neurons

    from aftermarket.data import load_split

    calib = load_split(cfg["fit_split"])
    c = cfg["safetune_neurons"]
    rep = identify_safety_neurons(
        model, {}, SafetyNeuronConfig(method=c["method"], activation_score=c["activation_score"],
                                      top_k_per_layer=c["top_k_per_layer"], target_layers=list(range(n_layers))),
        tokenizer=tok, harmful_prompts=[_render(tok, r["harmful"]) for r in calib],
        harmless_prompts=[_render(tok, r["harmless"]) for r in calib])
    return [float(np.mean([abs(s) for _, s in rep.per_layer[layer]])) if rep.per_layer.get(layer) else 0.0
            for layer in range(n_layers)]


def eap_ig_worker(cfg: dict, out: Path, device: str) -> None:
    """Runs inside its own process (CircuitKIT loads its own TransformerLens copy of BASE)."""
    import tempfile

    import pandas as pd
    from circuitkit import discover_circuit

    from aftermarket.data import load_split
    from aftermarket.load import load_base_tokenizer, load_registry

    c = cfg["eap_ig"]
    n = c["num_examples"] if device == "cuda" else c["cpu_num_examples"]
    tok = load_base_tokenizer()
    calib = load_split(cfg["fit_split"])[:n]
    tmp = Path(tempfile.mkdtemp(prefix="am_eap_"))
    csv = tmp / "calib_refusal.csv"
    pd.DataFrame({"clean": [_render(tok, r["harmful"]) for r in calib],
                  "corrupt": [_render(tok, r["harmless"]) for r in calib],
                  "clean_answer": c["clean_answer"], "corrupt_answer": c["corrupt_answer"]}).to_csv(csv, index=False)
    reg = load_registry()
    conf = {
        "model": {"name": reg["base_model"], "precision": "float16" if device == "cuda" else "bfloat16", "device": device},
        "discovery": {"algorithm": "eap-ig", "level": "node", "ig_steps": c["ig_steps"], "batch_size": c["batch_size"],
                      "data_params": {"batch_size": c["batch_size"], "num_examples": len(calib)}, "seed": 42},
        "pruning": {"target_sparsity": c["target_sparsity"], "scope": "both"},
        "data": {"type": "template", "path": str(csv), "align_strategy": "none", "max_records": len(calib),
                 "template": {"clean_prompt": "{clean}", "corrupt_prompt": "{corrupt}",
                              "clean_answer": "{clean_answer}", "corrupt_answer": "{corrupt_answer}"}},
        "output_path": str(tmp / "eap_ig.pt"),
    }
    t0 = time.time()
    nodes = discover_circuit(conf)
    nodes = nodes if isinstance(nodes, list) else list(nodes.get("mlp", [])) + list(nodes.get("heads", []))
    out.write_text(json.dumps({"device": device, "n_examples": len(calib), "seconds": round(time.time() - t0, 1),
                               "nodes": [str(x) for x in nodes],
                               "note": "circuitkit model config has no revision pin: loads the Hub default branch"}),
                   encoding="utf-8")


def run_eap_ig(cfg: dict, config_path: str, out_dir: Path) -> dict | None:
    """EAP-IG in a subprocess: GPU first, then CPU fallback. None if both fail (locate continues without it)."""
    out = out_dir / "eap_ig.json"
    if out.exists():
        return json.loads(out.read_text(encoding="utf-8"))
    for device in ("cuda", "cpu"):
        env = {**os.environ, "PYTHONPATH": str(REPO_ROOT / "src")}
        if device == "cpu":
            env["CUDA_VISIBLE_DEVICES"] = ""
        cmd = [sys.executable, "-m", "aftermarket.defense.locate", "--config", config_path, "--eap-ig-worker", device,
               "--out", str(out)]
        print(f"EAP-IG on {device} ...", flush=True)
        try:
            r = subprocess.run(cmd, env=env, timeout=cfg["eap_ig"]["timeout_s"], text=True)
            if r.returncode == 0 and out.exists():
                return json.loads(out.read_text(encoding="utf-8"))
            print(f"EAP-IG on {device} failed (exit {r.returncode})", flush=True)
        except subprocess.TimeoutExpired:
            print(f"EAP-IG on {device} timed out", flush=True)
    return None


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="locate protected layers L*")
    ap.add_argument("--config", required=True)
    ap.add_argument("--no-pull", action="store_true")
    ap.add_argument("--eap-ig-worker", choices=["cuda", "cpu"], help=argparse.SUPPRESS)
    ap.add_argument("--out", help=argparse.SUPPRESS)
    args = ap.parse_args(argv)
    dcfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    cfg = dcfg["locate"]
    if args.eap_ig_worker:
        eap_ig_worker(cfg, Path(args.out), args.eap_ig_worker)
        return

    from aftermarket.data import verify_checksums
    from aftermarket.defense import store

    verify_checksums()
    if not args.no_pull:
        store.pull_all(dcfg, want=lambda top: top == LOCATE)
    out_dir = store.defense_dir() / LOCATE
    out_dir.mkdir(parents=True, exist_ok=True)
    final = out_dir / "locate.json"
    if final.exists():
        print(f"locate already done: {final}")
        print_summary(json.loads(final.read_text(encoding="utf-8")))
        return

    import torch

    from aftermarket.load import free_memory, load_model

    t0 = time.time()
    scores: dict[str, list[float]] = {}
    failures: dict[str, str] = {}
    model, tok = load_model(dcfg["base_run"], precision="fp16")
    n_layers = model.config.num_hidden_layers
    if "refusal_direction" in cfg["methods"]:
        sep, direction, d_layer = refusal_direction_scores(model, tok, cfg)
        scores["refusal_direction"] = sep
        np.save(out_dir / "direction.npy", direction)
    if "safetune_neurons" in cfg["methods"]:
        try:
            scores["safetune_neurons"] = safetune_scores(model, tok, cfg, n_layers)
        except Exception as e:
            failures["safetune_neurons"] = f"{type(e).__name__}: {e}"
    del model
    free_memory()
    torch.cuda.empty_cache()
    eap = None
    if "eap_ig" in cfg["methods"]:
        eap = run_eap_ig(cfg, args.config, out_dir)
        if eap is None:
            failures["eap_ig"] = "failed on GPU and CPU (see log)"
        else:
            scores["eap_ig"] = eap_layer_scores(eap["nodes"], n_layers)
    order, mean_rank = merge_rankings(scores)
    result = {
        "n_layers": n_layers, "methods_used": sorted(scores), "failures": failures,
        "scores": {m: [round(x, 5) for x in v] for m, v in scores.items()},
        "merged_order": order, "mean_rank": [round(x, 3) for x in mean_rank],
        "L_star": {str(k): top_k(order, k) for k in cfg["k_grid"]},
        "overlap": method_overlap(scores, cfg["k_grid"]),
        "direction_layer": d_layer if "refusal_direction" in scores else None,
        "eap_ig_run": {k: v for k, v in (eap or {}).items() if k != "nodes"},
        "data": {"fit": cfg["fit_split"], "select_on": cfg["select_on"]},
        "seconds": round(time.time() - t0, 1),
    }
    final.write_text(json.dumps(result, indent=1), encoding="utf-8")
    for f in sorted(out_dir.iterdir()):
        store.push_file(f, dcfg, f"{LOCATE}/{f.name}")
    print_summary(result)


def print_summary(r: dict) -> None:
    print(f"methods: {r['methods_used']} | failures: {r['failures'] or 'none'} | A3 direction layer {r['direction_layer']}")
    for k, layers in r["L_star"].items():
        ov = r["overlap"][k]
        print(f"L*_{k} = {layers} | in all methods: {ov['in_all']} | jaccard {ov['jaccard']}")


if __name__ == "__main__":
    main()
