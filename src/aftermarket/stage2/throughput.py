"""E2 serving throughput (STAGE2_PLAN.md §3 E2). One job per GPU type; every config is measured the same way.

    PYTHONPATH=src python -m aftermarket.stage2.throughput --final                     # all configs
    PYTHONPATH=src python -m aftermarket.stage2.throughput --configs BASE,GC --final

Method (fixed in the plan): vLLM offline engine, fp16. The clock starts after the model is loaded, CUDA graphs are
captured and a warm-up batch of 32 prompts has run. Then 3 timed repeats of the same 256 benign Alpaca prompts
(configs/stage2/subsets.yaml throughput.timed_ids), max_tokens = min_tokens = 256, ignore_eos, temperature 0, so
every config does the same work; one natural-length pass is logged too. Output tok/s and total tok/s, mean +- sd.

Configs: merged fixes on vLLM (BASE, SDT, GC, IL, RL with its guard template, RL_noguard, ST, SP, AM); AM as an
unmerged LoRA (vLLM enable_lora); on a separate, labelled HF-transformers backend (batch 16): BASE fp16 (the bridge),
BASE HQQ 3-bit and AM HQQ 3-bit with its 8-bit shield; and BASE + Llama-Guard-3-1B screening every output.
Weights are rebuilt per config on local disk and deleted afterwards. Output: numbers only (no text) ->
results/stage2/throughput/<gpu>/<config>.json and the private store stage2/throughput/.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import statistics
import subprocess
import sys
import time
from pathlib import Path

from aftermarket.stage2.subsets import REPO_ROOT, load_subsets

VLLM_RUNS = ["BASE", "SDT", "GC", "IL", "RL", "RL_noguard", "ST", "SP", "AM"]
HF_CONFIGS = {"hf_BASE": ("BASE", "A0"), "hf_BASE_hqq3": ("BASE", "A4q3"), "hf_AM_hqq3": ("AM", "A4q3")}
ALL = VLLM_RUNS + ["AM_lora"] + list(HF_CONFIGS) + ["guard_BASE"]
N_REPEATS, MAX_TOK, HF_BATCH = 3, 256, 16
OUT = REPO_ROOT / "results" / "stage2" / "throughput"


# ----------------------------------------------------------------------------- prompts and arithmetic

def alpaca_prompts() -> tuple[list[str], list[str]]:
    """(warm-up 32, timed 256) instruction texts, from the pinned Alpaca parquet."""
    import pandas as pd
    from huggingface_hub import hf_hub_download

    t = load_subsets()["throughput"]
    df = pd.read_parquet(hf_hub_download(t["hf_dataset"], t["file"], revision=t["revision"], repo_type="dataset"))
    pick = lambda ids: [str(df.iloc[int(i.split(":")[1])]["instruction"]) for i in ids]  # noqa: E731
    return pick(t["warmup_ids"]), pick(t["timed_ids"])


def summarize(reps: list[dict]) -> dict:
    """Per-repeat {seconds, prompt_tokens, output_tokens} -> mean/sd of output and total tok/s."""
    out_tps = [r["output_tokens"] / r["seconds"] for r in reps]
    tot_tps = [(r["output_tokens"] + r["prompt_tokens"]) / r["seconds"] for r in reps]
    sd = lambda x: statistics.stdev(x) if len(x) > 1 else 0.0  # noqa: E731
    return {"output_tps_mean": statistics.mean(out_tps), "output_tps_sd": sd(out_tps),
            "total_tps_mean": statistics.mean(tot_tps), "total_tps_sd": sd(tot_tps), "repeats": reps}


def usd_per_million(price_per_hour: float, output_tps: float) -> float:
    """$ per 1M output tokens = price/h / (tok/s * 3600) * 1e6."""
    return price_per_hour / (output_tps * 3600) * 1e6


# ----------------------------------------------------------------------------- benches (run in a child process)

def bench_vllm(model_dir: str, lora: str | None = None, save_outputs: str | None = None) -> dict:
    from transformers import AutoTokenizer
    from vllm import SamplingParams, TokensPrompt

    from aftermarket.stage2.vgen import make_llm, prompt_ids, stop_ids

    tok = AutoTokenizer.from_pretrained(model_dir)
    kw = {}
    lreq = None
    if lora:
        from vllm.lora.request import LoRARequest

        r = json.loads((Path(lora) / "adapter_config.json").read_text(encoding="utf-8"))["r"]
        kw = {"enable_lora": True, "max_lora_rank": max(8, int(r)), "max_loras": 1}
        lreq = LoRARequest("adapter", 1, lora)
    t0 = time.time()
    llm = make_llm(model_dir, **kw)
    load_s = time.time() - t0
    warm, timed = alpaca_prompts()
    P = lambda texts: [TokensPrompt(prompt_token_ids=prompt_ids(tok, t)) for t in texts]  # noqa: E731
    fixed = SamplingParams(temperature=0.0, max_tokens=MAX_TOK, min_tokens=MAX_TOK, ignore_eos=True)
    llm.generate(P(warm), fixed, use_tqdm=False, lora_request=lreq)
    reps, outs = [], None
    for _ in range(N_REPEATS):
        prompts = P(timed)
        t0 = time.time()
        outs = llm.generate(prompts, fixed, use_tqdm=False, lora_request=lreq)
        dt = time.time() - t0
        reps.append({"seconds": dt, "prompt_tokens": sum(len(p["prompt_token_ids"]) for p in prompts),
                     "output_tokens": sum(len(o.outputs[0].token_ids) for o in outs)})
    nat = SamplingParams(temperature=0.0, max_tokens=MAX_TOK, stop_token_ids=stop_ids(tok))
    prompts = P(timed)
    t0 = time.time()
    nouts = llm.generate(prompts, nat, use_tqdm=False, lora_request=lreq)
    ndt = time.time() - t0
    natural = {"seconds": ndt, "prompt_tokens": sum(len(p["prompt_token_ids"]) for p in prompts),
               "output_tokens": sum(len(o.outputs[0].token_ids) for o in nouts)}
    if save_outputs:  # benign Alpaca outputs, for the guard pipeline only (local temp file, never pushed)
        Path(save_outputs).write_text(json.dumps([[t, o.outputs[0].text] for t, o in zip(timed, outs)]), encoding="utf-8")
    return {"backend": "vllm fp16", "load_seconds": load_s, "fixed": summarize(reps),
            "natural": {**natural, "output_tps": natural["output_tokens"] / ndt},
            "mean_prompt_tokens": reps[0]["prompt_tokens"] / len(timed), "lora": bool(lora)}


def bench_guard(pairs_path: str) -> dict:
    """Llama-Guard-3-1B (Stage 1 revision) on vLLM screening (request, 256-token reply) pairs: classification time."""
    import yaml
    from transformers import AutoTokenizer
    from vllm import SamplingParams, TokensPrompt

    from aftermarket.stage2.vgen import make_llm

    g = yaml.safe_load((REPO_ROOT / "configs" / "judge.yaml").read_text(encoding="utf-8"))["guard"]
    pairs = json.loads(Path(pairs_path).read_text(encoding="utf-8"))
    tok = AutoTokenizer.from_pretrained(g["model"], revision=g["revision"])
    llm = make_llm(g["model"], revision=g["revision"], tokenizer_revision=g["revision"])
    ids = []
    for q, r in pairs:
        conv = [{"role": "user", "content": [{"type": "text", "text": q}]},
                {"role": "assistant", "content": [{"type": "text", "text": r}]}]
        ids.append(tok(tok.apply_chat_template(conv, tokenize=False), add_special_tokens=False)["input_ids"])
    sp = SamplingParams(temperature=0.0, max_tokens=10)
    llm.generate([TokensPrompt(prompt_token_ids=i) for i in ids[:32]], sp, use_tqdm=False)  # warm-up
    secs = []
    for _ in range(N_REPEATS):
        t0 = time.time()
        llm.generate([TokensPrompt(prompt_token_ids=i) for i in ids], sp, use_tqdm=False)
        secs.append(time.time() - t0)
    return {"backend": "vllm fp16", "n": len(ids), "seconds": secs, "seconds_mean": statistics.mean(secs)}


def bench_hf(run_id: str, attack_id: str) -> dict:
    """HF transformers backend, batch 16, the same fixed-length work (min_new_tokens = max_new_tokens = 256).
    A4q3 keeps the real HQQ 3-bit model (AM: 8-bit shield), as deployed; A0 is fp16."""
    import torch

    from aftermarket.generate import prepare_model
    from aftermarket.stage2.vgen import prompt_ids
    from aftermarket.stage2.weights import stage2_attack

    base_attack, attack = stage2_attack(attack_id)
    t0 = time.time()
    model, tok, info = prepare_model(run_id, base_attack, attack, {"precision": "fp16", "seed": 42})
    load_s = time.time() - t0
    warm, timed = alpaca_prompts()
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id

    def run(texts: list[str]) -> dict:
        ids = [prompt_ids(tok, t) for t in texts]
        n_out, t0 = 0, time.time()
        for i in range(0, len(ids), HF_BATCH):
            chunk = ids[i:i + HF_BATCH]
            L = max(len(c) for c in chunk)
            x = torch.tensor([[pad] * (L - len(c)) + c for c in chunk], device=model.device)
            m = torch.tensor([[0] * (L - len(c)) + [1] * len(c) for c in chunk], device=model.device)
            with torch.no_grad():
                y = model.generate(input_ids=x, attention_mask=m, max_new_tokens=MAX_TOK, min_new_tokens=MAX_TOK,
                                   do_sample=False, pad_token_id=pad)
            n_out += (y.shape[1] - L) * len(chunk)
        torch.cuda.synchronize()
        return {"seconds": time.time() - t0, "prompt_tokens": sum(len(i) for i in ids), "output_tokens": n_out}

    run(warm)
    reps = [run(timed) for _ in range(N_REPEATS)]
    return {"backend": f"hf transformers batch {HF_BATCH}" + (" + HQQ 3-bit" if attack_id == "A4q3" else " fp16"),
            "load_seconds": load_s, "fixed": summarize(reps), "attack_info": {k: v for k, v in info.items()
                                                                               if isinstance(v, (int, float, str))}}


# ----------------------------------------------------------------------------- orchestration (parent process)

def _child(args: list[str], python: str | None = None) -> dict:
    from aftermarket.stage2.vgen import vllm_python

    res = Path(os.environ.get("AFTERMARKET_TMP", "/tmp")) / f"e2_{int(time.time() * 1000)}.json"
    cmd = [python or vllm_python(), "-m", "aftermarket.stage2.throughput", "--child", *args, "--result", str(res)]
    print("$", " ".join(cmd), flush=True)
    rc = subprocess.run(cmd, env={**os.environ, "PYTHONPATH": str(REPO_ROOT / "src")}).returncode
    if rc != 0 or not res.exists():
        return {"error": f"rc {rc}"}
    return json.loads(res.read_text(encoding="utf-8"))


def _weights(run_id: str, tmp: Path) -> Path:
    d = tmp / f"e2w_{run_id}"
    if not (d / "config.json").exists():
        rc = subprocess.run([sys.executable, "-m", "aftermarket.stage2.weights", "--run", run_id, "--attack", "A0",
                             "--out", str(d)], env={**os.environ, "PYTHONPATH": str(REPO_ROOT / "src")}).returncode
        if rc != 0:
            raise RuntimeError(f"weights {run_id} rc {rc}")
    return d


def am_adapter(tmp: Path) -> Path:
    import yaml
    from huggingface_hub import snapshot_download

    am = yaml.safe_load((REPO_ROOT / "configs" / "am_frozen.yaml").read_text(encoding="utf-8"))["models"]["AM"]
    d = snapshot_download(am["hf_path"], revision=am["revision"], allow_patterns=[f"{am['subfolder']}/*"],
                          local_dir=tmp / "am_adapter")
    return Path(d) / am["subfolder"]


def gpu_slug() -> str:
    from aftermarket.stage2.cell import gpu_name

    return gpu_name().split(",")[0].replace("NVIDIA ", "").replace("Tesla ", "").strip().replace(" ", "_") or "unknown"


def run_all(configs: list[str], push: bool = True) -> dict:
    from aftermarket.stage2.cell import git_sha, gpu_name, versions

    tmp = Path(os.environ.get("AFTERMARKET_TMP", "/tmp")) / "e2"
    tmp.mkdir(parents=True, exist_ok=True)
    out = OUT / gpu_slug()
    out.mkdir(parents=True, exist_ok=True)
    meta = {"gpu": gpu_name(), "git_sha": git_sha(), "versions": versions(), "n_timed": 256, "n_warmup": 32,
            "max_tokens": MAX_TOK, "repeats": N_REPEATS, "gpu_memory_utilization": None}
    from aftermarket.stage2.vgen import gpu_mem_util

    try:
        meta["gpu_memory_utilization"] = gpu_mem_util()
    except Exception:
        pass
    results = {}
    try:
        for c in configs:
            t0 = time.time()
            try:
                if c in VLLM_RUNS:
                    r = _child(["vllm", "--model", str(_weights(c, tmp))])
                elif c == "AM_lora":
                    r = _child(["vllm", "--model", str(_weights("BASE", tmp)), "--lora", str(am_adapter(tmp))])
                elif c in HF_CONFIGS:
                    r = _child(["hf", "--run", HF_CONFIGS[c][0], "--attack", HF_CONFIGS[c][1]], python=sys.executable)
                elif c == "guard_BASE":
                    pairs = tmp / "base_outputs.json"
                    gen = _child(["vllm", "--model", str(_weights("BASE", tmp)), "--save-outputs", str(pairs)])
                    g = _child(["guard", "--pairs", str(pairs)])
                    pairs.unlink(missing_ok=True)
                    r = {"generation": gen, "guard": g}
                    if "error" not in gen and "error" not in g:
                        gen_s = statistics.mean(x["seconds"] for x in gen["fixed"]["repeats"])
                        n_out = gen["fixed"]["repeats"][0]["output_tokens"]
                        r["pipeline_output_tps"] = n_out / (gen_s + g["seconds_mean"])
                else:
                    raise KeyError(c)
            except Exception as e:  # one config failing never stops the others
                r = {"error": f"{type(e).__name__}: {e}"}
            r.update({"config": c, "wall_seconds": time.time() - t0, **meta})
            (out / f"{c}.json").write_text(json.dumps(r, indent=1, default=str), encoding="utf-8")
            results[c] = r.get("fixed", {}).get("output_tps_mean") or r.get("pipeline_output_tps") or r.get("error")
            print(f"[e2] {c}: {results[c]}", flush=True)
            if c in VLLM_RUNS and c != "BASE":  # keep BASE weights for AM_lora / guard_BASE
                shutil.rmtree(tmp / f"e2w_{c}", ignore_errors=True)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)  # rebuilt weights never leave the machine
    if push:
        from huggingface_hub import HfApi

        from aftermarket.generate import ensure_private_repo

        repo = ensure_private_repo("aftermarket-runs")
        HfApi().upload_folder(folder_path=str(out), repo_id=repo, repo_type="dataset",
                              path_in_repo=f"stage2/throughput/{out.name}", commit_message=f"stage2 E2 {out.name}")
    return results


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--configs", default=",".join(ALL))
    ap.add_argument("--final", action="store_true")
    ap.add_argument("--no-push", action="store_true")
    ap.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("kind", nargs="?", help=argparse.SUPPRESS)
    ap.add_argument("--model")
    ap.add_argument("--lora")
    ap.add_argument("--save-outputs")
    ap.add_argument("--pairs")
    ap.add_argument("--run")
    ap.add_argument("--attack")
    ap.add_argument("--result")
    args = ap.parse_args(argv)
    if args.child:
        kind = args.kind
        r = (bench_vllm(args.model, args.lora, args.save_outputs) if kind == "vllm" else
             bench_guard(args.pairs) if kind == "guard" else bench_hf(args.run, args.attack))
        Path(args.result).write_text(json.dumps(r, default=str), encoding="utf-8")
        return
    if not args.final:
        raise SystemExit("E2 rebuilds released models only, but runs with --final like every Stage 2 job")
    print(json.dumps(run_all(args.configs.split(","), push=not args.no_push), indent=1, default=str), flush=True)


if __name__ == "__main__":
    main()
