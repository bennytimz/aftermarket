"""vLLM greedy generation of Stage 2 prompt sets for one saved (run, attack) model.

    PYTHONPATH=src python -m aftermarket.stage2.vgen --model /tmp/w/BASE__A0 --run BASE --sets ml_en,ml_yo,sr \
        --out results/stage2/cells/BASE__A0/gen

Same decoding contract as Stage 1 (configs/eval.yaml): greedy, max 256 new tokens, the run's own chat template
(saved with the model by stage2.weights), prompt = template text tokenized with add_special_tokens=False (as
load.build_prompt_ids), stop on EOS / <|END_OF_TURN_TOKEN|> / <|END_RESPONSE|>. Only the backend differs (vLLM
instead of HF generate); English is regenerated in the same job so language comparisons stay within one backend.
Outputs may contain harmful completions: they go to the gitignored results dir and the PRIVATE Hub store only.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

BACKEND_ENV = "AFTERMARKET_S2_BACKEND"  # "vllm" (default) | "hf" (fallback: Stage 1's HF generate path)


def backend() -> str:
    import os

    b = os.environ.get(BACKEND_ENV, "vllm").lower()
    if b not in {"vllm", "hf"}:
        raise ValueError(f"{BACKEND_ENV} must be vllm or hf")
    return b


def vllm_python() -> str:
    """Interpreter for vLLM work: AFTERMARKET_VLLM_PYTHON (Kaggle: a separate venv, so vLLM's torch never replaces
    Kaggle's), else this interpreter (HF Jobs: the vLLM image). The HF fallback always uses this interpreter."""
    import os
    import sys

    return os.environ.get("AFTERMARKET_VLLM_PYTHON") or sys.executable if backend() == "vllm" else sys.executable


def gpu_mem_util() -> float:
    """vLLM memory fraction capped at ~15 GB, so every run fits a 16 GB T4 budget (CLAUDE.md): 0.62 of a 24 GB L4,
    0.90 of a 15 GB T4."""
    import os

    if os.environ.get("AFTERMARKET_GPU_MEM_UTIL"):  # E2 on A100 only: the plan's stated exception to the 16 GB cap
        return float(os.environ["AFTERMARKET_GPU_MEM_UTIL"])
    try:
        import torch

        total = torch.cuda.get_device_properties(0).total_memory
    except Exception:
        return 0.62
    return round(min(0.90, 15.0e9 / total), 2)


def stop_ids(tok) -> list[int]:
    ids = {tok.eos_token_id}
    for t in ("<|END_OF_TURN_TOKEN|>", "<|END_RESPONSE|>"):
        i = tok.convert_tokens_to_ids(t)
        if isinstance(i, int) and i != tok.unk_token_id:
            ids.add(i)
    return sorted(i for i in ids if i is not None)


def prompt_ids(tok, text: str) -> list[int]:
    templ = tok.apply_chat_template([{"role": "user", "content": text}], tokenize=False, add_generation_prompt=True)
    return tok(templ, add_special_tokens=False)["input_ids"]


def make_llm(model_dir: str, seed: int = 42, max_model_len: int = 4096, **kw):
    from vllm import LLM

    return LLM(model=model_dir, tokenizer=model_dir, dtype="float16", seed=seed, max_model_len=max_model_len,
               gpu_memory_utilization=kw.pop("gpu_memory_utilization", gpu_mem_util()), generation_config="vllm", **kw)


def generate_rows(llm, tok, rows: list[dict], max_new: int = 256) -> tuple[list[dict], dict]:
    from vllm import SamplingParams, TokensPrompt

    sp = SamplingParams(temperature=0.0, max_tokens=max_new, stop_token_ids=stop_ids(tok), skip_special_tokens=True)
    ids = [prompt_ids(tok, r["prompt"]) for r in rows]
    t0 = time.time()
    outs = llm.generate([TokensPrompt(prompt_token_ids=i) for i in ids], sp, use_tqdm=False)
    dt = time.time() - t0
    res, n_new = [], 0
    for r, i, o in zip(rows, ids, outs):
        c = o.outputs[0]
        n_new += len(c.token_ids)
        res.append({**r, "response": c.text.strip(), "n_prompt_tokens": len(i), "n_new_tokens": len(c.token_ids),
                    "finish_reason": c.finish_reason})
    return res, {"n": len(rows), "seconds": round(dt, 2), "new_tokens": n_new,
                 "new_tokens_per_sec": round(n_new / dt, 1) if dt else None}


def load_hf(model_dir: str):
    import torch
    from transformers import AutoModelForCausalLM

    return AutoModelForCausalLM.from_pretrained(model_dir, dtype=torch.float16, device_map={"": 0}).eval()


def generate_rows_hf(model, tok, rows: list[dict], max_new: int = 256, batch_size: int = 16) -> tuple[list[dict], dict]:
    """Fallback backend: Stage 1's batched HF greedy decoding (aftermarket.generate.generate_batched)."""
    from aftermarket.generate import generate_batched

    ids = [prompt_ids(tok, r["prompt"]) for r in rows]
    t0 = time.time()
    res, bs = generate_batched(model, tok, ids, max_new, batch_size)
    dt = time.time() - t0
    n_new = sum(r["n_new_tokens"] for r in res)
    out = [{**r, "response": g["response"], "n_prompt_tokens": len(i), "n_new_tokens": g["n_new_tokens"],
            "finish_reason": g["finish_reason"]} for r, i, g in zip(rows, ids, res)]
    return out, {"n": len(rows), "seconds": round(dt, 2), "new_tokens": n_new, "batch_size_final": bs,
                 "new_tokens_per_sec": round(n_new / dt, 1) if dt else None}


def run(model_dir: str, run_id: str, attack_id: str, sets: list[str], out: Path, seed: int = 42) -> dict:
    import pandas as pd
    from transformers import AutoTokenizer

    from aftermarket.stage2.prompts import build_set

    out.mkdir(parents=True, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(model_dir)
    t0 = time.time()
    be = backend()
    llm = make_llm(model_dir, seed=seed) if be == "vllm" else load_hf(model_dir)
    gen = generate_rows if be == "vllm" else generate_rows_hf
    stats = {"backend": be, "load_seconds": round(time.time() - t0, 1), "sets": {}}
    for s in sets:
        rows, st = gen(llm, tok, build_set(s))
        df = pd.DataFrame(rows)
        df.insert(0, "attack_id", attack_id)
        df.insert(0, "run_id", run_id)
        df.to_parquet(out / f"{s}.parquet", index=False)
        stats["sets"][s] = st
        print(f"[vgen] {run_id}/{attack_id} {s}: {st}", flush=True)
    return stats


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True)
    ap.add_argument("--run", required=True)
    ap.add_argument("--attack", required=True)
    ap.add_argument("--sets", required=True)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--stats", type=Path)
    args = ap.parse_args(argv)
    st = run(args.model, args.run, args.attack, args.sets.split(","), args.out)
    if args.stats:
        args.stats.write_text(json.dumps(st, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
