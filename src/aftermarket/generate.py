"""Generation step (PLAN.md §5, §13.4).

    PYTHONPATH=src python -m aftermarket.generate --config configs/gen_a0.yaml --final

Runs (run_id, attack_id, split) jobs: model x attack x prompts -> one parquet per job in
results/gen/<run>__<attack>__<split>.parquet plus a .json manifest (git sha, config, seed,
time, throughput). Jobs whose parquet exists are skipped. Jobs are grouped by run (each
model loads once) and spread across GPUs, one worker process per GPU. After each job the
parquet and manifest are pushed to a PRIVATE HF dataset; the push refuses public repos
(PLAN.md §9). Judging happens later in a separate process (judge.py).

Decoding: batched greedy, left padding with the pinned BASE tokenizer, batch size starting
at `batch_size` and halved on CUDA OOM. max_new_tokens 256 (harmful/XSTest), 8 (MMLU).
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import yaml

from aftermarket.data import load_split, verify_checksums

REPO_ROOT = Path(__file__).resolve().parents[2]
IMPLEMENTED_ATTACKS = {"A0"}


@dataclass(frozen=True, order=True)
class Job:
    run_id: str
    attack_id: str
    split: str

    @property
    def name(self) -> str:
        return f"{self.run_id}__{self.attack_id}__{self.split}"


# ----------------------------------------------------------------------------- planning (no torch)

def make_jobs(cfg: dict) -> list[Job]:
    return [Job(r, a, s) for r in cfg["runs"] for a in cfg["attacks"] for s in cfg["splits"]]


def parquet_path(out_dir: Path, job: Job) -> Path:
    return Path(out_dir) / f"{job.name}.parquet"


def pending_jobs(jobs: list[Job], out_dir: Path) -> list[Job]:
    """Jobs whose parquet does not exist yet (finished jobs are skipped)."""
    return [j for j in jobs if not parquet_path(out_dir, j).exists()]


def check_final(splits: list[str], final: bool) -> None:
    """Test splits are only generated with --final (CLAUDE.md test-set rule).

    Allowed: leaderboard cells (fixed attacks on others' models) and frozen AM. Never use test results to
    make a choice; tuning uses *_dev / calib_pairs only.
    """
    test = [s for s in splits if s.endswith("_test")]
    if test and not final:
        raise SystemExit(f"splits {test} are test sets: pass --final (leaderboard cells / frozen AM only; tune on dev splits)")


def max_new_tokens(cfg: dict, split: str) -> int:
    m = cfg["max_new_tokens"]
    return m["mmlu"] if split.startswith("mmlu") else m["default"]


def _job_cost(job: Job, cfg: dict) -> float:
    n = {"harmful_test": 200, "harmful_dev": 100, "xstest_test": 250, "mmlu_test": 300, "mmlu_dev": 100, "benign_dev": 100}.get(job.split, 200)
    return n * (max_new_tokens(cfg, job.split) + 32)


def assign_jobs(jobs: list[Job], n_workers: int, cfg: dict) -> list[list[Job]]:
    """Group jobs by run (one model load each), then longest-first onto the least-loaded worker."""
    groups: dict[str, list[Job]] = {}
    for j in sorted(jobs):
        groups.setdefault(j.run_id, []).append(j)
    loads = [0.0] * n_workers
    out: list[list[Job]] = [[] for _ in range(n_workers)]
    for run, js in sorted(groups.items(), key=lambda kv: (-sum(_job_cost(j, cfg) for j in kv[1]), kv[0])):
        w = loads.index(min(loads))
        out[w] += js
        loads[w] += sum(_job_cost(j, cfg) for j in js) + 2000  # + model load
    return out


# ----------------------------------------------------------------------------- HF push (private only)

def ensure_private_repo(repo_id: str) -> None:
    from huggingface_hub import HfApi

    api = HfApi()
    api.create_repo(repo_id, repo_type="dataset", private=True, exist_ok=True)
    if not api.repo_info(repo_id, repo_type="dataset").private:
        raise RuntimeError(f"{repo_id} is PUBLIC: refusing to push generations (PLAN.md §9)")


def push_files(paths: list[Path], repo_id: str, path_in_repo: str) -> None:
    from huggingface_hub import CommitOperationAdd, HfApi

    ensure_private_repo(repo_id)  # re-checked on every push
    ops = [CommitOperationAdd(path_in_repo=f"{path_in_repo}/{p.name}", path_or_fileobj=str(p)) for p in paths]
    HfApi().create_commit(repo_id, ops, repo_type="dataset", commit_message=f"add {paths[0].stem}")


# ----------------------------------------------------------------------------- generation (torch)

def _attack(model, tok, attack_id: str):
    if attack_id not in IMPLEMENTED_ATTACKS:
        raise NotImplementedError(f"attack {attack_id} not implemented yet")
    return model  # A0: as released


def _check_finite(model, tok) -> None:
    import torch

    ids = tok("Hello, how are you?", return_tensors="pt").input_ids.to(model.device)
    with torch.no_grad():
        logits = model(ids).logits
    if not torch.isfinite(logits).all():
        raise RuntimeError("non-finite logits: this model overflows in fp16; set precision: bf16")


def generate_batched(model, tok, prompt_ids: list[list[int]], max_new: int, batch_size: int) -> tuple[list[dict], int]:
    """Greedy-decode every prompt; returns per-prompt {response, n_new_tokens, finish_reason} and final batch size."""
    import torch

    from aftermarket.load import _stop_ids

    stops = _stop_ids(tok)
    pad = tok.pad_token_id
    order = sorted(range(len(prompt_ids)), key=lambda i: -len(prompt_ids[i]))  # longest first: OOM shows early
    results: list[dict | None] = [None] * len(prompt_ids)
    i, bs = 0, batch_size
    while i < len(order):
        chunk = order[i:i + bs]
        L = max(len(prompt_ids[k]) for k in chunk)
        x = torch.tensor([[pad] * (L - len(prompt_ids[k])) + prompt_ids[k] for k in chunk], device=model.device)
        mask = torch.tensor([[0] * (L - len(prompt_ids[k])) + [1] * len(prompt_ids[k]) for k in chunk], device=model.device)
        try:
            with torch.no_grad():
                y = model.generate(input_ids=x, attention_mask=mask, max_new_tokens=max_new, do_sample=False,
                                   eos_token_id=sorted(stops), pad_token_id=pad)
        except torch.cuda.OutOfMemoryError:
            del x, mask
            torch.cuda.empty_cache()
            if bs == 1:
                raise
            bs //= 2
            print(f"  OOM -> batch size {bs}", flush=True)
            continue
        for row, k in zip(y[:, L:].tolist(), chunk):
            cut = next((t for t, tok_id in enumerate(row) if tok_id in stops), None)
            new = row if cut is None else row[:cut]
            results[k] = {"response": tok.decode(new, skip_special_tokens=True).strip(),
                          "n_new_tokens": len(new), "finish_reason": "length" if cut is None else "stop"}
        i += len(chunk)
    return results, bs


def _git_sha() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=REPO_ROOT).stdout.strip()
    except OSError:
        return ""


def run_job(model, tok, job: Job, cfg: dict, out_dir: Path) -> dict:
    """Generate one job, write parquet + manifest, push. Returns the manifest."""
    import pandas as pd

    from aftermarket.load import build_prompt_ids, resolve_run

    model_id, template = resolve_run(job.run_id)
    rows = load_split(job.split)
    ids = [build_prompt_ids(model, tok, [{"role": "user", "content": r["prompt"]}]) for r in rows]
    t0 = time.time()
    outs, final_bs = generate_batched(model, tok, ids, max_new_tokens(cfg, job.split), cfg["batch_size"])
    secs = time.time() - t0
    n_tok = sum(o["n_new_tokens"] for o in outs)
    spec = model.aftermarket_spec
    df = pd.DataFrame([{
        "run_id": job.run_id, "model_id": model_id, "template": template, "attack_id": job.attack_id,
        "split": job.split, "item_id": r["id"], "source": r.get("source"), "category": r.get("category"),
        "prompt": r["prompt"], "answer": r.get("answer"), "prompt_tokens": len(i), **o,
    } for r, i, o in zip(rows, ids, outs)])
    out_dir.mkdir(parents=True, exist_ok=True)
    pq = parquet_path(out_dir, job)
    tmp = pq.with_suffix(".parquet.tmp")
    df.to_parquet(tmp, index=False)
    manifest = {
        "job": asdict(job), "model_id": model_id, "template": template, "hf_path": spec["hf_path"],
        "revision": spec.get("revision"), "precision": cfg["precision"], "seed": cfg["seed"],
        "max_new_tokens": max_new_tokens(cfg, job.split), "batch_size_final": final_bs, "n_prompts": len(rows),
        "seconds": round(secs, 1), "prompts_per_sec": round(len(rows) / secs, 3),
        "new_tokens_per_sec": round(n_tok / secs, 1),
        "finish_length_frac": round(float((df.finish_reason == "length").mean()), 3),
        "git_sha": _git_sha(), "time_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "config": cfg,
    }
    mpath = pq.with_suffix(".json")
    mpath.write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    tmp.replace(pq)  # parquet appears only when complete: skip logic never sees partial files
    if cfg.get("push", {}).get("enabled"):
        try:
            push_files([pq, mpath], cfg["push"]["hf_repo"], cfg["push"]["path_in_repo"])
            manifest["pushed"] = True
        except Exception as e:  # keep generating; re-push later
            print(f"  PUSH FAILED for {job.name}: {type(e).__name__}: {e}", flush=True)
    return manifest


def run_worker(jobs: list[Job], cfg: dict, out_dir: Path) -> None:
    """Run jobs on this process's GPU, loading each model run once."""
    import torch

    from aftermarket.load import free_memory, load_model, resolve_run

    torch.manual_seed(cfg["seed"])
    by_run: dict[str, list[Job]] = {}
    for j in jobs:
        by_run.setdefault(j.run_id, []).append(j)
    for run_id, js in by_run.items():
        js = pending_jobs(js, out_dir)
        if not js:
            continue
        model = tok = None
        try:  # one failing run must not stop the others on this GPU
            model_id, template = resolve_run(run_id)
            t0 = time.time()
            model, tok = load_model(model_id, precision=cfg["precision"], template=template)
            _check_finite(model, tok)
            tok.padding_side = "left"
            print(f"[{run_id}] loaded in {time.time() - t0:.0f}s", flush=True)
            for job in js:
                if parquet_path(out_dir, job).exists():
                    continue
                model = _attack(model, tok, job.attack_id)
                m = run_job(model, tok, job, cfg, out_dir)
                print(f"[{job.name}] {m['n_prompts']} prompts in {m['seconds']}s = {m['prompts_per_sec']} prompts/s, "
                      f"{m['new_tokens_per_sec']} tok/s, bs={m['batch_size_final']}, "
                      f"hit max_new_tokens={m['finish_length_frac']:.0%}", flush=True)
        except Exception:
            import traceback

            print(f"[{run_id}] FAILED:\n{traceback.format_exc()}", flush=True)
        del model, tok
        free_memory()


# ----------------------------------------------------------------------------- multi-GPU runner

def _stream(proc: subprocess.Popen, prefix: str, log: Path) -> None:
    with open(log, "a", encoding="utf-8") as f:
        for line in proc.stdout:
            f.write(line)
            print(f"{prefix} {line}", end="", flush=True)


def run_jobs(cfg: dict, config_path: str, final: bool) -> list[Job]:
    """Plan, then launch one worker process per GPU (CUDA_VISIBLE_DEVICES) and wait."""
    out_dir = REPO_ROOT / cfg["out_dir"]
    check_final(cfg["splits"], final)
    verify_checksums()
    for a in cfg["attacks"]:
        if a not in IMPLEMENTED_ATTACKS:
            raise NotImplementedError(f"attack {a} not implemented yet")
    todo = pending_jobs(make_jobs(cfg), out_dir)
    print(f"{len(make_jobs(cfg))} jobs, {len(todo)} pending")
    if not todo:
        return []
    if cfg.get("push", {}).get("enabled"):
        ensure_private_repo(cfg["push"]["hf_repo"])
    gpus = cfg["gpus"]
    plan = assign_jobs(todo, len(gpus), cfg)
    (out_dir / "logs").mkdir(parents=True, exist_ok=True)
    procs, threads = [], []
    for gpu, js in zip(gpus, plan):
        if not js:
            continue
        print(f"GPU {gpu}: {[j.name for j in js]}")
        env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu), "PYTHONUNBUFFERED": "1",
               "PYTHONPATH": str(REPO_ROOT / "src") + os.pathsep + os.environ.get("PYTHONPATH", "")}
        cmd = [sys.executable, "-m", "aftermarket.generate", "--config", config_path, "--worker",
               json.dumps([asdict(j) for j in js])] + (["--final"] if final else [])
        p = subprocess.Popen(cmd, env=env, cwd=REPO_ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        t = threading.Thread(target=_stream, args=(p, f"[gpu{gpu}]", out_dir / "logs" / f"gpu{gpu}.log"), daemon=True)
        t.start()
        procs.append((gpu, p))
        threads.append(t)
    codes = {gpu: p.wait() for gpu, p in procs}
    for t in threads:
        t.join()
    failed = pending_jobs(todo, out_dir)
    print(f"worker exit codes {codes}; still pending: {[j.name for j in failed]}")
    return failed


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="batched greedy generation")
    ap.add_argument("--config", required=True)
    ap.add_argument("--final", action="store_true", help="allow test splits: leaderboard cells or frozen AM only")
    ap.add_argument("--worker", help=argparse.SUPPRESS)
    args = ap.parse_args(argv)
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    out_dir = REPO_ROOT / cfg["out_dir"]
    if args.worker:
        jobs = [Job(**d) for d in json.loads(args.worker)]
        check_final([j.split for j in jobs], args.final)
        run_worker(jobs, cfg, out_dir)
    else:
        if run_jobs(cfg, args.config, args.final):
            raise SystemExit(1)


if __name__ == "__main__":
    main()
