"""One Stage 2 cell job: (run, attack) -> rebuilt weights -> lm-eval (E1) -> vLLM generation (E3/E5/E6) -> push.

    PYTHONPATH=src python -m aftermarket.stage2.cell --config configs/stage2/cells.yaml --run BASE --attack A0 --final

Each phase runs in its own subprocess, so the HF training / editing stack and vLLM never share GPU memory.
Outputs: results/stage2/cells/<run>__<attack>/ {manifest.json, weights_info.json, vgen_stats.json,
lm_eval/**, gen/*.parquet}, pushed to the PRIVATE store <user>/aftermarket-runs under stage2/cells/<cell>/.
The rebuilt weights live under $AFTERMARKET_TMP only and are deleted in `finally`, whatever happens (PLAN.md §9).
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import yaml

from aftermarket.stage2.subsets import REPO_ROOT

PUSHABLE = {".parquet", ".json", ".jsonl"}


def load_cfg(path: Path) -> dict:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def sets_for(cfg: dict, attack_id: str) -> list[str]:
    if attack_id in cfg.get("seed_attacks", []):
        return list(cfg["seed_sets"])
    return list(cfg["sets"].get(attack_id, []))


def does_capability(cfg: dict, attack_id: str) -> bool:
    return attack_id in cfg["capability_attacks"]


def cell_name(run_id: str, attack_id: str) -> str:
    return f"{run_id}__{attack_id}"


def check_final(final: bool) -> None:
    """Every Stage 2 cell uses test-derived items (CLAUDE.md test-split rule)."""
    if not final:
        raise SystemExit("Stage 2 cells use test-derived items: pass --final (no choice is made from them)")


def _sh(cmd: list[str], log: Path) -> int:
    print("$", " ".join(cmd)[:300], flush=True)
    with log.open("a", encoding="utf-8") as f:
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1,
                             env={**os.environ, "PYTHONPATH": str(REPO_ROOT / "src")})
        for line in p.stdout:
            sys.stdout.write(line)
            f.write(line)
        return p.wait()


def gpu_name() -> str:
    try:
        return subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
                              capture_output=True, text=True).stdout.strip()
    except OSError:
        return "unknown"


def versions() -> dict:
    import importlib.metadata as md

    out = {}
    for p in ["torch", "transformers", "vllm", "lm_eval", "peft", "hqq", "bitsandbytes"]:
        try:
            out[p] = md.version(p)
        except md.PackageNotFoundError:
            out[p] = None
    return out


def file_sha256(path: Path) -> str | None:
    import hashlib

    return hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else None


def git_sha() -> str:
    if os.environ.get("AFTERMARKET_GIT_SHA"):  # HF Jobs image has no git: the launcher passes the sha
        return os.environ["AFTERMARKET_GIT_SHA"]
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=REPO_ROOT).stdout.strip()
    except OSError:
        return "unknown"


def pushable_files(out: Path, weights_dir: Path) -> list[Path]:
    files = [p for p in out.rglob("*") if p.is_file() and p.suffix in PUSHABLE]
    wd = weights_dir.resolve()
    for p in files:
        if wd in p.resolve().parents or "adapter" in p.name.lower() or p.suffix == ".safetensors":
            raise PermissionError(f"refusing to push {p}")
    return files


def push(out: Path, weights_dir: Path, repo: str, path_in_repo: str) -> str:
    from huggingface_hub import CommitOperationAdd, HfApi

    from aftermarket.generate import ensure_private_repo

    repo_id = ensure_private_repo(repo)
    ops = [CommitOperationAdd(path_in_repo=f"{path_in_repo}/{p.relative_to(out).as_posix()}", path_or_fileobj=str(p))
           for p in pushable_files(out, weights_dir)]
    HfApi().create_commit(repo_id, ops, repo_type="dataset", commit_message=f"stage2 {out.name}")
    return repo_id


def run_cell(cfg: dict, run_id: str, attack_id: str, phases: list[str], smoke: bool = False,
             do_push: bool = True, tag: str = "") -> dict:
    name = cell_name(run_id, attack_id) + ("__smoke" if smoke else "") + (f"__{tag}" if tag else "")
    out = REPO_ROOT / cfg["out_dir"] / name
    out.mkdir(parents=True, exist_ok=True)
    tmp = Path(os.environ.get("AFTERMARKET_TMP", "/tmp")) / "s2"
    wdir = tmp / "weights" / name
    log = out / "log.txt"
    man = {"cell": name, "run_id": run_id, "attack_id": attack_id, "stage": "stage2 (not pre-registered)",
           "git_sha": git_sha(), "seed": cfg["seed"], "time_utc_start": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "gpu": gpu_name(), "versions": versions(), "config": cfg, "phases": {}, "smoke": smoke,
           "translations_sha256": file_sha256(REPO_ROOT / "data" / "stage2" / "translations.jsonl")}
    from aftermarket.stage2.vgen import backend, vllm_python

    py = [sys.executable, "-m"]
    try:
        t = time.time()
        rc = _sh(py + ["aftermarket.stage2.weights", "--run", run_id, "--attack", attack_id, "--out", str(wdir),
                       "--info", str(out / "weights_info.json")], log)
        man["phases"]["weights"] = {"rc": rc, "seconds": round(time.time() - t, 1)}
        if rc:
            raise RuntimeError(f"weights failed (rc {rc})")
        if "capability" in phases and does_capability(cfg, attack_id):
            from aftermarket.stage2 import capability

            t = time.time()
            limit = cfg.get("smoke_limits") if smoke else None
            rc = capability.run(str(wdir), out / "lm_eval", tmp, limit_override=limit)
            man["phases"]["capability"] = {"rc": rc, "seconds": round(time.time() - t, 1),
                                           "backend": f"lm-eval {backend()} (fp16" + (", fake-quant HQQ" if attack_id.startswith("A4") else "") + ")"}
            man["capability"] = capability.collect(out / "lm_eval")
        sets = sets_for(cfg, attack_id)
        if "gen" in phases and sets:
            t = time.time()
            rc = _sh([vllm_python(), "-m", "aftermarket.stage2.vgen", "--model", str(wdir), "--run", run_id, "--attack", attack_id,
                           "--sets", ",".join(sets), "--out", str(out / "gen"),
                           "--stats", str(out / "vgen_stats.json")], log)
            man["phases"]["gen"] = {"rc": rc, "seconds": round(time.time() - t, 1), "sets": sets, "backend": f"{backend()} fp16"}
    except Exception as e:
        man["error"] = f"{type(e).__name__}: {e}"
        print("CELL FAILED:", man["error"], flush=True)
    finally:
        shutil.rmtree(wdir, ignore_errors=True)  # attacked weights never outlive the job
        man["weights_deleted"] = not wdir.exists()
        man["time_utc_end"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        (out / "manifest.json").write_text(json.dumps(man, indent=1, default=str), encoding="utf-8")
    if do_push and cfg.get("push", {}).get("enabled"):
        man["pushed_to"] = push(out, wdir, cfg["push"]["hf_repo"], f"{cfg['push']['path_in_repo']}/{name}")
        (out / "manifest.json").write_text(json.dumps(man, indent=1, default=str), encoding="utf-8")
    return man


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=str(REPO_ROOT / "configs" / "stage2" / "cells.yaml"))
    ap.add_argument("--run", required=True)
    ap.add_argument("--attack", required=True)
    ap.add_argument("--phases", default="weights,capability,gen")
    ap.add_argument("--final", action="store_true")
    ap.add_argument("--smoke", action="store_true", help="tiny lm-eval subsets (cfg smoke_limits)")
    ap.add_argument("--no-push", action="store_true")
    ap.add_argument("--tag", default="", help="suffix for a replicate cell, e.g. l4 -> BASE__A0__l4 (kept out of main tables)")
    args = ap.parse_args(argv)
    check_final(args.final)
    cfg = load_cfg(Path(args.config))
    man = run_cell(cfg, args.run, args.attack, args.phases.split(","), smoke=args.smoke, do_push=not args.no_push,
                   tag=args.tag)
    if man.get("error"):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
