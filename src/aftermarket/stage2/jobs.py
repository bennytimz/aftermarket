"""Launch Stage 2 work on Hugging Face Jobs (STAGE2_PLAN.md §1, §6). Every launch is a batch the user approved.

    # dry run: print the jobs and the estimated cost, launch nothing
    PYTHONPATH=src python -m aftermarket.stage2.jobs smoke
    PYTHONPATH=src python -m aftermarket.stage2.jobs cells --attacks A0 --runs BASE,SDT
    # actually launch
    PYTHONPATH=src python -m aftermarket.stage2.jobs cells --attacks A0 --launch
    PYTHONPATH=src python -m aftermarket.stage2.jobs status

Each job clones the public repo at the exact pushed commit (code only), installs requirements-stage2.txt on the
vLLM image with torch/vLLM held fixed, runs one module, and pushes results to the PRIVATE store itself.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path

from aftermarket.stage2.subsets import REPO_ROOT

IMAGE = "vllm/vllm-openai:v0.34.0"
REPO_URL = "https://github.com/bennytimz/aftermarket"
BRANCH = "stage2"
NAMESPACE = "Aftermarket-lexsi"  # the org holds the HF credits; results still go to the private bennytimz store
LOG = REPO_ROOT / "results" / "stage2" / "jobs.jsonl"
PRICE_PER_MIN = {"l4x1": 0.80 / 60, "t4-small": 0.40 / 60, "t4-medium": 0.60 / 60, "a100-large": 2.50 / 60,
                 "a10g-small": 1.00 / 60}
# minutes per cell (estimate, refined after the smoke job): setup ~6 + weights + lm-eval + generation
EST_MIN = {"A0": 25, "A1": 38, "A2": 45, "A3": 30, "A4q3": 28, "A2s1": 25, "A2s2": 25}


def head_sha() -> str:
    return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=REPO_ROOT).stdout.strip()


def pushed(sha: str) -> bool:
    r = subprocess.run(["git", "branch", "-r", "--contains", sha], capture_output=True, text=True, cwd=REPO_ROOT)
    return f"origin/{BRANCH}" in r.stdout


def script(sha: str, module_cmd: str) -> str:
    return "\n".join([
        "set -euo pipefail",
        "nvidia-smi || true",
        f"git clone -q -b {BRANCH} {REPO_URL} /work/aftermarket",
        "cd /work/aftermarket",
        f"git checkout -q {sha}",
        "pip freeze | grep -iE '^(torch|torchvision|torchaudio|vllm|triton|xformers|flashinfer[^=]*)==' > /tmp/keep.txt || true",
        "pip install -q -r requirements-stage2.txt -c /tmp/keep.txt",
        "export PYTHONPATH=src AFTERMARKET_TMP=/tmp AFTERMARKET_ADAPTERS=/tmp/adapters HF_HUB_DISABLE_PROGRESS_BARS=1",
        "export HF_HUB_ENABLE_HF_TRANSFER=0 TOKENIZERS_PARALLELISM=false",
        f"python -m {module_cmd}",
    ])


def job_specs(kind: str, runs: list[str], attacks: list[str]) -> list[dict]:
    if kind == "smoke":
        return [{"name": "s2-smoke", "flavor": "l4x1", "timeout": "2h", "est_min": 60,
                 "cmd": "aftermarket.stage2.smoke_job --final"}]
    if kind == "cells":
        return [{"name": f"s2-{r}-{a}".lower().replace("_", "-"), "flavor": "l4x1", "timeout": "3h",
                 "est_min": EST_MIN[a], "cmd": f"aftermarket.stage2.cell --run {r} --attack {a} --final"}
                for a in attacks for r in runs]
    raise KeyError(kind)


def launch(specs: list[dict], sha: str) -> list[dict]:
    from huggingface_hub import get_token, run_job

    token = get_token()
    out = []
    for s in specs:
        job = run_job(image=IMAGE, command=["bash", "-c", script(sha, s["cmd"])], flavor=s["flavor"],
                      timeout=s["timeout"], secrets={"HF_TOKEN": token}, name=s["name"], namespace=NAMESPACE,
                      labels={"project": "aftermarket-stage2"})
        rec = {**s, "job_id": job.id, "namespace": NAMESPACE, "sha": sha, "time_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
        out.append(rec)
        LOG.parent.mkdir(parents=True, exist_ok=True)
        with LOG.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")
        print(f"launched {s['name']} -> {job.id}", flush=True)
    return out


def status() -> None:
    from huggingface_hub import inspect_job

    if not LOG.exists():
        print("no jobs launched")
        return
    for line in LOG.read_text(encoding="utf-8").splitlines():
        rec = json.loads(line)
        try:
            st = inspect_job(job_id=rec["job_id"], namespace=rec.get("namespace", NAMESPACE)).status
            stage = getattr(st, "stage", st)
        except Exception as e:
            stage = f"? ({type(e).__name__})"
        print(f"{rec['name']:28s} {rec['job_id']}  {stage}")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("kind", choices=["smoke", "cells", "status"])
    ap.add_argument("--runs", default="BASE,SDT,GC,IL,RL,RL_noguard,ST,SP,AM,AM_rand0,AM_rand1,AM_all")
    ap.add_argument("--attacks", default="A0")
    ap.add_argument("--launch", action="store_true", help="really launch (default: dry run)")
    args = ap.parse_args(argv)
    if args.kind == "status":
        return status()
    specs = job_specs(args.kind, args.runs.split(","), args.attacks.split(","))
    cost = sum(s["est_min"] * PRICE_PER_MIN[s["flavor"]] for s in specs)
    for s in specs:
        print(f"{s['name']:28s} {s['flavor']:10s} ~{s['est_min']} min  {s['cmd']}")
    print(f"{len(specs)} jobs, estimated ${cost:.2f}")
    if not args.launch:
        print("dry run: add --launch to start them")
        return
    sha = head_sha()
    if not pushed(sha):
        raise SystemExit(f"HEAD {sha[:8]} is not on origin/{BRANCH}: push first (jobs clone the public repo)")
    launch(specs, sha)


if __name__ == "__main__":
    main()
