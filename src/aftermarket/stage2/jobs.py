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
TARBALL = "https://codeload.github.com/bennytimz/aftermarket/tar.gz"
BRANCH = "stage2"
NAMESPACE = "Aftermarket-lexsi"  # the org holds the HF credits; results still go to the private bennytimz store
LOG = REPO_ROOT / "results" / "stage2" / "jobs.jsonl"
PRICE_PER_MIN = {"l4x1": 0.80 / 60, "t4-small": 0.40 / 60, "t4-medium": 0.60 / 60, "a100-large": 2.50 / 60,
                 "a10g-small": 1.00 / 60}
# L4 minutes per cell (22 Nov): Kaggle T4 session-1 timings / ~1.35 (assumed L4 speedup, checked by the first
# cell) + ~4 min image pull and setup. RL's guard template makes every prompt longer; GC is short.
EST_MIN = {"A0": 36, "A1": 45, "A2": 50, "A3": 41, "A4q3": 35, "A2s1": 30, "A2s2": 30}
RUN_FACTOR = {"RL": 1.6, "GC": 0.45}
# Judging (estimate; refined by the first judge job): NLLB on ~950 non-English replies, WildGuard 4-bit on every row,
# Llama Guard on harmful rows. Rows per cell: A0/A3 1,350, A2 1,550, A4q3 1,200, A1 200.
JUDGE_MIN_PER_CELL = {"A0": 14, "A2": 16, "A3": 14, "A4q3": 12, "A1": 3, "A2s1": 3, "A2s2": 3}
JUDGE_LOAD_MIN = 10  # image, three model downloads + loads


def head_sha() -> str:
    return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=REPO_ROOT).stdout.strip()


def pushed(sha: str) -> bool:
    r = subprocess.run(["git", "branch", "-r", "--contains", sha], capture_output=True, text=True, cwd=REPO_ROOT)
    return f"origin/{BRANCH}" in r.stdout


def script(sha: str, module_cmd: str) -> str:
    return "\n".join([
        "set -euo pipefail",
        "T0=$(date +%s)",
        "nvidia-smi || true",
        # the vLLM image has no git: fetch the exact pushed commit as a tarball (code only)
        "mkdir -p /work && cd /work",
        "python3 -c \"import io, tarfile, urllib.request as u; "
        f"tarfile.open(fileobj=io.BytesIO(u.urlopen('{TARBALL}/{sha}').read())).extractall('/work')\"",
        f"mv /work/aftermarket-{sha} /work/aftermarket && cd /work/aftermarket",
        f"export AFTERMARKET_GIT_SHA={sha}",
        "pip freeze | grep -iE '^(torch|torchvision|torchaudio|vllm|triton|xformers|flashinfer[^=]*)==' > /tmp/keep.txt || true",
        "pip install -q -r requirements-stage2.txt -c /tmp/keep.txt",
        "export PYTHONPATH=src AFTERMARKET_TMP=/tmp AFTERMARKET_ADAPTERS=/tmp/adapters HF_HUB_DISABLE_PROGRESS_BARS=1",
        "export HF_HUB_ENABLE_HF_TRANSFER=0 TOKENIZERS_PARALLELISM=false",
        "export AFTERMARKET_SETUP_SECONDS=$(( $(date +%s) - T0 ))",
        f"python3 -m {module_cmd}",  # the image has python3 only
    ])


def job_specs(kind: str, runs: list[str], attacks: list[str], tag: str = "", per_job: int = 6) -> list[dict]:
    if kind == "check":  # tiny: GPU + packages + token + private-store write, ~10 min
        return [{"name": "s2-check", "flavor": "l4x1", "timeout": "25m", "est_min": 12,
                 "cmd": "aftermarket.stage2.job_check"}]
    if kind == "smoke":
        return [{"name": "s2-smoke", "flavor": "l4x1", "timeout": "2h", "est_min": 60,
                 "cmd": "aftermarket.stage2.smoke_job --final"}]
    if kind == "cells":
        sfx = f" --tag {tag}" if tag else ""
        return [{"name": f"s2-{r}-{a}{'-' + tag if tag else ''}".lower().replace("_", "-"), "flavor": "l4x1",
                 "timeout": "3h", "est_min": round(EST_MIN[a] * RUN_FACTOR.get(r, 1.0)),
                 "cmd": f"aftermarket.stage2.cell --run {r} --attack {a} --final{sfx}"}
                for a in attacks for r in runs]
    if kind == "e2":  # one throughput job per GPU type (attacks = flavors here)
        est = {"l4x1": 210, "a100-large": 120, "t4-small": 300}
        return [{"name": f"s2-e2-{f}", "flavor": f, "timeout": "6h", "est_min": est[f],
                 "env": {"AFTERMARKET_GPU_MEM_UTIL": "0.90"} if f.startswith("a100") else {},
                 "cmd": "aftermarket.stage2.throughput --final"} for f in attacks]
    if kind == "judge":  # batches of finished cells; models load once per job
        cells = [f"{r}__{a}" for a in attacks for r in runs]
        out = []
        for i in range(0, len(cells), per_job):
            batch = cells[i:i + per_job]
            est = JUDGE_LOAD_MIN + sum(JUDGE_MIN_PER_CELL[c.split("__")[1]] for c in batch)
            out.append({"name": f"s2-judge-{'-'.join(sorted({c.split('__')[1] for c in batch}))}-{i // per_job}".lower(),
                        "flavor": "l4x1", "timeout": "4h", "est_min": round(est),
                        "cmd": f"aftermarket.stage2.judge_job --cells {','.join(batch)} --final"})
        return out
    raise KeyError(kind)


def launch(specs: list[dict], sha: str) -> list[dict]:
    from huggingface_hub import get_token, run_job

    token = get_token()
    out = []
    for s in specs:
        job = run_job(image=IMAGE, command=["bash", "-c", script(sha, s["cmd"])], flavor=s["flavor"],
                      timeout=s["timeout"], secrets={"HF_TOKEN": token}, env=s.get("env") or None,
                      name=s["name"], namespace=NAMESPACE,
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
    ap.add_argument("kind", choices=["check", "smoke", "cells", "judge", "e2", "status"])
    ap.add_argument("--runs", default="BASE,SDT,GC,IL,RL,RL_noguard,ST,SP,AM,AM_rand0,AM_rand1,AM_all")
    ap.add_argument("--attacks", default="A0")
    ap.add_argument("--tag", default="", help="replicate-cell suffix (cell.py --tag)")
    ap.add_argument("--per-job", type=int, default=6, help="judge: cells per job")
    ap.add_argument("--launch", action="store_true", help="really launch (default: dry run)")
    args = ap.parse_args(argv)
    if args.kind == "status":
        return status()
    specs = job_specs(args.kind, args.runs.split(","), args.attacks.split(","), args.tag, args.per_job)
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
