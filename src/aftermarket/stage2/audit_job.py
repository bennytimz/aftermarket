"""E7: time one full `aftermarket audit` (GC, all 7 attacks) on a single HF Jobs GPU and push only the timing,
the report card (aggregate numbers) and the leaderboard comparison to stage2/audit/<name>/.

    python -m aftermarket.stage2.audit_job --final [--quick]

Generations and judgements stay in the job's disk (results/audit/, never pushed): they contain completions.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import yaml

from aftermarket.audit import AUDIT_DIR, REPO_ROOT

PUSH_PREFIX = "stage2/audit"


def gpu_name() -> str:
    try:
        import torch

        return torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    except Exception:
        return "?"


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--final", action="store_true", help="required: uses the test prompts (leaderboard rule)")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--run", default="GC")
    ap.add_argument("--no-push", action="store_true")
    args = ap.parse_args(argv)
    if not args.final:
        raise SystemExit("--final is required (fixed attacks on test prompts, no choices made from them)")
    reg = yaml.safe_load((REPO_ROOT / "configs" / "models.yaml").read_text(encoding="utf-8"))
    m = reg["models"][args.run]
    name = f"audit-{args.run.lower()}-{'quick' if args.quick else 'full'}"
    cmd = [sys.executable, "-m", "aftermarket.cli", "audit", "--model", reg["base_model"], "--adapter", m["hf_path"],
           "--revision", m["revision"], "--name", name, "--gpus", "0"] + (["--quick"] if args.quick else [])
    t0 = time.time()
    rc = subprocess.call(cmd)
    wall = time.time() - t0
    out = AUDIT_DIR / name
    info = {"name": name, "run": args.run, "mode": "quick" if args.quick else "full", "rc": rc, "gpu": gpu_name(),
            "audit_seconds": round(wall, 1), "setup_seconds": int(os.environ.get("AFTERMARKET_SETUP_SECONDS", "0")),
            "git_sha": os.environ.get("AFTERMARKET_GIT_SHA", ""), "time_utc_end": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    if rc == 0:
        try:
            from aftermarket.audit import validate

            v = validate(out, args.run, quick=args.quick)
            info["validate"] = json.loads(json.dumps(v, default=float))
        except Exception as e:  # comparison is a bonus; the timing is the point
            info["validate_error"] = f"{type(e).__name__}: {e}"
    print(json.dumps({k: v for k, v in info.items() if k != "validate"}, indent=1), flush=True)
    stage = Path("/tmp/audit_push")
    stage.mkdir(parents=True, exist_ok=True)
    (stage / "timing.json").write_text(json.dumps(info, indent=1), encoding="utf-8")
    for f in (out / "card").glob("*") if (out / "card").exists() else []:
        if f.suffix in (".md", ".png", ".json"):
            (stage / f.name).write_bytes(f.read_bytes())
    if not args.no_push:
        from huggingface_hub import HfApi

        from aftermarket.generate import ensure_private_repo

        HfApi().upload_folder(folder_path=str(stage), path_in_repo=f"{PUSH_PREFIX}/{name}",
                              repo_id=ensure_private_repo("aftermarket-runs"),
                              repo_type="dataset", commit_message=f"E7 {name}")
    sys.exit(rc)


if __name__ == "__main__":
    main()
