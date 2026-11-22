"""Tiny HF Jobs check before any real launch: GPU, image packages, token, private-store read and write.

    PYTHONPATH=src python -m aftermarket.stage2.job_check

Writes stage2/jobs/check_<time>.json (GPU name, package versions, setup seconds; no secrets) to the private store.
"""

from __future__ import annotations

import json
import os
import subprocess
import time

from aftermarket.generate import ensure_private_repo

STORE = "aftermarket-runs"


def check() -> dict:
    import importlib.metadata as md

    from huggingface_hub import HfApi, whoami

    out: dict = {"time_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    out["gpu"] = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"],
                                capture_output=True, text=True).stdout.strip()
    out["versions"] = {}
    for p in ["torch", "vllm", "transformers", "lm_eval", "peft", "hqq", "bitsandbytes", "huggingface_hub"]:
        try:
            out["versions"][p] = md.version(p)
        except md.PackageNotFoundError:
            out["versions"][p] = None
    import torch

    out["cuda_available"] = torch.cuda.is_available()
    out["hf_user"] = whoami()["name"]
    repo = ensure_private_repo(STORE)  # refuses a public repo
    out["store_files"] = sum(1 for f in HfApi().list_repo_files(repo, repo_type="dataset") if f.startswith("stage2/cells/"))
    out["setup_seconds"] = float(os.environ.get("AFTERMARKET_SETUP_SECONDS", "nan"))
    HfApi().upload_file(path_or_fileobj=json.dumps(out, indent=1).encode(), repo_id=repo, repo_type="dataset",
                        path_in_repo=f"stage2/jobs/check_{int(time.time())}.json", commit_message="HF Jobs check")
    return out


if __name__ == "__main__":
    print(json.dumps(check(), indent=1), flush=True)
