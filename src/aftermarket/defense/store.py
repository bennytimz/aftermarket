"""Where AM candidates, controls and the frozen AM live (PLAN.md §6).

Local: $AFTERMARKET_DEFENSE_DIR (default: <tmp>/aftermarket_defense), one folder per model id with the
adapter and `defense_meta.json`. load.load_registry() turns every such folder
into a model spec, so the normal generate/attack runner can evaluate candidates on dev.

Hub: <whoami>/aftermarket-am, a PRIVATE model repo, mirrors the folders so a dead Kaggle session can resume
(pull before each stage). These are defense weights; attack adapters never come here (they live under
attacks.finetune.ADAPTER_ROOT and are refused). AM may be released publicly at the very end, by hand.
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from pathlib import Path

ENV = "AFTERMARKET_DEFENSE_DIR"
META = "defense_meta.json"
FROZEN_FILE = "am_frozen.yaml"


def defense_dir() -> Path:
    return Path(os.environ.get(ENV, Path(tempfile.gettempdir()) / "aftermarket_defense"))


def model_dir(model_id: str) -> Path:
    if not model_id.startswith("AM"):
        raise ValueError(f"defense model ids start with 'AM': {model_id!r}")
    return defense_dir() / model_id


def write_meta(model_id: str, meta: dict) -> Path:
    d = model_dir(model_id)
    d.mkdir(parents=True, exist_ok=True)
    p = d / META
    p.write_text(json.dumps({"id": model_id, **meta}, indent=1), encoding="utf-8")
    return p


def read_meta(model_id: str) -> dict | None:
    p = model_dir(model_id) / META
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def local_specs(root: Path | None = None) -> dict[str, dict]:
    """Model specs for every finished local defense folder (adapter or dense weights + meta)."""
    root = Path(root or defense_dir())
    out = {}
    if not root.is_dir():
        return out
    for meta_path in sorted(root.glob(f"AM*/{META}")):
        m = json.loads(meta_path.read_text(encoding="utf-8"))
        d = meta_path.parent
        weights = "adapter_model.safetensors" if m["format"] == "lora" else "model.safetensors"
        if not any(d.glob(weights.replace(".safetensors", "*.safetensors"))):
            continue  # half-written folder
        out[d.name] = {"name": m.get("name", d.name), "status": "active", "format": m["format"], "hf_path": str(d),
                       "revision": None, "base": "BASE", "fix_type": "AM defense (ours)", "local": True,
                       "files": sorted(p.name for p in d.iterdir()),
                       "quant_shield": m.get("quant_shield"), "defense": m}
    return out


# ----------------------------------------------------------------------------- private Hub mirror

def _repo(cfg: dict) -> str:
    from aftermarket.generate import resolve_repo_id

    return resolve_repo_id(cfg["store"]["hf_repo"])


def ensure_private_model_repo(repo_id: str) -> str:
    from huggingface_hub import HfApi

    from aftermarket.generate import PublicRepoError

    api = HfApi()
    api.create_repo(repo_id, repo_type="model", private=True, exist_ok=True)
    if not api.repo_info(repo_id, repo_type="model").private:
        raise PublicRepoError(f"{repo_id} is PUBLIC: defense candidates/controls are only pushed privately")
    return repo_id


def _check_not_attack(path: Path) -> None:
    from aftermarket.attacks.finetune import ADAPTER_ROOT

    root, rp = Path(ADAPTER_ROOT).resolve(), Path(path).resolve()
    if rp == root or root in rp.parents:
        raise PermissionError(f"refusing to push {path}: attack adapters are never pushed (PLAN.md §9)")


def push_model(model_id: str, cfg: dict) -> str | None:
    """Mirror one local defense folder to the private repo. Returns the commit sha, or None (non-fatal)."""
    from huggingface_hub import HfApi

    d = model_dir(model_id)
    _check_not_attack(d)
    try:
        repo = ensure_private_model_repo(_repo(cfg))
        info = HfApi().upload_folder(repo_id=repo, folder_path=str(d), path_in_repo=model_id, repo_type="model",
                                     commit_message=f"defense {model_id}")
        return getattr(info, "oid", None)
    except Exception as e:
        print(f"WARNING could not push defense model {model_id} (kept locally): {type(e).__name__}: {e}", flush=True)
        return None


def push_file(path: Path, cfg: dict, path_in_repo: str) -> str | None:
    from huggingface_hub import HfApi

    try:
        repo = ensure_private_model_repo(_repo(cfg))
        info = HfApi().upload_file(path_or_fileobj=str(path), path_in_repo=path_in_repo, repo_id=repo,
                                   repo_type="model", commit_message=f"add {path_in_repo}")
        return getattr(info, "oid", None)
    except Exception as e:
        print(f"WARNING could not push {path_in_repo}: {type(e).__name__}: {e}", flush=True)
        return None


def pull_all(cfg: dict, want=None, suffixes: tuple[str, ...] | None = None) -> int:
    """Resume: download defense folders (and locate/, am_frozen.yaml) that are missing locally. Non-fatal."""
    from huggingface_hub import HfApi, hf_hub_download

    try:
        repo = _repo(cfg)
        files = HfApi().list_repo_files(repo, repo_type="model")
    except Exception as e:
        print(f"WARNING defense pull skipped ({type(e).__name__}: {e})", flush=True)
        return 0
    root = defense_dir()
    n = 0
    # metadata last, so a folder only becomes a model spec once its weights are complete
    for f in sorted(files, key=lambda f: (f.endswith(META), f)):
        top = f.split("/", 1)[0]
        if f.startswith(".") or (want is not None and not want(top)) or (suffixes and not f.endswith(suffixes)):
            continue
        dest = root / f
        if dest.exists():
            continue
        try:
            src = hf_hub_download(repo, f, repo_type="model")
            dest.parent.mkdir(parents=True, exist_ok=True)
            tmp = dest.with_name(dest.name + ".pull")
            shutil.copyfile(src, tmp)
            tmp.replace(dest)
            n += 1
        except Exception as e:
            print(f"WARNING could not pull {f}: {type(e).__name__}: {e}", flush=True)
    print(f"defense pull: {n} files from {repo}", flush=True)
    return n
