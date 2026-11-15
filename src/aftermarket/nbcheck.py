"""Is this Kaggle notebook the repo's current version? Kaggle runs its own saved copy of a notebook: `git pull`
updates the code it imports, never its cells (notebook 11 version #2 re-ran the old, broken cells against a fixed
repo). Each such notebook carries `NB_VERSION = "<hash>"`, the fingerprint of its code cells (that line excluded);
right after cloning it compares that with the repo's copy and stops if they differ. No dependencies: runs before
any install."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

VERSION_PREFIX = "NB_VERSION = "


def nb_hash(path: str | Path) -> str:
    cells = json.loads(Path(path).read_text(encoding="utf-8"))["cells"]
    srcs = ["".join(c["source"]) for c in cells if c["cell_type"] == "code"]
    srcs = ["\n".join(l for l in s.splitlines() if not l.startswith(VERSION_PREFIX)) for s in srcs]
    return hashlib.sha256("\n\n".join(srcs).encode("utf-8")).hexdigest()[:12]


def check_current(nb_version: str, repo_copy: str | Path) -> None:
    """Raise if the running notebook (fingerprint nb_version) is not the repo's copy."""
    current = nb_hash(repo_copy)
    if nb_version != current:
        raise RuntimeError(
            f"This notebook is an old copy (fingerprint {nb_version}, repo has {current}): Kaggle runs its saved "
            f"cells, which `git pull` does not update. Re-import {Path(repo_copy).as_posix()} from GitHub "
            "(File -> Import Notebook -> GitHub, or replace the cells), then Save & Run All again.")
