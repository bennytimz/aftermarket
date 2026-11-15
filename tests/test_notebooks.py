"""Notebooks run top to bottom in Kaggle commit mode, so a name read before any cell defines it only fails there,
after the setup cells have spent quota (notebook 11 died on `MAX_HOURS` at cell 7). Statically check every code
cell: each name it reads, in Python or as `{NAME}` in a `!` shell line, is a builtin, IPython global, or defined
by this cell or an earlier one."""

import ast
import builtins
import json
import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
NOTEBOOKS = sorted((REPO / "notebooks").glob("*.ipynb"))
IPYTHON_GLOBALS = {"get_ipython", "display", "In", "Out", "exit", "quit"}
SHELL_REF = re.compile(r"\{([A-Za-z_][A-Za-z0-9_]*)")  # {NAME} or {NAME:...} / {NAME!r} in a ! line


def split_cell(src: str) -> tuple[str, set[str]]:
    """(Python source with IPython magics / shell lines blanked, names those lines interpolate)."""
    py, refs = [], set()
    for line in src.splitlines():
        s = line.lstrip()
        if s.startswith(("!", "%")):
            refs |= set(SHELL_REF.findall(s))
            py.append(line[: len(line) - len(s)] + "pass")  # keep indentation valid inside if/for blocks
        else:
            py.append(line)
    return "\n".join(py), refs


def names(tree: ast.AST) -> tuple[set[str], set[str]]:
    """(names defined anywhere in the cell, names read anywhere in the cell)."""
    defined, used = set(), set()
    for n in ast.walk(tree):
        if isinstance(n, ast.Name):
            (defined if isinstance(n.ctx, (ast.Store, ast.Del)) else used).add(n.id)
        elif isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            defined.add(n.name)
        elif isinstance(n, ast.arg):
            defined.add(n.arg)
        elif isinstance(n, (ast.Import, ast.ImportFrom)):
            defined |= {(a.asname or a.name).split(".")[0] for a in n.names}
        elif isinstance(n, ast.ExceptHandler) and n.name:
            defined.add(n.name)
        elif isinstance(n, (ast.Global, ast.Nonlocal)):
            defined |= set(n.names)
    return defined, used


def undefined_names(cells: list[str]) -> list[tuple[int, str]]:
    """(cell index, name) for every name read before any cell defines it."""
    known = set(dir(builtins)) | IPYTHON_GLOBALS
    missing = []
    for i, src in enumerate(cells):
        py, refs = split_cell(src)
        defined, used = names(ast.parse(py))
        known |= defined
        missing += [(i, n) for n in sorted((used | refs) - known)]
    return missing


def code_cells(path: Path) -> list[str]:
    nb = json.loads(path.read_text(encoding="utf-8"))
    return ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]


def test_checker_catches_the_notebook_11_failure():
    cells = ["import os\nGPUS = '0,1'", "print('max hours:', MAX_HOURS, '| gpus:', GPUS)"]
    assert undefined_names(cells) == [(1, "MAX_HOURS")]
    assert undefined_names(["!run --gpus {GPUS} --max {MAX_HOURS:.1f}", "GPUS = 1"]) == [(0, "GPUS"), (0, "MAX_HOURS")]
    ok = ["import yaml\nfor k in (1,):\n    !echo {k}\nf = lambda x: x\ntry:\n    pass\nexcept Exception as e:\n    print(e)"]
    assert undefined_names(ok) == []


@pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda p: p.name)
def test_notebook_names_defined_before_use(path):
    assert undefined_names(code_cells(path)) == []


# ----------------------------------------------------------------------------- stale Kaggle copies

def stamped_version(path: Path) -> str | None:
    from aftermarket.nbcheck import VERSION_PREFIX

    for src in code_cells(path):
        for line in src.splitlines():
            if line.startswith(VERSION_PREFIX):
                return line[len(VERSION_PREFIX):].strip().strip("\"'")
    return None


@pytest.mark.parametrize("path", [p for p in NOTEBOOKS if stamped_version(p) is not None], ids=lambda p: p.name)
def test_notebook_fingerprint_is_current(path):
    """Edit a fingerprinted notebook -> restamp NB_VERSION = nb_hash(path), or Kaggle can't tell a stale copy."""
    from aftermarket.nbcheck import nb_hash

    assert stamped_version(path) == nb_hash(path), f"restamp NB_VERSION in {path.name} to {nb_hash(path)}"


def test_stale_copy_is_stopped(tmp_path):
    """Notebook 11 version #2: Kaggle re-ran its saved (old) cells against the fixed repo."""
    from aftermarket.nbcheck import check_current, nb_hash

    nb11 = REPO / "notebooks" / "11_audit_demo.ipynb"
    check_current(stamped_version(nb11), nb11)
    old = json.loads(nb11.read_text(encoding="utf-8"))
    old["cells"][7]["source"] = ["print('max hours:', MAX_HOURS)"]
    stale = tmp_path / "old.ipynb"
    stale.write_text(json.dumps(old), encoding="utf-8")
    with pytest.raises(RuntimeError, match="old copy"):
        check_current(nb_hash(stale), nb11)
