"""Fail the build on broken LaTeX text: control characters left by a Python string eating an escape (\\r in \\ref,
\\t in \\textbf, \\f in \\frac, \\b in \\bf, \\a in \\att), and any TODO left in the sources or generated files.

    python check_tex.py            # exit 1 with a list of problems
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
FILES = ["main.tex", "numbers.tex", "sections/*.tex", "tables/*.tex"]
EATEN = {"\r": r"\r (\ref, \run)", "\t": r"\t (\textbf, \tabcolsep)", "\f": r"\f (\frac, \footnote)",
         "\b": r"\b (\bf, \bottomrule)", "\a": r"\a (\att)", "\v": r"\v (\vspace)"}


def problems(text: str, name: str) -> list[str]:
    out = []
    body = text.replace("\r\n", "\n")
    for i, line in enumerate(body.split("\n"), 1):
        for ch, what in EATEN.items():
            if ch in line:
                out.append(f"{name}:{i}: control character {what!s} eaten: {line.strip()[:80]!r}")
        for ch in line:
            if ord(ch) < 32 and ch not in EATEN:
                out.append(f"{name}:{i}: control character 0x{ord(ch):02x}")
        if re.search(r"\\todo\{|TODO", line) and not line.lstrip().startswith("%") and "newcommand{\\todo}" not in line:
            out.append(f"{name}:{i}: TODO left: {line.strip()[:80]}")
    return out


def missing_numbers() -> list[str]:
    """Every \\V{key} in the sources must exist in derived/numbers.json (a missing number is a silent hole)."""
    import json

    nums = HERE / "derived" / "numbers.json"
    have = set(json.loads(nums.read_text(encoding="utf-8"))) if nums.exists() else set()
    out = []
    for f in sorted(HERE.glob("sections/*.tex")) + [HERE / "main.tex"]:
        for i, line in enumerate(f.read_text(encoding="utf-8").splitlines(), 1):
            if line.lstrip().startswith("%"):
                continue
            for k in re.findall(r"\\V\{([^}]+)\}", line):
                if k not in have:
                    out.append(f"{f.relative_to(HERE)}:{i}: number \\V{{{k}}} missing from derived/numbers.json")
    return out


def main() -> int:
    errs = missing_numbers()
    for pat in FILES:
        for f in sorted(HERE.glob(pat)):
            errs += problems(f.read_bytes().decode("utf-8"), str(f.relative_to(HERE)))
    print("\n".join(errs) if errs else "check_tex: no eaten escapes, no TODOs")
    return 1 if errs else 0


if __name__ == "__main__":
    sys.exit(main())
