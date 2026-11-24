"""Bundle the paper for Overleaf: a flat, self-contained zip that compiles with pdfLaTeX + BibTeX.

    python make_overleaf.py            (run from paper/; `make overleaf`)

Root: main.tex, acl.sty, acl_natbib.bst, custom.bib, numbers.tex; plus sections/, tables/, figures/*.pdf.
Fails if a file that main.tex or a section \\input's or \\includegraphics's is missing from the bundle.
Entries carry a fixed timestamp, so the same sources give identical contents (and, with the same zlib, identical bytes).
"""

from __future__ import annotations

import re
import sys
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / "aftermarket_overleaf.zip"
FIXED_TIME = (2026, 1, 1, 0, 0, 0)
REF = re.compile(r"\\(?:input|includegraphics)(?:\[[^\]]*\])?\{([^}]+)\}")


def bundle_files() -> dict[str, Path]:
    """Archive name -> source path."""
    files = {name: HERE / name for name in ("main.tex", "custom.bib", "numbers.tex")}
    files.update({name: HERE / "template" / name for name in ("acl.sty", "acl_natbib.bst")})
    for pattern in ("sections/*.tex", "tables/*.tex", "figures/*.pdf"):
        files.update({p.relative_to(HERE).as_posix(): p for p in sorted(HERE.glob(pattern))})
    return files


def missing_refs(files: dict[str, Path]) -> list[str]:
    """Files referenced by \\input / \\includegraphics in the bundled .tex that the bundle lacks."""
    missing = []
    for name, path in files.items():
        if name.endswith(".tex"):
            for ref in REF.findall(path.read_text(encoding="utf-8")):
                target = ref if Path(ref).suffix else ref + ".tex"
                if target not in files:
                    missing.append(f"{name} -> {target}")
    return missing


def write_zip(files: dict[str, Path], out: Path) -> None:
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        for name in sorted(files):
            info = zipfile.ZipInfo(name, FIXED_TIME)
            info.compress_type = zipfile.ZIP_DEFLATED
            z.writestr(info, files[name].read_bytes())


def main() -> int:
    files = bundle_files()
    absent = [n for n, p in files.items() if not p.is_file()]
    missing = absent + missing_refs(files)
    if missing:
        print("missing from the Overleaf bundle:\n  " + "\n  ".join(missing), file=sys.stderr)
        return 1
    write_zip(files, OUT)
    print(f"{OUT.name}: {len(files)} files, {OUT.stat().st_size // 1024} KB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
