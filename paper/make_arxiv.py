"""Bundle the paper for arXiv: the Overleaf files plus a compiled main.bbl, with \\pdfoutput=1 on main.tex's first line.

    python make_arxiv.py path/to/main.bbl     (run from paper/; `make arxiv` builds the .bbl and calls this)

arXiv does not run BibTeX, so main.bbl must come from a BibTeX run on the current custom.bib + kits.bib.
No .aux/.log/.pdf go in; figures are already PDF (make_overleaf.bundle_files). Fixed timestamps as in make_overleaf.
"""

from __future__ import annotations

import sys
import zipfile
from pathlib import Path

from make_overleaf import FIXED_TIME, HERE, bundle_files, missing_refs

OUT = HERE / "aftermarket_arxiv.zip"
PDFOUTPUT = "\\pdfoutput=1\n"


def arxiv_main_tex(src: str) -> str:
    """main.tex with \\pdfoutput=1 as its first line (once)."""
    return src if src.startswith("\\pdfoutput=1") else PDFOUTPUT + src


def bbl_problems(bbl: Path, files: dict[str, Path]) -> list[str]:
    """Reasons the .bbl cannot be trusted: absent, empty, or older than a .bib it was built from."""
    if not bbl.is_file() or "\\bibitem" not in bbl.read_text(encoding="utf-8"):
        return [f"{bbl}: missing or has no \\bibitem"]
    return [f"{bbl} is older than {n}" for n in files if n.endswith(".bib") and files[n].stat().st_mtime > bbl.stat().st_mtime]


def contents(files: dict[str, Path], bbl: Path) -> dict[str, bytes]:
    out = {name: path.read_bytes() for name, path in files.items()}
    out["main.tex"] = arxiv_main_tex(files["main.tex"].read_text(encoding="utf-8")).encode("utf-8")
    out["main.bbl"] = bbl.read_bytes()
    return out


def write_zip(data: dict[str, bytes], out: Path) -> None:
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        for name in sorted(data):
            z.writestr(zipfile.ZipInfo(name, FIXED_TIME), data[name], zipfile.ZIP_DEFLATED)


def main(argv: list[str]) -> int:
    if len(argv) != 1:
        print(__doc__, file=sys.stderr)
        return 2
    files, bbl = bundle_files(), Path(argv[0])
    problems = [n for n, p in files.items() if not p.is_file()] + missing_refs(files) + bbl_problems(bbl, files)
    if problems:
        print("cannot build the arXiv bundle:\n  " + "\n  ".join(problems), file=sys.stderr)
        return 1
    write_zip(contents(files, bbl), OUT)
    print(f"{OUT.name}: {len(files) + 1} files, {OUT.stat().st_size // 1024} KB")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
