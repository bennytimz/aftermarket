"""Write the README's "Stage 2 results" section from paper/readme_stage2.md, filling {{key}} from
derived/numbers.json and {{COST_TABLE}} from tables/cost.tex, so the README carries the paper's numbers exactly.

    python paper/make_readme.py          # rewrites the block between the stage2 markers in README.md

Fails on any {{key}} missing from numbers.json (as check_tex does for \\V{key}).
"""

from __future__ import annotations

import json
import re
from pathlib import Path

PAPER = Path(__file__).resolve().parent
README = PAPER.parent / "README.md"
START, END = "<!-- stage2:start -->", "<!-- stage2:end -->"


def latex_to_md(s: str) -> str:
    """The few macros the generated tables use -> plain text."""
    s = re.sub(r"\\(run|att)\{([^}]*)\}", r"\2", s)
    return s.replace(r"\$", "$").replace(r"\%", "%").replace("~", " ").strip()


def cost_table_md(tex: str) -> str:
    rows = [r.strip() for r in tex.split(r"\\")]
    body = [r for r in rows if "&" in r and "multicolumn" not in r and not r.lstrip().startswith("&")]
    gpus = re.findall(r"\\multicolumn\{2\}\{c\}\{([^}]*)\}", tex)
    head = ["Fix", "prompt tok"] + [f"{g} {h}" for g in gpus for h in ("tok/s", "$/1M")]
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * 1 + "---:|" * (len(head) - 1)]
    for r in body:
        r = re.sub(r"\\(toprule|midrule|bottomrule)", "", r)
        lines.append("| " + " | ".join(latex_to_md(c) for c in r.split("&")) + " |")
    return "\n".join(lines)


def render(template: str, numbers: dict, cost_tex: str) -> str:
    missing = sorted({k for k in re.findall(r"\{\{([^}]+)\}\}", template) if k != "COST_TABLE" and k not in numbers})
    if missing:
        raise SystemExit(f"keys missing from derived/numbers.json: {', '.join(missing)}")
    out = template.replace("{{COST_TABLE}}", cost_table_md(cost_tex))
    return re.sub(r"\{\{([^}]+)\}\}", lambda m: latex_to_md(str(numbers[m.group(1)])), out)


def splice(readme: str, block: str, anchor: str = "## Problem statement and approach") -> str:
    block = f"{START}\n{block.rstrip()}\n{END}\n"
    if START in readme:
        return re.sub(re.escape(START) + r".*?" + re.escape(END) + r"\n?", lambda _: block, readme, flags=re.S)
    i = readme.index(anchor)
    return readme[:i] + block + "\n" + readme[i:]


def main() -> None:
    numbers = json.loads((PAPER / "derived" / "numbers.json").read_text(encoding="utf-8"))
    block = render((PAPER / "readme_stage2.md").read_text(encoding="utf-8"), numbers,
                   (PAPER / "tables" / "cost.tex").read_text(encoding="utf-8"))
    README.write_text(splice(README.read_text(encoding="utf-8"), block), encoding="utf-8")
    print(f"README.md: Stage 2 section written ({block.count(chr(10))} lines)")


if __name__ == "__main__":
    main()
