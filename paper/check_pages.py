"""Page-budget check: how much of the 6-page limit the main text uses, and where each section starts.

    python paper/check_pages.py [paper/main.pdf]

Main text = from the top of page 1 to the "Limitations" heading (Limitations, Ethics and References don't count).
Length is measured in columns: a page is two columns, so a heading at the middle of a page's left column = +0.25 page.
Exits 1 if the main text exceeds the limit.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pymupdf

LIMIT = 6.0
TOP, BOTTOM = 73.0, 770.0  # text block of the ACL template, points
HEADINGS = ["1 Introduction", "2 Related work", "3 Setup", "4 Results", "4.1 ", "4.2 ", "4.3 ", "4.4 ", "4.5 ", "4.6 ",
            "4.7 ", "5 Practical takeaways", "6 Conclusion", "Limitations"]


def position(doc, text: str) -> float | None:
    """Pages used before `text` starts (fractional, column-aware), or None if absent."""
    for i, page in enumerate(doc):
        for r in page.search_for(text):
            # headings are bold section titles: require the match to be at a column's left edge
            if r.x0 < 80 or 300 < r.x0 < 320:
                col = 0 if r.x0 < 200 else 1
                frac = (r.y0 - TOP) / (BOTTOM - TOP)
                return i + 0.5 * col + 0.5 * max(0.0, min(1.0, frac))
    return None


def main() -> None:
    pdf = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).parent / "main.pdf"
    doc = pymupdf.open(pdf)
    marks = {h: position(doc, h) for h in HEADINGS}
    prev = None
    for h in HEADINGS:
        p = marks[h]
        if p is None:
            print(f"  {h.strip():22s} not found")
            continue
        span = "" if prev is None else f"(previous section {p - prev:.2f} pp)"
        print(f"  {h.strip():22s} starts at {p:.2f} pp {span}")
        prev = p
    used = marks["Limitations"]
    print(f"main text: {used:.2f} of {LIMIT:.0f} pages; whole PDF {len(doc)} pages")
    sys.exit(0 if used is not None and used <= LIMIT else 1)


if __name__ == "__main__":
    main()
