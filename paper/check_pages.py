"""Page-budget check: everything before References must end within 6 pages, and the appendix within 6 pages.

    python paper/check_pages.py [paper/main.pdf]

Counted part = from the top of page 1 to the "References" heading, so it includes Limitations and Ethics (the
organizer counts everything before References). Appendix = from its first heading ("A Models under test") to the
last line of text in the PDF.
Length is measured in columns: a page is two columns, so a heading at the middle of a page's left column = +0.25 page.
Exits 1 if References starts after 6.0 pages, if the appendix is longer than 6.0 pages, or if either is not found.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pymupdf

LIMIT = 6.0  # pages before References
APPENDIX_LIMIT = 6.0
TOP, BOTTOM = 73.0, 770.0  # text block of the ACL template, points
MID = 300.0  # x that separates the two columns
HEADINGS = ["1 Introduction", "2 Related work", "3 Setup", "4 Results", "4.1 ", "4.2 ", "4.3 ", "4.4 ", "4.5 ", "4.6 ",
            "4.7 ", "5 Practical takeaways", "6 Conclusion", "Limitations", "Ethics statement", "References",
            "A Models under test"]
REFERENCES, APPENDIX = "References", "A Models under test"


def pages_at(page_no: int, x0: float, y: float) -> float:
    """Pages used up to point (x0, y) on page `page_no` (0-based), column-aware."""
    col = 0 if x0 < MID else 1
    frac = (y - TOP) / (BOTTOM - TOP)
    return page_no + 0.5 * col + 0.5 * max(0.0, min(1.0, frac))


def is_bold_heading(page, rect) -> bool:
    """True if the text in `rect` is set in a bold font (section headings are; body mentions are not)."""
    for block in page.get_text("dict", clip=rect)["blocks"]:
        for line in block.get("lines", []):
            for span in line["spans"]:
                if span["text"].strip() and (span["flags"] & 16 or "Bold" in span["font"]):
                    return True
    return False


def position(doc, text: str) -> float | None:
    """Pages used before heading `text` starts (fractional, column-aware), or None if absent."""
    for i, page in enumerate(doc):
        for r in page.search_for(text):
            # headings are bold section titles at a column's left edge
            if (r.x0 < 80 or 300 < r.x0 < 320) and is_bold_heading(page, r):
                return pages_at(i, r.x0, r.y0)
    return None


def text_end(doc) -> float:
    """Pages used up to the last line of text in the PDF (column-aware; page numbers in the footer ignored)."""
    i = len(doc) - 1
    lines = [b for b in doc[i].get_text("blocks") if b[4].strip() and b[1] < BOTTOM]
    if not lines:
        return float(i)
    right = [b for b in lines if b[0] >= MID]
    x0, y1 = (MID + 1, max(b[3] for b in right)) if right else (0.0, max(b[3] for b in lines))
    return pages_at(i, x0, y1)


def report(doc) -> tuple[float | None, float | None]:
    """Print where each section starts; return (pages before References, appendix length)."""
    marks = {h: position(doc, h) for h in HEADINGS}
    prev = None
    for h in HEADINGS:
        p = marks[h]
        if p is None:
            print(f"  {h.strip():22s} not found")
            continue
        span = "" if prev is None else f"(previous section {p - prev:.2f} pp)"
        print(f"  {h.strip():22s} starts at {p:.2f} pp, page {int(p) + 1} {span}")
        prev = p
    before_refs = marks[REFERENCES]
    appendix = None if marks[APPENDIX] is None else text_end(doc) - marks[APPENDIX]
    return before_refs, appendix


def main() -> None:
    pdf = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).parent / "main.pdf"
    doc = pymupdf.open(pdf)
    before_refs, appendix = report(doc)
    ok = True
    if before_refs is None:
        print("FAIL: References heading not found")
        ok = False
    else:
        verdict = "ok" if before_refs <= LIMIT else "FAIL"
        print(f"before References: {before_refs:.2f} of {LIMIT:.0f} pages "
              f"(References start on page {int(before_refs) + 1}) {verdict}")
        ok &= before_refs <= LIMIT
    if appendix is None:
        print("FAIL: appendix heading not found")
        ok = False
    else:
        verdict = "ok" if appendix <= APPENDIX_LIMIT else "FAIL"
        print(f"appendix: {appendix:.2f} of {APPENDIX_LIMIT:.0f} pages {verdict}")
        ok &= appendix <= APPENDIX_LIMIT
    print(f"whole PDF {len(doc)} pages")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
