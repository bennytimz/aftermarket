"""Screenshots of the live Space for the README (Playwright, headless Chromium, 1600 px wide).

    python docs/screenshots/capture.py      # needs: pip install playwright pillow && playwright install chromium

Captures the Space's static app (the same files huggingface.co/spaces/bennytimz/aftermarket serves in its frame).
The Space shows aggregates only (no prompts or completions); the script also fails if a page shows response text.
"""

from __future__ import annotations

import io
from pathlib import Path

from PIL import Image
from playwright.sync_api import sync_playwright

URL = "https://bennytimz-aftermarket.static.hf.space/index.html"
OUT = Path(__file__).resolve().parent
WIDTH = 1600
MAX_BYTES = 300_000


def save(png: bytes, name: str) -> Path:
    """Palette-quantized PNG (charts have few colours), downscaled only if still over MAX_BYTES."""
    img = Image.open(io.BytesIO(png)).convert("RGB")
    path = OUT / f"{name}.png"
    while True:
        img.quantize(colors=128, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE).save(
            path, optimize=True)
        if path.stat().st_size <= MAX_BYTES or img.width < 900:
            return path
        img = img.resize((int(img.width * 0.85), int(img.height * 0.85)), Image.LANCZOS)


def shot(page, selector: str, name: str) -> Path:
    page.wait_for_timeout(600)
    return save(page.locator(selector).screenshot(), name)


def only_attacks(page, keep: set[str]) -> None:
    for box in page.locator("#f-attacks input[type=checkbox]").all():
        if (box.get_attribute("value") in keep) != box.is_checked():
            box.click()


def main() -> list[Path]:
    paths = []
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(viewport={"width": WIDTH, "height": 1000}, device_scale_factor=1)
        page.goto(URL, wait_until="networkidle")
        page.wait_for_selector("#heatmap svg, #heatmap table, #heatmap div", timeout=30_000)
        text = page.inner_text("body").lower()
        assert "response:" not in text and "completion:" not in text, "page shows output text"
        paths.append(save(page.screenshot(full_page=True), "space_heatmap"))
        only_attacks(page, {"A0", "A2"})
        paths.append(shot(page, "main", "space_heatmap_a2"))
        page.click("button[data-tab=retention]")
        paths.append(shot(page, "#tab-retention", "space_retention"))
        page.click("button[data-tab=cards]")
        for run in ("GC", "AM"):
            page.select_option("#card-run", run)
            paths.append(shot(page, "#card", f"card_{run}"))
        browser.close()
    for path in paths:
        print(f"{path.name}: {path.stat().st_size / 1000:.0f} KB, {Image.open(path).size}")
    return paths


if __name__ == "__main__":
    main()
