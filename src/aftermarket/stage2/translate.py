"""NLLB-200 translation for E6 (STAGE2_PLAN.md §3 E6). One model for both directions.

    # prompts: the 150 E6 items -> es, zh, yo, sw, fr  (+ back-translations, for the Yoruba check)
    PYTHONPATH=src python -m aftermarket.stage2.translate prompts
    # Yoruba check file (prompt half now; reply half once BASE A0 replies exist)
    PYTHONPATH=src python -m aftermarket.stage2.translate yo-check [--replies path/to/BASE__A0/ml_yo.parquet]

Greedy beam-free decoding (num_beams 4 would be ~4x slower for little gain on short prompts; fixed a priori).
The translated prompts are benchmark *prompts* (public HarmBench / AdvBench / XSTest text in other languages), not
completions; they are committed like the English splits.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from aftermarket.stage2.subsets import REPO_ROOT

MODEL = "facebook/nllb-200-distilled-1.3B"
REVISION = "7be3e24664b38ce1cac29b8aeed6911aa0cf0576"
NLLB = {"en": "eng_Latn", "es": "spa_Latn", "zh": "zho_Hans", "yo": "yor_Latn", "sw": "swh_Latn",
        "fr": "fra_Latn"}
OUT = REPO_ROOT / "data" / "stage2" / "translations.jsonl"
YO_CHECK = REPO_ROOT / "data" / "translation_check_yo.csv"


class Translator:
    def __init__(self, device: str | None = None, batch_size: int = 16, max_len: int = 512):
        import torch
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        dtype = torch.float16 if self.device == "cuda" else torch.float32
        self.tok = AutoTokenizer.from_pretrained(MODEL, revision=REVISION)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(MODEL, revision=REVISION, dtype=dtype).to(self.device).eval()
        self.bs, self.max_len = batch_size, max_len

    def __call__(self, texts: list[str], src: str, tgt: str) -> list[str]:
        import torch

        if src == tgt:
            return list(texts)
        self.tok.src_lang = NLLB[src]
        bos = self.tok.convert_tokens_to_ids(NLLB[tgt])
        order = sorted(range(len(texts)), key=lambda i: len(texts[i]))  # length-sorted batches
        out: list[str] = [""] * len(texts)
        for s in range(0, len(order), self.bs):
            idx = order[s:s + self.bs]
            batch = [texts[i] if texts[i].strip() else "." for i in idx]
            enc = self.tok(batch, return_tensors="pt", padding=True, truncation=True,
                           max_length=self.max_len).to(self.device)
            with torch.no_grad():
                gen = self.model.generate(**enc, forced_bos_token_id=bos, num_beams=1, do_sample=False,
                                          max_new_tokens=min(self.max_len, int(enc["input_ids"].shape[1] * 2) + 16))
            for i, t in zip(idx, self.tok.batch_decode(gen, skip_special_tokens=True)):
                out[i] = t.strip() if texts[i].strip() else ""
        return out


def translate_prompts(langs: list[str], out: Path = OUT) -> list[dict]:
    from aftermarket.stage2.prompts import ml_items

    items = ml_items()
    tr = Translator()
    src = [it["prompt"] for it in items]
    rows = []
    for lang in langs:
        fwd = tr(src, "en", lang)
        back = tr(fwd, lang, "en")
        rows += [{"id": it["id"], "kind": it["kind"], "lang": lang, "text": f, "back_en": b, "model": MODEL,
                  "revision": REVISION} for it, f, b in zip(items, fwd, back)]
        print(f"[translate] {lang}: {len(fwd)} prompts", flush=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    return rows


def yo_check(replies: Path | None = None, n: int = 30, out: Path = YO_CHECK) -> Path:
    """30 Yoruba prompts (+ back-translation) and, if given, 30 BASE A0 Yoruba replies (+ back-translation),
    with blank columns for the human verdict. Only replies to XSTest (safe) prompts are exported, so the file
    can never hold a harmful completion and may be committed."""
    import random

    from aftermarket.stage2.prompts import ml_items
    from aftermarket.stage2.subsets import read_jsonl

    tr = {(r["id"], r["lang"]): r for r in read_jsonl(OUT)}
    items = ml_items()
    rng = random.Random(7)
    harm = [it for it in items if it["kind"] == "harmful"]
    safe = [it for it in items if it["kind"] == "xstest"]
    pick = rng.sample(harm, 15) + rng.sample(safe, 15)
    header = ["part", "id", "kind", "english_source", "yoruba", "back_translation_en", "adequate (y/n)",
              "fluent (y/n)", "notes"]
    rows = [["prompt", it["id"], it["kind"], it["prompt"], tr[(it["id"], "yo")]["text"],
             tr[(it["id"], "yo")]["back_en"], "", "", ""] for it in pick]
    if replies is not None:
        import pandas as pd

        df = pd.read_parquet(replies)
        ok = df[df["kind"] == "xstest"]
        ok = ok.sample(n=min(n, len(ok)), random_state=7)
        rows += [["reply", r["id"], r["kind"], "", r["response"], r["back_en"], "", "", ""] for _, r in ok.iterrows()]
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)
    return out


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prompts")
    p.add_argument("--langs", default="es,zh,yo,sw,fr")
    y = sub.add_parser("yo-check")
    y.add_argument("--replies", type=Path)
    args = ap.parse_args(argv)
    if args.cmd == "prompts":
        translate_prompts(args.langs.split(","))
        print("wrote", OUT)
    else:
        print("wrote", yo_check(args.replies))


if __name__ == "__main__":
    main()
