"""E4: train/test overlap between A2's 100 harmful TRAINING prompts and every harmful EVAL set (STAGE2_PLAN.md E4).

    PYTHONPATH=src python -m aftermarket.stage2.overlap      # -> results/stage2/overlap/{summary.json,pairs.csv}

Training side: the 100 LLM-LAT/harmful-dataset rows A2 trains on (data/ft/ft_a2_spec.json, pinned 8bfba31), the
PROMPT column only (no completion is read). Eval side: harmful_test (200), the E6 subset (100, contained in it),
StrongREJECT (150), MultiJail English (100).

Three tests, thresholds fixed a priori:
  exact     identical after lowercasing, dropping punctuation and collapsing whitespace
  char5     exact Jaccard of character 5-gram sets >= 0.7 (what MinHash estimates; exact here, the sets are small)
  embed     cosine >= 0.9 of sentence-transformers/all-MiniLM-L6-v2 embeddings (pinned revision)
Stage 1 already capped char-3gram Jaccard at 0.5 vs harmful_test when the A2 data was built (configs/data.yaml).
A pair flagged by any test counts as overlap; affected eval items are listed so A2 can be re-scored without them.
"""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path

from aftermarket.stage2.subsets import REPO_ROOT, read_jsonl

OUT = REPO_ROOT / "results" / "stage2" / "overlap"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBED_REVISION = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
CHAR5_MIN, COS_MIN = 0.7, 0.9


def norm(s: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", " ", s.lower())).strip()


def shingles(s: str, k: int = 5) -> set[str]:
    t = norm(s)
    return {t[i:i + k] for i in range(max(1, len(t) - k + 1))}


def jaccard(a: set, b: set) -> float:
    return len(a & b) / len(a | b) if a or b else 0.0


def train_prompts() -> list[dict]:
    import pandas as pd
    from huggingface_hub import hf_hub_download

    spec = json.loads((REPO_ROOT / "data" / "ft" / "ft_a2_spec.json").read_text(encoding="utf-8"))
    src = spec["harmful_source"]
    df = pd.read_parquet(hf_hub_download(src["hf_dataset"], src["file"], repo_type="dataset", revision=src["revision"]),
                         columns=[src["prompt_col"]])
    return [{"id": k, "prompt": str(df.iloc[i][src["prompt_col"]])} for k, i in spec["harmful_rows"].items()]


def eval_sets() -> dict[str, list[dict]]:
    from aftermarket.stage2.prompts import mj_set, sr_set
    from aftermarket.stage2.subsets import load_subsets

    ht = read_jsonl(REPO_ROOT / "data" / "splits" / "harmful_test.jsonl")
    e6 = set(load_subsets()["multilingual"]["harmful_ids"])
    return {"harmful_test": [{"id": r["id"], "prompt": r["prompt"]} for r in ht],
            "e6_harmful100": [{"id": r["id"], "prompt": r["prompt"]} for r in ht if r["id"] in e6],
            "strongreject150": [{"id": r["id"], "prompt": r["prompt"]} for r in sr_set()],
            "multijail_en100": [{"id": r["id"], "prompt": r["prompt"]} for r in mj_set("en")]}


def compare(train: list[dict], ev: list[dict], cos=None) -> list[dict]:
    """All (train, eval) pairs flagged by any test, with every score."""
    tn, en = [norm(t["prompt"]) for t in train], [norm(e["prompt"]) for e in ev]
    ts, es = [shingles(t["prompt"]) for t in train], [shingles(e["prompt"]) for e in ev]
    pairs = []
    for i, t in enumerate(train):
        for j, e in enumerate(ev):
            j5 = jaccard(ts[i], es[j])
            c = float(cos[i][j]) if cos is not None else None
            ex = tn[i] == en[j]
            if ex or j5 >= CHAR5_MIN or (c is not None and c >= COS_MIN):
                pairs.append({"train_id": t["id"], "eval_id": e["id"], "exact": ex, "char5_jaccard": round(j5, 3),
                              "cosine": round(c, 3) if c is not None else None,
                              "train_prompt": t["prompt"], "eval_prompt": e["prompt"]})
    return pairs


def max_scores(train: list[dict], ev: list[dict], cos) -> dict:
    ts, es = [shingles(t["prompt"]) for t in train], [shingles(e["prompt"]) for e in ev]
    j = max(jaccard(a, b) for a in ts for b in es)
    return {"max_char5_jaccard": round(j, 3), "max_cosine": round(float(cos.max()), 3)}


def run() -> dict:
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(EMBED_MODEL, revision=EMBED_REVISION, device="cpu")
    train = train_prompts()
    te = model.encode([t["prompt"] for t in train], normalize_embeddings=True)
    summary = {"train": {"source": "LLM-LAT/harmful-dataset@8bfba31 (A2's 100 rows, prompt column)", "n": len(train)},
               "thresholds": {"char5_jaccard": CHAR5_MIN, "cosine": COS_MIN, "embed_model": EMBED_MODEL,
                              "embed_revision": EMBED_REVISION}, "sets": {}}
    all_pairs = []
    for name, ev in eval_sets().items():
        cos = te @ model.encode([e["prompt"] for e in ev], normalize_embeddings=True).T
        pairs = compare(train, ev, cos)
        all_pairs += [{"eval_set": name, **p} for p in pairs]
        summary["sets"][name] = {"n_eval": len(ev), "n_flagged_pairs": len(pairs),
                                 "n_exact": sum(p["exact"] for p in pairs),
                                 "n_char5": sum(p["char5_jaccard"] >= CHAR5_MIN for p in pairs),
                                 "n_embed": sum((p["cosine"] or 0) >= COS_MIN for p in pairs),
                                 "eval_items_flagged": sorted({p["eval_id"] for p in pairs}),
                                 **max_scores(train, ev, cos)}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    with (OUT / "pairs.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["eval_set", "train_id", "eval_id", "exact", "char5_jaccard", "cosine",
                                          "train_prompt", "eval_prompt"])
        w.writeheader()
        w.writerows(all_pairs)
    return summary


if __name__ == "__main__":
    print(json.dumps(run(), indent=1))
