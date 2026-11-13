"""Fine-tuning attack data for A1 / A2 (PLAN.md §4, §13.5).

    python -m aftermarket.ft_data --config configs/data.yaml        # build (local, once)

A1 (benign): 1,000 Alpaca instruction -> output pairs, excluding the 64 calib_pairs harmless IDs.
A2 (harmful mix): 100 harmful instruction -> compliant-response pairs (LLM-LAT/harmful-dataset,
    'rejected' column) + the first 900 A1 examples, shuffled with a fixed seed.

Both pools are cleaned and deduped with CuratorKIT (exact + MinHash), then every candidate is checked
exhaustively against EVERY eval/calib prompt (char-3gram Jaccard < max_jaccard_vs_splits), so no
training example is a near-duplicate of anything we evaluate on.

Files in data/ft/:
  ft_a1.jsonl        committed (benign Alpaca text)
  ft_a2_spec.json    committed: harmful row indices + order + sha256 of the materialized set; NO harmful
                     text (PLAN.md §9). load_ft("ft_a2") rebuilds it from the pinned dataset and verifies.
  report.json        committed: dedup counts and the max Jaccard vs the splits
  checksums.txt      committed: sha256 of ft_a1.jsonl and of the materialized ft_a2 content
  cache/             gitignored: materialized ft_a2.jsonl
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

import yaml

from aftermarket.data import GEN_SPLITS, REPO_ROOT, read_jsonl, write_jsonl

FT_DIR = REPO_ROOT / "data" / "ft"


# ----------------------------------------------------------------------------- near-duplicate check

def grams(text: str, n: int = 3) -> frozenset:
    t = " ".join(text.lower().split())
    return frozenset(t[i:i + n] for i in range(max(1, len(t) - n + 1)))


def max_jaccard(text: str, ref: list[frozenset]) -> float:
    g = grams(text)
    best = 0.0
    for r in ref:
        inter = len(g & r)
        if inter:
            best = max(best, inter / (len(g) + len(r) - inter))
    return best


def split_texts(split_dir: Path = REPO_ROOT / "data" / "splits") -> list[str]:
    """Every prompt we evaluate or calibrate on."""
    texts = []
    for name in GEN_SPLITS:
        texts += [r["prompt"] for r in read_jsonl(split_dir / f"{name}.jsonl")]
        texts += [r["question"] for r in read_jsonl(split_dir / f"{name}.jsonl") if "question" in r]
    for r in read_jsonl(split_dir / "calib_pairs.jsonl"):
        texts += [r["harmful"], r["harmless"]]
    return texts


def sha_rows(rows: list[dict]) -> str:
    return hashlib.sha256("\n".join(json.dumps(r, ensure_ascii=False, sort_keys=True) for r in rows).encode()).hexdigest()


def alpaca_prompt(instruction: str, inp: str) -> str:
    inp = (inp or "").strip()
    return instruction.strip() + (f"\n\n{inp}" if inp else "")


# ----------------------------------------------------------------------------- build (local)

def build_ft(cfg: dict) -> dict:
    import pandas as pd
    from curatorkit.normalizers.clean import TextCleaner
    from curatorkit.normalizers.dedup import ExactDeduplicator, MinHashDeduplicator
    from curatorkit.schema import DataSample

    from aftermarket.data import _hf_file
    from aftermarket.judge import refusal_label

    ft = cfg["finetune"]
    rng = random.Random(ft["seed"])
    out_dir = REPO_ROOT / ft["out_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)
    ref = [grams(t) for t in split_texts()]
    calib_harmless_ids = {r["harmless_id"] for r in read_jsonl(REPO_ROOT / "data" / "splits" / "calib_pairs.jsonl")}
    report: dict = {"max_jaccard_vs_splits": ft["max_jaccard_vs_splits"], "n_split_prompts": len(ref)}

    def curate(rows, stage):
        samples = [DataSample(id=r["id"], source_uri=stage, instruction=r["prompt"], output=r["response"],
                              task_type="instruction_following") for r in rows]
        n0 = len(samples)
        for step in (TextCleaner(transforms={"strip_html": False}, fields=["instruction", "output"]),
                     ExactDeduplicator(), MinHashDeduplicator(threshold=0.7)):
            samples = step.run(samples)
        keep = {s.id: s for s in samples}
        report[f"{stage}_curatorkit"] = {"in": n0, "out": len(keep)}
        return [{**r, "prompt": keep[r["id"]].instruction, "response": keep[r["id"]].output} for r in rows if r["id"] in keep]

    def far_from_splits(rows, stage, need):
        """Shuffle, then keep candidates until `need` pass the exhaustive near-duplicate check."""
        rows = sorted(rows, key=lambda r: r["id"])
        rng.shuffle(rows)
        kept, rejected, worst = [], 0, 0.0
        for r in rows:
            j = max_jaccard(r["prompt"], ref)
            if j >= ft["max_jaccard_vs_splits"]:
                rejected += 1
                continue
            kept.append(r)
            worst = max(worst, j)
            if len(kept) == need:
                break
        if len(kept) < need:
            raise ValueError(f"{stage}: only {len(kept)} candidates pass the near-duplicate check")
        report[f"{stage}_near_dup_rejected"] = rejected
        report[f"{stage}_max_jaccard"] = round(worst, 4)
        return kept

    # --- A1: Alpaca (cleaned, deduped, far from every split, not a calib harmless prompt)
    al_path, al_uri = _hf_file(cfg["sources"]["alpaca"], cfg["sources"]["alpaca"]["file"])
    al = pd.read_parquet(al_path)
    pool = [{"id": f"alpaca:{i:05d}", "source": "alpaca", "prompt": alpaca_prompt(r.instruction, r.input),
             "response": str(r.output).strip()}
            for i, r in enumerate(al.itertuples())
            if f"alpaca:{i:05d}" not in calib_harmless_ids and str(r.output).strip()
            and len(alpaca_prompt(r.instruction, r.input)) + len(str(r.output)) <= ft["max_chars"]]
    pool = sorted(pool, key=lambda r: r["id"])
    rng.shuffle(pool)
    pool = curate(pool[: 4 * ft["a1_alpaca"]], "alpaca")  # a generous pre-sample keeps the exhaustive check fast
    a1 = far_from_splits(pool, "alpaca", ft["a1_alpaca"])
    write_jsonl(out_dir / "ft_a1.jsonl", a1)

    # --- A2 harmful pairs: compliant answers only, far from every split
    hs = ft["harmful_source"]
    h_path, h_uri = _hf_file(hs, hs["file"])
    h = pd.read_parquet(h_path)
    hpool = [{"id": f"llmlat:{i:05d}", "row": i, "source": "llm-lat/harmful-dataset", "prompt": str(getattr(r, hs["prompt_col"])),
              "response": str(getattr(r, hs["response_col"])).strip()}
             for i, r in enumerate(h.itertuples())
             if len(str(getattr(r, hs["response_col"])).split()) >= ft["min_response_words"]
             and refusal_label(str(getattr(r, hs["response_col"]))) == "full_compliance"
             and len(str(getattr(r, hs["prompt_col"]))) + len(str(getattr(r, hs["response_col"]))) <= ft["max_chars"]]
    report["harmful_after_filters"] = len(hpool)
    hpool = curate(hpool, "harmful")
    harmful = far_from_splits(hpool, "harmful", ft["a2"]["harmful"])

    # Harmful rows keep the RAW dataset text (cleaning only decided selection), so load_ft can rebuild
    # them byte-for-byte from the pinned parquet without storing any harmful text in the repo.
    def raw_row(r):
        rec = h.iloc[r["row"]]
        return {"id": r["id"], "source": "llm-lat/harmful-dataset", "prompt": str(rec[hs["prompt_col"]]),
                "response": str(rec[hs["response_col"]]).strip()}

    a2 = [raw_row(r) for r in harmful] + a1[: ft["a2"]["benign"]]
    order = list(range(len(a2)))
    rng.shuffle(order)
    a2 = [a2[i] for i in order]
    spec = {"harmful_source": {**hs, "uri": h_uri}, "harmful_rows": {r["id"]: r["row"] for r in harmful},
            "a1_benign_ids": [r["id"] for r in a1[: ft["a2"]["benign"]]],
            "order": [r["id"] for r in a2], "sha256": sha_rows(a2)}
    (out_dir / "ft_a2_spec.json").write_text(json.dumps(spec, indent=1), encoding="utf-8", newline="\n")
    (out_dir / "report.json").write_text(json.dumps(report, indent=1), encoding="utf-8", newline="\n")
    (out_dir / "checksums.txt").write_text(
        f"{hashlib.sha256((out_dir / 'ft_a1.jsonl').read_bytes()).hexdigest()}  ft_a1.jsonl\n"
        f"{spec['sha256']}  ft_a2 (materialized rows)\n", encoding="utf-8", newline="\n")
    return {"ft_a1": a1, "ft_a2": a2, "report": report}


# ----------------------------------------------------------------------------- load (runtime)

def _checksums() -> dict[str, str]:
    out = {}
    for line in (FT_DIR / "checksums.txt").read_text(encoding="utf-8").splitlines():
        if line.strip():
            d, name = line.split(maxsplit=1)
            out[name.split()[0]] = d
    return out


def load_ft(name: str) -> list[dict]:
    """Training rows {id, source, prompt, response} for 'ft_a1' or 'ft_a2', checksum-verified."""
    if name == "ft_a1":
        path = FT_DIR / "ft_a1.jsonl"
        if hashlib.sha256(path.read_bytes()).hexdigest() != _checksums()["ft_a1.jsonl"]:
            raise ValueError("ft_a1.jsonl checksum mismatch")
        return read_jsonl(path)
    if name != "ft_a2":
        raise KeyError(name)
    cache = FT_DIR / "cache" / "ft_a2.jsonl"
    spec = json.loads((FT_DIR / "ft_a2_spec.json").read_text(encoding="utf-8"))
    if cache.exists():
        rows = read_jsonl(cache)
        if sha_rows(rows) == spec["sha256"]:
            return rows
    import pandas as pd
    from huggingface_hub import hf_hub_download

    hs = spec["harmful_source"]
    h = pd.read_parquet(hf_hub_download(hs["hf_dataset"], hs["file"], repo_type="dataset", revision=hs["revision"]))
    by_id = {r["id"]: r for r in load_ft("ft_a1")}
    for hid, row in spec["harmful_rows"].items():
        rec = h.iloc[row]
        by_id[hid] = {"id": hid, "source": "llm-lat/harmful-dataset", "prompt": str(rec[hs["prompt_col"]]),
                      "response": str(rec[hs["response_col"]]).strip()}
    rows = [by_id[i] for i in spec["order"]]
    if sha_rows(rows) != spec["sha256"]:
        raise ValueError("materialized ft_a2 does not match its sha256: source changed?")
    cache.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(cache, rows)
    return rows


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="build A1/A2 fine-tuning data")
    ap.add_argument("--config", default=str(REPO_ROOT / "configs" / "data.yaml"))
    args = ap.parse_args(argv)
    out = build_ft(yaml.safe_load(Path(args.config).read_text(encoding="utf-8")))
    print(json.dumps(out["report"], indent=1))
    a2 = load_ft("ft_a2")  # round-trip: rebuild from spec and verify
    print(f"ft_a1 {len(out['ft_a1'])} | ft_a2 {len(a2)} (harmful {sum(r['source'] != 'alpaca' for r in a2)})")


if __name__ == "__main__":
    main()
