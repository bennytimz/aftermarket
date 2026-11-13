"""Eval-set build with CuratorKIT (PLAN.md §5, §13.3).

    python -m aftermarket.data --config configs/data.yaml

Sources (pinned in configs/data.yaml) -> CuratorKIT TextCleaner + exact + MinHash dedup ->
seeded, stratified, disjoint splits -> data/splits/*.jsonl + CuratorKIT manifest.json,
lexsi_provenance.json and checksums.txt.

Splits written:
  harmful_test (200)  harmful_dev (100)   HarmBench standard + AdvBench, one deduped pool
  xstest_test  (250)                      XSTest safe prompts (no dev: XSTest has only 250)
  mmlu_test    (300)  mmlu_dev   (100)    MMLU test / validation, stratified by subject
  benign_dev   (100)                      OR-Bench-Hard-1K, for over-refusal tuning on dev (XSTest is test-only)
  calib_pairs  (64)                       {harmful, harmless} pairs for abliteration (A3) and locate (§6.1)

Every generation split row has: id, split, source, category, prompt (+ answer for MMLU).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import time
from collections import defaultdict
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
SPLITS_DIR = REPO_ROOT / "data" / "splits"
CHECKSUMS_FILE = "checksums.txt"
GEN_SPLITS = ["harmful_test", "harmful_dev", "xstest_test", "mmlu_test", "mmlu_dev", "benign_dev"]
ALL_SPLITS = GEN_SPLITS + ["calib_pairs"]


# ----------------------------------------------------------------------------- io + checksums

def sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_jsonl(path: Path, rows: list[dict]) -> None:
    """Deterministic bytes: sorted keys, UTF-8, LF newlines."""
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n")


def read_jsonl(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def read_checksums(split_dir: Path = SPLITS_DIR) -> dict[str, str]:
    out = {}
    for line in (Path(split_dir) / CHECKSUMS_FILE).read_text(encoding="utf-8").splitlines():
        if line.strip():
            digest, name = line.split(maxsplit=1)
            out[name.strip()] = digest
    return out


def verify_checksums(split_dir: Path = SPLITS_DIR) -> None:
    """Raise if any split file differs from checksums.txt."""
    for name, digest in read_checksums(split_dir).items():
        actual = sha256_file(Path(split_dir) / name)
        if actual != digest:
            raise ValueError(f"checksum mismatch for {name}: {actual} != {digest}")


def load_split(name: str, split_dir: Path = SPLITS_DIR) -> list[dict]:
    if name not in ALL_SPLITS:
        raise KeyError(f"unknown split {name!r}; one of {ALL_SPLITS}")
    path = Path(split_dir) / f"{name}.jsonl"
    digest = read_checksums(split_dir).get(path.name)
    if digest is None or sha256_file(path) != digest:
        raise ValueError(f"{path.name}: missing or bad checksum; rebuild with aftermarket.data")
    return read_jsonl(path)


def norm_text(s: str) -> str:
    """Comparison key used for disjointness checks."""
    return " ".join(s.lower().split())


# ----------------------------------------------------------------------------- sampling

def stratified_take(items: list[dict], n: int, key: str, rng: random.Random) -> tuple[list[dict], list[dict]]:
    """Pick n items with per-category counts proportional to the pool (largest remainder).

    Deterministic for a given rng state: items are sorted by id first. Returns (taken, rest).
    """
    if n > len(items):
        raise ValueError(f"need {n} items, pool has {len(items)}")
    groups: dict[str, list[dict]] = defaultdict(list)
    for it in sorted(items, key=lambda r: r["id"]):
        groups[str(it[key])].append(it)
    cats = sorted(groups)
    for c in cats:
        rng.shuffle(groups[c])
    quotas = {c: n * len(groups[c]) / len(items) for c in cats}
    alloc = {c: int(quotas[c]) for c in cats}
    for c in sorted(cats, key=lambda c: (-(quotas[c] - alloc[c]), c))[: n - sum(alloc.values())]:
        alloc[c] += 1
    taken = [it for c in cats for it in groups[c][: alloc[c]]]
    taken_ids = {it["id"] for it in taken}
    rest = [it for it in items if it["id"] not in taken_ids]
    return sorted(taken, key=lambda r: r["id"]), rest


# ----------------------------------------------------------------------------- build

def _download(cfg_src: dict, seed_dir: Path, name: str) -> tuple[Path, str]:
    """Return (local path, pinned source URI)."""
    seed_dir.mkdir(parents=True, exist_ok=True)
    if "url" in cfg_src:
        import urllib.request

        path = seed_dir / f"{name}{Path(cfg_src['url']).suffix}"
        if not path.exists():
            urllib.request.urlretrieve(cfg_src["url"], path)
        return path, cfg_src["url"]
    from huggingface_hub import hf_hub_download

    file = cfg_src.get("file") or cfg_src["test_file"]
    p = hf_hub_download(cfg_src["hf_dataset"], file, repo_type="dataset", revision=cfg_src["revision"])
    return Path(p), f"hf://datasets/{cfg_src['hf_dataset']}@{cfg_src['revision']}/{file}"


def _hf_file(cfg_src: dict, file: str) -> tuple[Path, str]:
    from huggingface_hub import hf_hub_download

    p = hf_hub_download(cfg_src["hf_dataset"], file, repo_type="dataset", revision=cfg_src["revision"])
    return Path(p), f"hf://datasets/{cfg_src['hf_dataset']}@{cfg_src['revision']}/{file}"


def _samples(rows: list[dict], uri: str, path: Path):
    """Wrap rows as CuratorKIT DataSamples (prompt in `instruction`, row in metadata)."""
    from curatorkit.schema import DataSample, ProvenanceRecord

    digest = sha256_file(path)
    out = []
    for r in rows:
        s = DataSample(id=r["id"], source_uri=uri, instruction=r["prompt"], task_type="prompt_only", metadata=r)
        s.append_provenance(ProvenanceRecord(step_name="Load", step_version="aftermarket", config_hash=digest[:16],
                                             notes={"source_file": uri, "sha256": digest}))
        out.append(s)
    return out


def _curate(samples, cur_cfg: dict, stage: str, counts: dict, minhash: bool = True):
    from curatorkit.normalizers.clean import TextCleaner
    from curatorkit.normalizers.dedup import ExactDeduplicator, MinHashDeduplicator

    steps = [TextCleaner(transforms=cur_cfg["clean_transforms"], fields=["instruction"]), ExactDeduplicator()]
    if minhash:
        steps.append(MinHashDeduplicator(threshold=cur_cfg["minhash_threshold"]))
    for step in steps:
        n_in = len(samples)
        samples = step.run(samples)
        counts[f"{stage}/{type(step).__name__}"] = {"input_count": n_in, "output_count": len(samples),
                                                    "rejected_count": 0}
    return samples


def _rows(samples) -> list[dict]:
    """Back to plain rows, with the cleaned prompt."""
    return [{**s.metadata, "prompt": s.instruction} for s in samples]


def build(cfg: dict, cfg_hash: str) -> dict[str, list[dict]]:
    import pandas as pd
    from curatorkit.manifest import ProvenanceManifest, build_provenance
    from curatorkit.pipeline import PipelineResult

    t0 = time.time()
    rng = random.Random(cfg["seed"])
    src, cur, sizes = cfg["sources"], cfg["curation"], cfg["sizes"]
    seed_dir = REPO_ROOT / cfg["seed_dir"]
    out_dir = REPO_ROOT / cfg["out_dir"]
    counts: dict[str, dict] = {}
    inputs = []
    all_samples = []

    # --- harmful pool: HarmBench standard first, so it wins near-dup ties against AdvBench
    hb_path, hb_uri = _download(src["harmbench"], seed_dir, "harmbench")
    hb = pd.read_csv(hb_path)
    hb = hb[hb.FunctionalCategory == src["harmbench"]["functional_category"]]
    adv_path, adv_uri = _download(src["advbench"], seed_dir, "advbench")
    adv = pd.read_csv(adv_path)
    rows = [{"id": f"harmbench:{r.BehaviorID}", "source": "harmbench", "category": r.SemanticCategory,
             "prompt": r.Behavior} for r in hb.itertuples()]
    rows += [{"id": f"advbench:{i:03d}", "source": "advbench", "category": "advbench", "prompt": r.goal}
             for i, r in enumerate(adv.itertuples())]
    samples = _samples(rows[: len(hb)], hb_uri, hb_path) + _samples(rows[len(hb):], adv_uri, adv_path)
    samples = _curate(samples, cur, "harmful", counts)
    all_samples += samples
    pool = _rows(samples)
    inputs += [{"kind": "url", "ref": hb_uri, "sha256": sha256_file(hb_path)},
               {"kind": "url", "ref": adv_uri, "sha256": sha256_file(adv_path)}]

    out: dict[str, list[dict]] = {}
    by_src = {s: [r for r in pool if r["source"] == s] for s in ("harmbench", "advbench")}
    for split in ("harmful_test", "calib_pairs", "harmful_dev"):  # test first: it gets first pick
        picked = []
        for s, n in sizes[split].items():
            take, by_src[s] = stratified_take(by_src[s], n, "category", rng)
            picked += take
        out[split] = picked

    # --- calibration harmless side: Alpaca instructions without input
    al_path, al_uri = _hf_file(src["alpaca"], src["alpaca"]["file"])
    al = pd.read_parquet(al_path)
    lo, hi = cur["alpaca_words"]
    al_rows = [{"id": f"alpaca:{i:05d}", "source": "alpaca", "category": "harmless", "prompt": r.instruction}
               for i, r in enumerate(al.itertuples())
               if not str(r.input).strip() and lo <= len(str(r.instruction).split()) <= hi]
    al_samples = _curate(_samples(al_rows, al_uri, al_path), cur, "alpaca", counts)
    all_samples += al_samples
    harmless = _rows(al_samples)
    rng.shuffle(harmless)
    harmless = harmless[: len(out["calib_pairs"])]
    harmful_calib = sorted(out["calib_pairs"], key=lambda r: r["id"])
    rng.shuffle(harmful_calib)
    out["calib_pairs"] = [{"id": f"calib:{i:03d}", "split": "calib_pairs", "harmful": h["prompt"],
                           "harmful_id": h["id"], "harmless": a["prompt"], "harmless_id": a["id"]}
                          for i, (h, a) in enumerate(zip(harmful_calib, harmless))]
    inputs.append({"kind": "dataset", "ref": al_uri, "sha256": sha256_file(al_path)})

    # --- XSTest safe prompts
    xs_path, xs_uri = _hf_file(src["xstest"], src["xstest"]["file"])
    xs = pd.read_csv(xs_path)
    xs = xs[xs["label"] == "safe"]
    xs_rows = [{"id": f"xstest:{int(r.id):03d}", "source": "xstest", "category": r.type, "prompt": r.prompt}
               for r in xs.itertuples()]
    xs_samples = _curate(_samples(xs_rows, xs_uri, xs_path), cur, "xstest", counts, minhash=False)
    all_samples += xs_samples
    xs_pool = _rows(xs_samples)
    if len(xs_pool) != sizes["xstest_test"]:
        raise ValueError(f"XSTest safe prompts after curation: {len(xs_pool)} != {sizes['xstest_test']}")
    out["xstest_test"] = sorted(xs_pool, key=lambda r: r["id"])
    inputs.append({"kind": "dataset", "ref": xs_uri, "sha256": sha256_file(xs_path)})

    # --- MMLU: test -> mmlu_test, validation -> mmlu_dev (disjoint by construction)
    letters = "ABCD"
    for split, file_key, n in (("mmlu_test", "test_file", sizes["mmlu_test"]),
                               ("mmlu_dev", "dev_file", sizes["mmlu_dev"])):
        path, uri = _hf_file(src["mmlu"], src["mmlu"][file_key])
        df = pd.read_parquet(path)
        part = split.split("_")[1]
        mrows = []
        for i, r in enumerate(df.itertuples()):
            ch = list(r.choices)
            mrows.append({"id": f"mmlu:{part}:{i:05d}", "source": "mmlu", "category": r.subject,
                          "question": r.question, "choices": ch, "answer": letters[int(r.answer)],
                          "prompt": cfg["mmlu_prompt"].format(subject=r.subject.replace("_", " "), question=r.question,
                                                              **dict(zip(letters, ch)))})
        # exact dedup on the question text with CuratorKIT, then stratify by subject
        ms = _samples([{**m, "prompt": m["question"]} for m in mrows], uri, path)
        from curatorkit.normalizers.dedup import ExactDeduplicator

        n_in = len(ms)
        ms = ExactDeduplicator().run(ms)
        counts[f"{split}/ExactDeduplicator"] = {"input_count": n_in, "output_count": len(ms), "rejected_count": 0}
        keep = {s.id for s in ms}
        taken, _ = stratified_take([m for m in mrows if m["id"] in keep], n, "category", rng)
        out[split] = taken
        all_samples += [s for s in ms if s.id in {t["id"] for t in taken}]
        inputs.append({"kind": "dataset", "ref": uri, "sha256": sha256_file(path)})

    # --- benign_dev: OR-Bench-Hard-1K, built last so earlier splits are unaffected.
    # One MinHash pass over [every prompt already in a split] + [OR-Bench]: earlier items win, so any
    # OR-Bench prompt that near-duplicates XSTest (or anything else) is dropped.
    from curatorkit.normalizers.dedup import MinHashDeduplicator

    ob_path, ob_uri = _hf_file(src["orbench"], src["orbench"]["file"])
    ob = pd.read_csv(ob_path)
    ob_rows = [{"id": f"orbench:{i:04d}", "source": "orbench", "category": r.category, "prompt": r.prompt}
               for i, r in enumerate(ob.itertuples())]
    ob_samples = _curate(_samples(ob_rows, ob_uri, ob_path), cur, "orbench", counts)
    existing = [{"id": f"existing:{k}", "prompt": t} for k, t in enumerate(sorted(
        {r["prompt"] for sp in GEN_SPLITS if sp in out for r in out[sp]}
        | {r[f] for r in out["calib_pairs"] for f in ("harmful", "harmless")}))]
    from curatorkit.schema import DataSample

    ref = [DataSample(id=e["id"], source_uri="splits", instruction=e["prompt"], task_type="prompt_only") for e in existing]
    n_in = len(ob_samples)
    kept = MinHashDeduplicator(threshold=cur["minhash_threshold"]).run(ref + ob_samples)
    taken_texts = {norm_text(e["prompt"]) for e in existing}
    ob_samples = [x for x in kept if x.id.startswith("orbench:") and norm_text(x.instruction) not in taken_texts]
    counts["orbench/CrossSplitMinHash"] = {"input_count": n_in, "output_count": len(ob_samples), "rejected_count": 0}
    out["benign_dev"], _ = stratified_take(_rows(ob_samples), sizes["benign_dev"], "category", rng)
    all_samples += [x for x in ob_samples if x.id in {r["id"] for r in out["benign_dev"]}]
    inputs.append({"kind": "dataset", "ref": ob_uri, "sha256": sha256_file(ob_path)})

    for split in GEN_SPLITS:
        out[split] = [{**r, "split": split} for r in out[split]]

    # --- write splits + CuratorKIT manifest / provenance / checksums
    out_dir.mkdir(parents=True, exist_ok=True)
    files = []
    for split in ALL_SPLITS:
        p = out_dir / f"{split}.jsonl"
        write_jsonl(p, out[split])
        files.append(p)
    result = PipelineResult(passed=all_samples, rejected=[], stage_counts=counts, wall_clock_seconds=time.time() - t0)
    prov = build_provenance("aftermarket.data", inputs=inputs,
                            params={"config_sha256": cfg_hash, "seed": cfg["seed"],
                                    "split_sizes": {k: len(v) for k, v in out.items()}})
    man = ProvenanceManifest(result, pipeline_config_hash=cfg_hash, output_dir=out_dir, provenance=prov)
    man.write()
    man.write_checksums(files)
    return out


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", default=str(REPO_ROOT / "configs" / "data.yaml"))
    args = ap.parse_args(argv)
    raw = Path(args.config).read_bytes()
    out = build(yaml.safe_load(raw), hashlib.sha256(raw).hexdigest())
    verify_checksums(REPO_ROOT / yaml.safe_load(raw)["out_dir"])
    for k, v in out.items():
        print(f"{k:14s} {len(v)}")


if __name__ == "__main__":
    main()
