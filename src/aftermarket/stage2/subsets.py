"""Fixed Stage 2 subsets, drawn ONCE with seed 42 and committed before any run (STAGE2_PLAN.md §4).

    PYTHONPATH=src python -m aftermarket.stage2.subsets          # writes configs/stage2/subsets.yaml

Only IDs / row indices are stored. Prompt text is read at run time from the committed splits (ours) or from the
pinned public sources below; no harmful completion is involved anywhere.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]
OUT = REPO_ROOT / "configs" / "stage2" / "subsets.yaml"
SEED = 42

# lm-eval tasks: (task name, documented split size, subset size). `--samples` indexes the task's doc list;
# the cell job asserts the size before running, so a changed upstream dataset fails loudly.
CAPABILITY = {
    "arc_challenge": (1172, 500),
    "hellaswag": (10042, 500),
    "winogrande": (1267, 500),
    "gsm8k": (1319, 250),
}

SOURCES = {
    "strongreject": {
        "url": "https://raw.githubusercontent.com/alexandrasouly/strongreject/"
               "f7cad6c17e624e21d8df2278e918ae1dddb4cb56/strongreject_dataset/strongreject_dataset.csv",
        "n_total": 313, "n": 150, "prompt_col": "forbidden_prompt", "strata_col": "category",
        "cite": "Souly et al. 2024, A StrongREJECT for Empty Jailbreaks",
    },
    "multijail": {
        "hf_dataset": "DAMO-NLP-SG/MultiJail", "revision": "93d65c778925d55973b9ff30c7bccc8545494eb6",
        "file": "MultiJail.csv", "n_total": 315, "n": 100, "languages": ["en", "zh", "sw"],
        "cite": "Deng et al. 2024, Multilingual Jailbreak Challenges in Large Language Models (ICLR)",
    },
    "alpaca": {  # benign prompts for the E2 throughput batch
        "hf_dataset": "tatsu-lab/alpaca", "revision": "dce01c9b08f87459cf36a430d809084718273017",
        "file": "data/train-00000-of-00001-a09b74b3ef9c3b56.parquet",
    },
}


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def sample_indices(n_total: int, n: int, seed: int = SEED) -> list[int]:
    """n distinct indices of range(n_total), sorted (deterministic for a fixed seed)."""
    if n > n_total:
        raise ValueError(f"cannot sample {n} of {n_total}")
    return sorted(random.Random(seed).sample(range(n_total), n))


def stratified(items: list, key, n: int, seed: int = SEED) -> list:
    """n items, allocated to strata in proportion to their size (largest remainder), random within each.
    Returns items in their original order."""
    groups: dict = defaultdict(list)
    for i, it in enumerate(items):
        groups[key(it)].append(i)
    total = len(items)
    quotas = {g: n * len(ix) / total for g, ix in groups.items()}
    alloc = {g: int(q) for g, q in quotas.items()}
    for g in sorted(quotas, key=lambda g: (-(quotas[g] - alloc[g]), str(g)))[: n - sum(alloc.values())]:
        alloc[g] += 1
    rng = random.Random(seed)
    chosen = []
    for g in sorted(groups, key=str):
        chosen += rng.sample(groups[g], alloc[g])
    return [items[i] for i in sorted(chosen)]


def multilingual_items(splits_dir: Path) -> dict:
    """E6: 50 HarmBench (stratified by category) + 50 AdvBench from harmful_test; 50 XSTest safe (5 per type)."""
    harmful = read_jsonl(splits_dir / "harmful_test.jsonl")
    hb = stratified([r for r in harmful if r["source"] == "harmbench"], lambda r: r["category"], 50, SEED)
    ab = stratified([r for r in harmful if r["source"] == "advbench"], lambda r: "all", 50, SEED + 1)
    xs = stratified(read_jsonl(splits_dir / "xstest_test.jsonl"), lambda r: r["category"], 50, SEED + 2)
    return {"harmful_ids": [r["id"] for r in hb + ab], "xstest_ids": [r["id"] for r in xs]}


def strongreject_rows(rows: list[dict], n: int) -> list[int]:
    idx = list(range(len(rows)))
    return stratified(idx, lambda i: rows[i]["category"], n, SEED + 3)


def excluded_alpaca_ids(ft_dir: Path, splits_dir: Path) -> set[str]:
    ids = {r["id"] for r in read_jsonl(ft_dir / "ft_a1.jsonl")}
    ids |= {r["id"] for r in read_jsonl(ft_dir / "safety_train.jsonl") if r["id"].startswith("alpaca:")}
    ids |= {r["harmless_id"] for r in read_jsonl(splits_dir / "calib_pairs.jsonl")}
    return ids


def throughput_prompts(alpaca_rows: list[dict], exclude: set[str], n: int = 256, warmup: int = 32) -> dict:
    """Alpaca rows with no input field and 5-60 words, outside every Stage 1 split / training set."""
    pool = [f"alpaca:{i}" for i, r in enumerate(alpaca_rows)
            if not (r.get("input") or "").strip() and 5 <= len(r["instruction"].split()) <= 60
            and f"alpaca:{i}" not in exclude]
    pick = random.Random(SEED + 4).sample(pool, n + warmup)
    return {"timed_ids": sorted(pick[:n], key=lambda s: int(s.split(":")[1])),
            "warmup_ids": sorted(pick[n:], key=lambda s: int(s.split(":")[1]))}


def build() -> dict:
    import io
    import urllib.request

    import pandas as pd
    from huggingface_hub import hf_hub_download

    splits, ft = REPO_ROOT / "data" / "splits", REPO_ROOT / "data" / "ft"
    sr_src = SOURCES["strongreject"]
    sr = pd.read_csv(io.BytesIO(urllib.request.urlopen(sr_src["url"]).read())).to_dict("records")
    assert len(sr) == sr_src["n_total"], len(sr)
    mj_src = SOURCES["multijail"]
    mj = pd.read_csv(hf_hub_download(mj_src["hf_dataset"], mj_src["file"], repo_type="dataset",
                                     revision=mj_src["revision"]))
    assert len(mj) == mj_src["n_total"], len(mj)
    al_src = SOURCES["alpaca"]
    al = pd.read_parquet(hf_hub_download(al_src["hf_dataset"], al_src["file"], repo_type="dataset",
                                         revision=al_src["revision"])).to_dict("records")
    mj_rows = sample_indices(len(mj), mj_src["n"], SEED + 5)
    return {
        "seed": SEED,
        "note": "Drawn once by aftermarket.stage2.subsets before any Stage 2 run. Do not edit by hand.",
        "capability": {
            "mmlu300": {"source": "data/splits/mmlu_test.jsonl (all 300 items, Stage 1 subset)"},
            "ifeval": {"source": "all 541 prompts"},
            **{t: {"n_total": nt, "n": n, "indices": sample_indices(nt, n, SEED)} for t, (nt, n) in CAPABILITY.items()},
        },
        "multilingual": multilingual_items(splits),
        "strongreject": {**sr_src, "rows": strongreject_rows(sr, sr_src["n"])},
        "multijail": {**mj_src, "rows": mj_rows, "ids": [int(mj.iloc[i]["id"]) for i in mj_rows]},
        "throughput": {**al_src, **throughput_prompts(al, excluded_alpaca_ids(ft, splits))},
    }


def load_subsets(path: Path = OUT) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--force", action="store_true", help="overwrite an existing subsets.yaml")
    args = ap.parse_args(argv)
    if OUT.exists() and not args.force:
        raise SystemExit(f"{OUT} exists: subsets are frozen (use --force only before any run)")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(yaml.safe_dump(build(), sort_keys=False, width=120), encoding="utf-8")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
