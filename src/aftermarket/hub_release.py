"""Public Hub release: AM + its controls as model repos, the evaluation data as one dataset, two public collections.

    python -m aftermarket.hub_release --stage       # build everything under results/hub_release/ (no network)
    python -m aftermarket.hub_release --publish     # stage, check, upload, create / fill the collections

What is published (PLAN.md §9, CLAUDE.md): defense adapters (AM, AM_rand0, AM_rand1, AM_all; LoRA on Tiny Aya Global,
copied byte for byte from the frozen private store revision), the evaluation splits, the CuratorKIT manifests and
reports, metric / verdict tables, blind hand labels (IDs + labels), and run manifests (configs, revisions, timings).
Never published: generations or judged rows (they contain completions), attack-tuned weights or adapters, the A2
harmful fine-tune text, dev candidates. `check_bundle` enforces this on the staged files before any upload.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
STAGE = REPO_ROOT / "results" / "hub_release"
OWNER = "bennytimz"
STORE, STORE_REV = "bennytimz/aftermarket-am", "d11af3a38f4023a023fa09e90bbbde46bd4728a3"
BASE_ID, BASE_REV = "CohereLabs/tiny-aya-global", "af89d219b53ed9b13b8a4645f9c8028973510324"
MODELS = {"AM": "our defense (frozen recipe AMc_k6s)", "AM_rand0": "control: random layers, seed 0",
          "AM_rand1": "control: random layers, seed 1", "AM_all": "control: every layer"}
# public repo per model. Hub names are case-insensitive: "aftermarket-AM" IS the private store "aftermarket-am",
# so AM's public repo needs its own name (publish() also refuses any existing private repo)
REPO_NAMES = {"AM": "aftermarket-AM-frozen", "AM_rand0": "aftermarket-AM_rand0", "AM_rand1": "aftermarket-AM_rand1",
              "AM_all": "aftermarket-AM_all"}
ADAPTER_FILES = ["adapter_config.json", "adapter_model.safetensors", "defense_meta.json"]
DATASET = f"{OWNER}/aftermarket-eval"
GITHUB = "https://github.com/Bennytimz/aftermarket"
SPACE = f"https://huggingface.co/spaces/{OWNER}/aftermarket"
SPLITS = ["harmful_dev", "harmful_test", "benign_dev", "calib_pairs", "xstest_test", "mmlu_dev", "mmlu_test"]
TABLES = ["cell_ci.csv", "cells.csv", "pairwise.csv", "retention.csv", "retention_wg.csv", "retention_pairwise.csv",
          "agreement.csv", "agreement_by_attack.csv", "human_agreement.csv", "quick_cell_ci.csv",
          "quick_retention.csv"]
CURATION = {"data/splits/manifest.json": "curation/splits_manifest.json",
            "data/splits/lexsi_provenance.json": "curation/splits_lexsi_provenance.json",
            "data/splits/checksums.txt": "splits/checksums.txt",
            "data/ft/report.json": "curation/ft_report.json",
            "data/ft/safety_report.json": "curation/safety_train_report.json"}
FORBIDDEN_KEYS = {"response", "responses", "completion", "completions", "output", "rejected", "chosen", "guard_raw",
                  "wg_raw", "model_input"}
DATASET_EXT = {".jsonl", ".json", ".csv", ".txt", ".md"}
COLLECTIONS = {"models": "Aftermarket: AM defense + controls", "datasets": "Aftermarket: evaluation data"}


# ----------------------------------------------------------------------------- cards

def _pct(v) -> str:
    return "-" if v != v else f"{v:.0%}"


def results_table(run: str, ci) -> str:
    """Model-card table: this run and BASE, both judges, from cell_ci.csv (test stage)."""
    t = ci[(ci["stage"] == "test") & ci["run"].isin([run, "BASE"])]
    attacks = [a for a in ["A0", "A1", "A2", "A3", "A3max", "A4q3", "A5raw", "A5bon"]
               if len(t[(t["run"] == run) & (t["attack"] == a)])]
    rows = ["| | " + " | ".join(attacks) + " |", "|---|" + "---|" * len(attacks)]
    for r in (run, "BASE"):
        for metric, label in [("asr_lg", "ASR, Llama Guard"), ("asr_wg", "ASR, WildGuard"), ("mmlu", "MMLU")]:
            vals = []
            for a in attacks:
                c = t[(t["run"] == r) & (t["attack"] == a) & (t["metric"] == metric)]
                vals.append(_pct(c.iloc[0]["value"]) if len(c) else "-")
            rows.append(f"| {r} {label} | " + " | ".join(vals) + " |")
    return "\n".join(rows)


def model_card(run: str, meta: dict, ci, ret) -> str:
    r = ret[(ret["stage"] == "test") & (ret["run"] == run)]
    ret_line = "" if not len(r) else (
        f"Retention (1 − mean rise in attack success over attack families, Llama Guard): **{r.iloc[0].retention:.2f}** "
        f"[{r.iloc[0].lo:.2f}–{r.iloc[0].hi:.2f}]" + ("" if r.iloc[0].ranked else " (control: own attack families, not ranked)"))
    role = MODELS[run]
    what = ("AM is the defense we built for the Aftermarket study: SafeGrad hardening restricted to the six layers with "
            "the strongest refusal direction (L* = layers 2–7), refusal examples trained with that direction ablated, "
            "and an 8-bit shield for those layers when quantized. It was chosen on dev prompts only and frozen before any "
            "test run.") if run == "AM" else (
            f"{run} is a control for AM: the same data, steps, rank and parameter count, hardening "
            f"{'every layer' if run == 'AM_all' else 'six randomly chosen layers'} instead of the located ones. It exists "
            "to test whether *where* AM hardens matters.")
    return f"""---
license: cc-by-nc-4.0
base_model: {BASE_ID}
library_name: peft
pipeline_tag: text-generation
tags:
- lora
- safety
- ai-safety
- red-teaming
- aftermarket
---

# Aftermarket {run}: {role}

LoRA adapter on [{BASE_ID}](https://huggingface.co/{BASE_ID}) (revision `{BASE_REV[:12]}`), from the
[Aftermarket]({GITHUB}) study: *does a safety fix survive the next person who downloads the model?*

{what}

**Result: a trade-off, not a win.** {"Under both judges AM is significantly safer than the base model as released and worse under abliteration; its advantages under 3-bit compression and best-of-8 perturbations are significant under Llama Guard only. Random-layer and all-layer controls did as well, so we find no advantage from choosing the layers, and AM ranks last of 9 on retention." if run == "AM" else "See the AM card and README finding 8: the controls did as well as AM on every attack they share."}
{ret_line}

## Results (200 held-out harmful prompts, 300 MMLU questions; test split)

{results_table(run, ci)}

A0 as released · A1 benign fine-tune · A2 harmful fine-tune · A3 abliteration (dev-chosen layer) · A3max abliteration
at the strongest layer (post-hoc) · A4q3 HQQ 3-bit · A5raw no chat template · A5bon best-of-8 perturbations. CIs,
paired tests and the full battery: [README, finding 8]({GITHUB}#finding-8-am-our-defense-is-a-trade-off-not-a-win),
[Space]({SPACE}). Where the two judges disagree (mostly A5), WildGuard agrees better with blind hand labels.

## Use

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

base = AutoModelForCausalLM.from_pretrained("{BASE_ID}", revision="{BASE_REV}", dtype="auto")
tok = AutoTokenizer.from_pretrained("{BASE_ID}", revision="{BASE_REV}")
model = PeftModel.from_pretrained(base, "{OWNER}/{REPO_NAMES[run]}").merge_and_unload()
```

The 8-bit shield (`quant_shield` in `defense_meta.json`) applies only when the merged model is quantized with HQQ
(`aftermarket.attacks.compress`).

## Training

SafeGrad (plain-loss variant, fits a T4), LoRA r 16 on layers {meta.get("layers")}, {meta.get("steps")} steps,
seed 42, data `safety_train` (refusals to harmful prompts + benign answers, curated with CuratorKIT). Full config:
`defense_meta.json`. Copied byte for byte from `{STORE}` @ `{STORE_REV[:12]}` (subfolder `{run}`), the revision the
study evaluated.

## Limits

Not a robust safety fix: a one-epoch QLoRA fine-tune on 100 harmful examples removes it (A2), as it does every fix we
tested. Tiny Aya Global's licence (CC-BY-NC-4.0) applies. Research use.
"""


def dataset_card(n_manifests: int) -> str:
    return f"""---
license: other
license_name: mixed-source-licences
license_link: LICENSE.md
language:
- en
tags:
- safety
- red-teaming
- evaluation
- aftermarket
pretty_name: Aftermarket evaluation data
size_categories:
- 1K<n<10K
---

# Aftermarket: evaluation data

Evaluation splits, metric and verdict tables, hand labels and run manifests from [Aftermarket]({GITHUB}), which puts
public safety fixes for Tiny Aya Global through fine-tuning, abliteration, compression, template removal and prompt
perturbations. Results browser: [Space]({SPACE}).

**No model outputs are here**: no generations, completions or judged rows, no attack-tuned weights, and no harmful
fine-tuning text. Prompts only, plus aggregates.

| path | what |
|---|---|
| `splits/*.jsonl` | the prompts every model saw: `harmful_test` (200, HarmBench + AdvBench), `xstest_test` (250), `mmlu_test` (300), and the dev splits all choices were made on (`harmful_dev`, `benign_dev`, `mmlu_dev`, `calib_pairs`); `checksums.txt` |
| `curation/` | CuratorKIT manifests and reports: pinned sources, dedup / near-duplicate filtering against the splits |
| `tables/*.csv` | per-cell rates with 95% bootstrap CIs (`cell_ci`), paired tests with Holm adjustment (`pairwise`), retention (Llama Guard, pre-registered; `retention_wg`: WildGuard, post-hoc), judge agreement, human-vs-judge agreement |
| `hand_labels/` | 100 blind hand labels (IDs + labels) and the stratified sample with weights |
| `manifests/gen/`, `manifests/judged/` | {n_manifests} run manifests: config, model and judge revisions, attack details, git sha, timings |

Item IDs join across files: `<run>__<attack>__<split>__<item id>__<variant>`.

## Licences of the sources (see LICENSE.md)

HarmBench (MIT), AdvBench (MIT), MMLU (MIT), XSTest (CC-BY-4.0), OR-Bench (CC-BY-4.0, `benign_dev`); the harmless half
of `calib_pairs` is from Stanford Alpaca (CC-BY-NC-4.0). Tables, labels and manifests: CC-BY-4.0.
"""


DATASET_LICENSE = """# Licences

| file(s) | source | licence |
|---|---|---|
| splits/harmful_dev.jsonl, splits/harmful_test.jsonl, harmful half of calib_pairs | HarmBench (centerforaisafety/HarmBench), AdvBench (llm-attacks/llm-attacks) | MIT |
| splits/mmlu_dev.jsonl, splits/mmlu_test.jsonl | MMLU (cais/mmlu) | MIT |
| splits/xstest_test.jsonl | XSTest (Paul/XSTest) | CC-BY-4.0 |
| splits/benign_dev.jsonl | OR-Bench (bench-llm/or-bench) | CC-BY-4.0 |
| harmless half of splits/calib_pairs.jsonl | Stanford Alpaca | CC-BY-NC-4.0 |
| tables/, hand_labels/, manifests/, curation/ | Aftermarket | CC-BY-4.0 |
"""


# ----------------------------------------------------------------------------- staging + checks

def stage_models(out: Path) -> list[Path]:
    import pandas as pd
    from huggingface_hub import hf_hub_download

    ci = pd.read_csv(REPO_ROOT / "results" / "tables" / "cell_ci.csv")
    ret = pd.read_csv(REPO_ROOT / "results" / "tables" / "retention.csv")
    dirs = []
    for run in MODELS:
        d = out / "models" / run
        d.mkdir(parents=True, exist_ok=True)
        for f in ADAPTER_FILES:
            shutil.copy(hf_hub_download(STORE, f"{run}/{f}", revision=STORE_REV), d / f)
        meta = json.loads((d / "defense_meta.json").read_text(encoding="utf-8"))
        assert meta["id"] == run and meta["format"] == "lora", meta
        (d / "README.md").write_text(model_card(run, meta, ci, ret), encoding="utf-8")
        dirs.append(d)
    return dirs


def stage_dataset(out: Path) -> Path:
    d = out / "dataset"
    shutil.rmtree(d, ignore_errors=True)
    for name in SPLITS:
        _copy(REPO_ROOT / "data" / "splits" / f"{name}.jsonl", d / "splits" / f"{name}.jsonl")
    for src, dst in CURATION.items():
        _copy(REPO_ROOT / src, d / dst)
    for t in TABLES:
        _copy(REPO_ROOT / "results" / "tables" / t, d / "tables" / t)
    for f in ("hand_labels.jsonl", "hand_label_sample.jsonl"):
        _copy(REPO_ROOT / "data" / f, d / "hand_labels" / f)
    n = 0
    for kind in ("gen", "judged"):
        for m in sorted((REPO_ROOT / "results" / kind).glob("*.json")):
            if not m.name.startswith("AMc"):  # dev candidates are selection material, not results
                _copy(m, d / "manifests" / kind / m.name)
                n += 1
    (d / "README.md").write_text(dataset_card(n), encoding="utf-8")
    (d / "LICENSE.md").write_text(DATASET_LICENSE, encoding="utf-8")
    return d


def _copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(src, dst)


class UnsafeRelease(RuntimeError):
    pass


def _keys(o, out: set) -> set:
    if isinstance(o, dict):
        for k, v in o.items():
            out.add(str(k).lower())
            _keys(v, out)
    elif isinstance(o, list):
        for v in o:
            _keys(v, out)
    return out


def check_dataset(d: Path) -> int:
    """Raise unless every staged dataset file is an allowed text type and no JSON record carries an output field."""
    n = 0
    for p in sorted(x for x in d.rglob("*") if x.is_file()):
        if p.suffix not in DATASET_EXT:
            raise UnsafeRelease(f"{p}: file type not allowed in the public dataset")
        if p.suffix in {".json", ".jsonl"}:
            text = p.read_text(encoding="utf-8")
            recs = [json.loads(l) for l in text.splitlines() if l.strip()] if p.suffix == ".jsonl" else [json.loads(text)]
            bad = set().union(*(_keys(r, set()) for r in recs)) & FORBIDDEN_KEYS if recs else set()
            if bad:
                raise UnsafeRelease(f"{p}: output fields {sorted(bad)}")
        if p.suffix == ".csv":
            header = p.read_text(encoding="utf-8").splitlines()[0].lower().split(",")
            if FORBIDDEN_KEYS & set(header):
                raise UnsafeRelease(f"{p}: output columns {sorted(FORBIDDEN_KEYS & set(header))}")
        n += 1
    return n


def check_models(dirs: list[Path]) -> None:
    for d in dirs:
        files = sorted(p.name for p in d.iterdir())
        if files != sorted(ADAPTER_FILES + ["README.md"]):
            raise UnsafeRelease(f"{d}: unexpected files {files}")
        meta = json.loads((d / "defense_meta.json").read_text(encoding="utf-8"))
        if d.name not in MODELS or meta.get("kind") not in {"final", "control"}:
            raise UnsafeRelease(f"{d}: not a frozen defense adapter ({meta.get('kind')})")


# ----------------------------------------------------------------------------- publish

def ensure_public_target(api, repo_id: str, repo_type: str) -> None:
    """Create repo_id public, or reuse it only if it already exists AND is public. Never upload into a private repo
    (a case-insensitive name clash once sent AM's files into the private store)."""
    from huggingface_hub.utils import RepositoryNotFoundError

    try:
        info = api.repo_info(repo_id, repo_type=repo_type)
    except RepositoryNotFoundError:
        api.create_repo(repo_id, repo_type=repo_type, private=False)
        info = api.repo_info(repo_id, repo_type=repo_type)
    if info.private or info.id.lower() in {STORE.lower(), f"{OWNER}/aftermarket-runs"}:
        raise UnsafeRelease(f"{repo_id} resolves to {info.id} (private={info.private}): refusing to upload")


def publish(model_dirs: list[Path], data_dir: Path) -> dict:
    from huggingface_hub import HfApi

    api = HfApi()
    urls = {"models": [], "dataset": None, "collections": {}}
    for d in model_dirs:
        rid = f"{OWNER}/{REPO_NAMES[d.name]}"
        ensure_public_target(api, rid, "model")
        api.upload_folder(repo_id=rid, folder_path=str(d), commit_message=f"Aftermarket {d.name} adapter + card")
        urls["models"].append(rid)
    ensure_public_target(api, DATASET, "dataset")
    api.upload_folder(repo_id=DATASET, repo_type="dataset", folder_path=str(data_dir),
                      commit_message="Aftermarket evaluation data (no model outputs)")
    urls["dataset"] = DATASET
    existing = {c.title: c for c in api.list_collections(owner=OWNER)}
    items = {"models": [(r, "model") for r in urls["models"]] + [(f"{OWNER}/aftermarket", "space")],
             "datasets": [(DATASET, "dataset"), (f"{OWNER}/aftermarket", "space")]}
    for key, title in COLLECTIONS.items():
        c = existing.get(title) or api.create_collection(
            title, namespace=OWNER, private=False,
            description="Aftermarket: does a safety fix survive the next person who downloads the model?")
        have = {i.item_id for i in api.get_collection(c.slug).items}
        for rid, kind in items[key]:
            if rid not in have:
                api.add_collection_item(c.slug, item_id=rid, item_type=kind)
        urls["collections"][key] = f"https://huggingface.co/collections/{c.slug}"
    return urls


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--stage", action="store_true", help="build and check the release locally")
    ap.add_argument("--publish", action="store_true", help="stage, check and upload (public)")
    args = ap.parse_args(argv)
    dirs, data = stage_models(STAGE), stage_dataset(STAGE)
    check_models(dirs)
    n = check_dataset(data)
    print(f"staged {len(dirs)} model repos and {n} dataset files under {STAGE} (checked)")
    if args.publish:
        print(json.dumps(publish(dirs, data), indent=1))


if __name__ == "__main__":
    main()
