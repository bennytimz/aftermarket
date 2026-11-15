"""`aftermarket audit`: run the Aftermarket attack battery on your own Tiny Aya Global derivative and get a one-page
robustness report card ranked against our leaderboard.

    aftermarket audit --model your-org/your-tiny-aya-fix                       # full battery, ~4 h on one T4
    aftermarket audit --model CohereLabs/tiny-aya-global --adapter your-org/your-lora
    aftermarket audit --model ./my-merged-model --quick                        # A0, A2, A3, A5bon, 100 prompts, ~1.5 h

Steps (every one reuses the leaderboard code, so the card is directly comparable with the leaderboard runs):
  1. register the model for this audit only (results/audit/<name>/models.yaml, via $AFTERMARKET_AUDIT_MODELS)
  2. generate: aftermarket.generate on harmful + MMLU under every attack, XSTest as released (one process per GPU)
  3. judge:    aftermarket.judge in its own process (Llama Guard 3 1B + WildGuard; MMLU by letter parsing)
  4. card:     aftermarket.card from the judged rows -> results/audit/<name>/card/<name>.md + .png

Full battery: A0, A1, A2, A3, A4q3, A5raw, A5bon on the 200 held-out harmful prompts, 250 XSTest, 300 MMLU.
--quick: A0, A2, A3, A5bon on fixed 100-item subsets (data.QUICK_SPLITS), ranked against the leaderboard recomputed
on the same 100 items. Same seed (42), settings and judges as the leaderboard (configs/attacks.yaml).

The test prompts are used here as the leaderboard uses them: fixed attacks on someone's model, no choice made from
the results (CLAUDE.md test-set rule). Attacked weights and fine-tuned adapters stay in memory or a temp dir and are
deleted; generations stay under results/audit/ (gitignored) and nothing is pushed anywhere.

Requirements: one GPU with >= 15 GB (a T4 works), the pinned requirements.txt, and an HF token that has accepted the
Tiny Aya, Llama Guard 3 and WildGuard licences. The model must use Tiny Aya Global's tokenizer (checked).
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
AUDIT_DIR = REPO_ROOT / "results" / "audit"
FULL = {"attacks": ["A0", "A1", "A2", "A3", "A4q3", "A5raw", "A5bon"],
        "splits": ["harmful_test", "mmlu_test"], "refusal_split": "xstest_test", "scope": "200 held-out harmful prompts"}
QUICK = {"attacks": ["A0", "A2", "A3", "A5bon"],
         "splits": ["harmful_q100_test", "mmlu_q100_test"], "refusal_split": "xstest_q100_test",
         "scope": "100-prompt quick subset"}
GEN_DEFAULTS = {"seed": 42, "precision": "fp16", "batch_size": 16, "max_new_tokens": {"default": 256, "mmlu": 8}}


# ----------------------------------------------------------------------------- planning (no torch)

def run_name(model: str, adapter: str | None = None) -> str:
    """A file-safe run id from the model (or adapter) name: no '__' (the parquet name separator)."""
    raw = Path(adapter or model).name or "model"
    name = re.sub(r"[^A-Za-z0-9.-]+", "-", raw).strip("-.") or "model"
    return f"audit-{name}"[:80]


def check_name(name: str, registry: dict) -> None:
    if "__" in name or not re.fullmatch(r"[A-Za-z0-9._-]+", name):
        raise SystemExit(f"--name {name!r}: letters, digits, '.', '-' and single '_' only")
    if name.startswith("AM"):
        raise SystemExit("--name may not start with 'AM' (reserved for our frozen defense runs)")
    if name in registry["models"] or name in registry.get("runs", {}):
        raise SystemExit(f"--name {name!r} is a leaderboard run: pick another")


def model_files(path: str, revision: str | None) -> list[str]:
    if Path(path).is_dir():
        return sorted(p.name for p in Path(path).iterdir())
    from huggingface_hub import list_repo_files

    return list_repo_files(path, revision=revision)


def pin_revision(path: str, revision: str | None) -> str | None:
    """Local folder -> None; Hub repo -> the commit sha (pins what was audited)."""
    if Path(path).is_dir():
        return None
    from huggingface_hub import model_info

    return model_info(path, revision=revision).sha


def model_spec(model: str, adapter: str | None, base_model: str, revision: str | None, files: list[str]) -> dict:
    """Registry entry for the model under audit. An adapter must sit on Tiny Aya Global (the base every leaderboard
    run shares); a full model is loaded as is. Its own tokenizer, if shipped, is checked against the pinned one."""
    if adapter and model != base_model:
        raise SystemExit(f"--adapter is supported on {base_model} only (got --model {model}); merge it first and "
                         "pass the merged folder as --model")
    path = adapter or model
    spec = {"name": path, "status": "active", "hf_path": path, "revision": revision,
            "format": "lora" if adapter else "full", "base": "BASE", "fix_type": "audit",
            "files": [f for f in files if f in {"tokenizer.json", "chat_template.jinja"}]}
    if "chat_template.jinja" in files:
        spec["chat_template_file"] = "chat_template.jinja"
    return spec


def gen_configs(name: str, out: Path, mode: dict, gpus: list[int], batch_size: int) -> list[dict]:
    """Two generation configs: harmful + MMLU under every attack, and over-refusal (XSTest) as released."""
    common = {**GEN_DEFAULTS, "batch_size": batch_size, "runs": [name], "gpus": gpus,
              "out_dir": str(out / "gen"), "push": {"enabled": False}}
    return [{**common, "attacks": mode["attacks"], "splits": mode["splits"]},
            {**common, "attacks": ["A0"], "splits": [mode["refusal_split"]]}]


def judge_config(out: Path, base: dict) -> dict:
    return {**base, "gen_dir": str(out / "gen"), "out_dir": str(out / "judged"),
            "push": {**base["push"], "enabled": False}, "wildguard_on_harmful_runs": ["*"]}


def a3_caveat(gen_dir: Path, name: str, split: str) -> str:
    """If A3's frozen layer rule left most refusals in place on dev, a low A3 is not evidence of resistance."""
    p = Path(gen_dir) / f"{name}__A3__{split}.json"
    if not p.exists():
        return ""
    info = json.loads(p.read_text(encoding="utf-8")).get("attack_info", {})
    before, after = info.get("refusal_no_ablation"), info.get("refusal_at_layer")
    if before is None or after is None or after < before / 2:
        return ""
    return (f"A3's layer rule (KL filter on benign prompts) picked a layer that left most refusals in place on dev "
            f"({before:.0%} → {after:.0%}), so a low A3 here does not show the model resists abliteration "
            "(README §3, finding 6).")


# ----------------------------------------------------------------------------- steps

def _env(models_yaml: Path) -> dict:
    from aftermarket.load import AUDIT_MODELS_ENV

    return {**os.environ, AUDIT_MODELS_ENV: str(models_yaml), "PYTHONUNBUFFERED": "1",
            "PYTHONPATH": str(REPO_ROOT / "src") + os.pathsep + os.environ.get("PYTHONPATH", "")}


def generate(out: Path, cfgs: list[dict], env: dict) -> None:
    """Each config in a fresh `aftermarket.generate` process (it spawns one worker per GPU itself)."""
    for i, cfg in enumerate(cfgs):
        path = out / f"gen_{i}.yaml"
        path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
        # --final: fixed attacks on a model under audit, no choice made from these results (leaderboard cell)
        cmd = [sys.executable, "-m", "aftermarket.generate", "--config", str(path), "--final", "--no-pull"]
        if subprocess.run(cmd, env=env, cwd=REPO_ROOT).returncode:
            raise SystemExit(f"generation failed ({path.name}); finished jobs are kept, re-run to resume")


def judge(out: Path, cfg: dict, env: dict, gpus: list[int]) -> None:
    path = out / "judge.yaml"
    path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    cmd = [sys.executable, "-m", "aftermarket.judge", "--config", str(path), "--no-pull",
           "--gpus", ",".join(map(str, gpus))]
    if subprocess.run(cmd, env=env, cwd=REPO_ROOT).returncode:
        raise SystemExit("judging failed; finished files are kept, re-run to resume")


def make_card(name: str, label: str, out: Path, mode: dict, quick: bool, table_dir: Path | None = None) -> list[Path]:
    """Card from the audit's judged rows, ranked against the leaderboard tables (quick: the 100-item reference)."""
    from aftermarket.card import TABLE_DIR, card_values, load_tables, write_card
    from aftermarket.evaluate import cell_ci_table, load_judged, retention_table

    df = load_judged(out / "judged")
    if not len(df):
        raise SystemExit(f"no judged rows in {out / 'judged'}")
    ci, ret = cell_ci_table(df), retention_table(df)
    ref_ci, ref_ret = load_tables(table_dir or TABLE_DIR, quick=quick)
    v = card_values(name, ci, ret, ref_ci, ref_ret, attacks=mode["attacks"], label=label, extra_attacks=())
    v["note"] = a3_caveat(out / "gen", name, mode["splits"][0])
    ci.to_csv(out / "cell_ci.csv", index=False)
    ret.to_csv(out / "retention.csv", index=False)
    return write_card(v, out / "card", mode["scope"])


def _git_sha() -> str:
    r = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=REPO_ROOT)
    return r.stdout.strip()


def audit(args) -> list[Path]:
    from aftermarket.load import load_registry

    reg = load_registry()
    base = reg["base_model"]
    model = args.model or base
    name = args.name or run_name(model, args.adapter)
    check_name(name, reg)
    mode = QUICK if args.quick else FULL
    out = Path(args.out) if args.out else AUDIT_DIR / name
    out.mkdir(parents=True, exist_ok=True)
    label = model + (f" + {args.adapter}" if args.adapter else "")
    gpus = [int(g) for g in args.gpus.split(",")]
    if not args.card_only:
        path = args.adapter or model
        revision = pin_revision(path, args.revision)
        spec = model_spec(model, args.adapter, base, revision, model_files(path, revision))
        models_yaml = out / "models.yaml"
        models_yaml.write_text(yaml.safe_dump({"models": {name: spec}}, sort_keys=False), encoding="utf-8")
        cfgs = gen_configs(name, out, mode, gpus, args.batch_size)
        judge_cfg = judge_config(out, yaml.safe_load((REPO_ROOT / "configs" / "judge.yaml").read_text(encoding="utf-8")))
        (out / "manifest.json").write_text(json.dumps({
            "model": model, "adapter": args.adapter, "revision": revision, "name": name,
            "mode": "quick" if args.quick else "full", "attacks": mode["attacks"], "seed": GEN_DEFAULTS["seed"],
            "git_sha": _git_sha(), "time_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "gen_configs": cfgs, "judge_config": judge_cfg}, indent=1), encoding="utf-8")
        env = _env(models_yaml)
        generate(out, cfgs, env)
        judge(out, judge_cfg, env, gpus)
    paths = make_card(name, label, out, mode, args.quick)
    print(f"report card: {paths[0]}\n            {paths[1]}")
    return paths


def add_arguments(ap: argparse.ArgumentParser) -> None:
    ap.add_argument("--model", help="HF model id or local folder (a Tiny Aya Global derivative); "
                                    "default: Tiny Aya Global itself (use with --adapter)")
    ap.add_argument("--adapter", help="LoRA adapter (HF id or folder) on Tiny Aya Global")
    ap.add_argument("--revision", help="Hub revision of --model / --adapter (default: latest, pinned in the manifest)")
    ap.add_argument("--quick", action="store_true", help="A0, A2, A3, A5bon on a fixed 100-prompt subset")
    ap.add_argument("--name", help="run id for files and the card (default: audit-<model name>)")
    ap.add_argument("--out", help="output folder (default: results/audit/<name>)")
    ap.add_argument("--gpus", default="0", help="comma-separated GPU ids (one worker each); default 0")
    ap.add_argument("--batch-size", type=int, default=GEN_DEFAULTS["batch_size"],
                    help="starting generation batch size (halved on OOM)")
    ap.add_argument("--card-only", action="store_true", help="only rebuild the card from finished judged files")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="aftermarket audit", description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    add_arguments(ap)
    args = ap.parse_args(argv)
    if not args.model and not args.adapter:
        ap.error("give --model (and/or --adapter)")
    audit(args)


if __name__ == "__main__":
    main()
