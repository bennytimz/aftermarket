"""Stage 2 smoke job (Batch 1, STAGE2_PLAN.md §6). On one L4:

  1. NLLB-1.3B translation of the 150 E6 prompts into es, zh, yo, sw, fr (+ back-translations)
  2. BASE A0 cell: weights -> lm-eval on tiny subsets of all 6 tasks -> vLLM generation of every A0 set (full size,
     so the timing is real); Yoruba check file (prompt half + 30 back-translated XSTest replies)
  3. A4q3 fake-quant equivalence on BASE: agreement of HF-HQQ vs vLLM-fake-quant, next to HF-fp16 vs vLLM-fp16
     (the backend's own noise), on mmlu_test (300 answer letters) and 32 harmful_dev greedy replies
Pushes translations, the check file and all aggregates to the private store under stage2/smoke/.

    PYTHONPATH=src python -m aftermarket.stage2.smoke_job --final
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from aftermarket.stage2.subsets import REPO_ROOT

OUT = REPO_ROOT / "results" / "stage2" / "smoke"
N_HARM_DEV = 32


def letter(text: str) -> str | None:
    for ch in text.strip()[:3]:
        if ch in "ABCD":
            return ch
    return None


def hf_generate(model_kind: str, rows: dict[str, list[dict]]) -> dict[str, list[str]]:
    """Stage 1 path: HF generate (fp16 or real HQQ 3-bit), greedy, batch 16."""
    from aftermarket.attacks.compress import load_hqq
    from aftermarket.generate import generate_batched
    from aftermarket.load import build_prompt_ids, free_memory, load_model

    if model_kind == "hqq3":
        model, tok, _ = load_hqq("BASE", "native", 3, 64)
    else:
        model, tok = load_model("BASE", precision="fp16")
    out = {}
    for name, rs in rows.items():
        ids = [build_prompt_ids(model, tok, [{"role": "user", "content": r["prompt"]}]) for r in rs]
        res, _ = generate_batched(model, tok, ids, 8 if name == "mmlu" else 256, 16)
        out[name] = [r["response"] for r in res]
    del model
    free_memory()
    return out


def vllm_generate(model_dir: str, rows: dict[str, list[dict]]) -> dict[str, list[str]]:
    from transformers import AutoTokenizer
    from vllm import SamplingParams, TokensPrompt

    from aftermarket.stage2.vgen import make_llm, prompt_ids, stop_ids

    tok = AutoTokenizer.from_pretrained(model_dir)
    llm = make_llm(model_dir)
    out = {}
    for name, rs in rows.items():
        sp = SamplingParams(temperature=0.0, max_tokens=8 if name == "mmlu" else 256, stop_token_ids=stop_ids(tok),
                            skip_special_tokens=True)
        res = llm.generate([TokensPrompt(prompt_token_ids=prompt_ids(tok, r["prompt"])) for r in rs], sp, use_tqdm=False)
        out[name] = [o.outputs[0].text.strip() for o in res]
    return out


def agreement(a: dict[str, list[str]], b: dict[str, list[str]]) -> dict:
    mm = sum(letter(x) == letter(y) for x, y in zip(a["mmlu"], b["mmlu"])) / len(a["mmlu"])
    ex = sum(x == y for x, y in zip(a["harm"], b["harm"])) / len(a["harm"])
    pre = sum(x[:200] == y[:200] for x, y in zip(a["harm"], b["harm"])) / len(a["harm"])
    return {"mmlu_letter_agreement": round(mm, 4), "harm_exact_text": round(ex, 4), "harm_first200_chars": round(pre, 4)}


def mmlu_acc(answers: list[str], rows: list[dict]) -> float:
    return round(sum(letter(x) == r["answer"] for x, r in zip(answers, rows)) / len(rows), 4)


def equivalence_part(which: str, model_dir: str | None, out: Path) -> None:
    """Runs in its own process: 'hf_fp16' | 'hf_hqq3' | 'vllm' (model_dir) -> out JSON."""
    from aftermarket.data import load_split

    rows = {"mmlu": load_split("mmlu_test"), "harm": load_split("harmful_dev")[:N_HARM_DEV]}
    t = time.time()
    res = vllm_generate(model_dir, rows) if which == "vllm" else hf_generate(which[3:], rows)
    out.write_text(json.dumps({"which": which, "seconds": round(time.time() - t, 1), **res}), encoding="utf-8")


def sh(cmd: list[str]) -> int:
    print("$", " ".join(cmd)[:300], flush=True)
    return subprocess.run(cmd, env={**os.environ, "PYTHONPATH": str(REPO_ROOT / "src")}).returncode


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--final", action="store_true")
    ap.add_argument("--part", help=argparse.SUPPRESS)
    ap.add_argument("--model-dir", help=argparse.SUPPRESS)
    ap.add_argument("--part-out", type=Path, help=argparse.SUPPRESS)
    ap.add_argument("--skip", default="", help="comma list of steps to skip: translate,cell,equiv")
    args = ap.parse_args(argv)
    if args.part:
        return equivalence_part(args.part, args.model_dir, args.part_out)
    if not args.final:
        raise SystemExit("smoke uses test-derived items: pass --final")
    skip = set(filter(None, args.skip.split(",")))
    OUT.mkdir(parents=True, exist_ok=True)
    py = [sys.executable, "-m"]
    report: dict = {"time_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "steps": {}}
    tmp = Path(os.environ.get("AFTERMARKET_TMP", "/tmp")) / "s2"

    if "translate" not in skip:
        t = time.time()
        rc = sh(py + ["aftermarket.stage2.translate", "prompts"])
        report["steps"]["translate"] = {"rc": rc, "seconds": round(time.time() - t, 1)}

    if "cell" not in skip:
        t = time.time()
        rc = sh(py + ["aftermarket.stage2.cell", "--run", "BASE", "--attack", "A0", "--final", "--smoke"])
        report["steps"]["cell_BASE_A0"] = {"rc": rc, "seconds": round(time.time() - t, 1)}
        yo = REPO_ROOT / "results" / "stage2" / "cells" / "BASE__A0__smoke" / "gen" / "ml_yo.parquet"
        if yo.exists():
            import pandas as pd

            from aftermarket.stage2.translate import Translator, yo_check

            df = pd.read_parquet(yo)
            df["back_en"] = Translator()(df["response"].tolist(), "yo", "en")
            df[df["kind"] == "xstest"].to_parquet(OUT / "base_a0_yo_xstest_backtranslated.parquet", index=False)
            report["yo_check"] = str(yo_check(OUT / "base_a0_yo_xstest_backtranslated.parquet"))

    if "equiv" not in skip:
        from aftermarket.stage2.weights import main as _  # noqa: F401  (import check)

        t = time.time()
        parts = {}
        for which in ("hf_fp16", "hf_hqq3"):
            p = tmp / f"equiv_{which}.json"
            parts[which] = (sh(py + ["aftermarket.stage2.smoke_job", "--part", which, "--part-out", str(p)]), p)
        for attack in ("A0", "A4q3"):
            wdir = tmp / "weights" / f"equiv_BASE__{attack}"
            rc = sh(py + ["aftermarket.stage2.weights", "--run", "BASE", "--attack", attack, "--out", str(wdir),
                          "--info", str(OUT / f"equiv_weights_{attack}.json")])
            p = tmp / f"equiv_vllm_{attack}.json"
            if rc == 0:
                rc = sh(py + ["aftermarket.stage2.smoke_job", "--part", "vllm", "--model-dir", str(wdir),
                              "--part-out", str(p)])
            subprocess.run(["rm", "-rf", str(wdir)])
            parts[f"vllm_{attack}"] = (rc, p)
        ok = all(rc == 0 and p.exists() for rc, p in parts.values())
        eq = {"rcs": {k: rc for k, (rc, _) in parts.items()}, "seconds": round(time.time() - t, 1)}
        if ok:
            from aftermarket.data import load_split

            d = {k: json.loads(p.read_text(encoding="utf-8")) for k, (_, p) in parts.items()}
            mm = load_split("mmlu_test")
            eq.update({
                "backend_noise_fp16 (HF fp16 vs vLLM fp16)": agreement(d["hf_fp16"], d["vllm_A0"]),
                "fake_quant (HF HQQ-3bit vs vLLM fake-quant)": agreement(d["hf_hqq3"], d["vllm_A4q3"]),
                "cross (HF fp16 vs HF HQQ-3bit, the attack's own effect)": agreement(d["hf_fp16"], d["hf_hqq3"]),
                "mmlu_acc": {k: mmlu_acc(v["mmlu"], mm) for k, v in d.items()},
                "part_seconds": {k: v["seconds"] for k, v in d.items()},
            })
        report["steps"]["equivalence"] = eq

    (OUT / "smoke_report.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(json.dumps(report, indent=1), flush=True)
    # push: translations (prompts only), Yoruba check (prompts + XSTest replies), aggregates. Never weights.
    from huggingface_hub import CommitOperationAdd, HfApi

    from aftermarket.generate import ensure_private_repo

    repo = ensure_private_repo("aftermarket-runs")
    files = [p for p in OUT.glob("*") if p.suffix in {".json", ".parquet"}]
    files += [f for f in (REPO_ROOT / "data" / "stage2" / "translations.jsonl", REPO_ROOT / "data" / "translation_check_yo.csv")
              if f.exists()]
    HfApi().create_commit(repo, [CommitOperationAdd(path_in_repo=f"stage2/smoke/{f.name}", path_or_fileobj=str(f))
                                 for f in files], repo_type="dataset", commit_message="stage2 smoke")
    print("pushed", [f.name for f in files], "to", repo, flush=True)


if __name__ == "__main__":
    main()
