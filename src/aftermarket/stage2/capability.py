"""E1 capability benchmarks with lm-evaluation-harness (vLLM backend) on one saved (run, attack) model.

Tasks (STAGE2_PLAN.md §3 E1; subsets frozen in configs/stage2/subsets.yaml):
  mmlu300        our 300 Stage 1 mmlu_test items as a custom lm-eval task (lm-eval's MMLU prompt, 0-shot, acc)
  arc_challenge  500 of 1,172 (acc_norm)      hellaswag   500 of 10,042 (acc_norm)
  winogrande     500 of 1,267 (acc)           ifeval      all 541 (prompt/inst-level strict acc)
  gsm8k          250 of 1,319, 5-shot as multi-turn chat (exact_match, strict-match + flexible-extract)
Every task runs with --apply_chat_template (the run's own template, saved with the model), seed 42, greedy.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from aftermarket.stage2.subsets import REPO_ROOT, load_subsets, read_jsonl

TASKS = ["mmlu300", "arc_challenge", "hellaswag", "winogrande", "ifeval", "gsm8k"]

MMLU300_YAML = """task: mmlu300
dataset_path: json
dataset_kwargs:
  data_files:
    test: {data}
test_split: test
output_type: multiple_choice
doc_to_text: "The following are multiple choice questions (with answers) about {{{{subject.replace('_', ' ')}}}}.\\n\\n{{{{question.strip()}}}}\\nA. {{{{choices[0]}}}}\\nB. {{{{choices[1]}}}}\\nC. {{{{choices[2]}}}}\\nD. {{{{choices[3]}}}}\\nAnswer:"
doc_to_choice: ["A", "B", "C", "D"]
doc_to_target: answer
metric_list:
  - metric: acc
    aggregation: mean
    higher_is_better: true
metadata:
  version: 1.0
"""


def write_mmlu300_task(task_dir: Path) -> Path:
    """Custom task over data/splits/mmlu_test.jsonl (answer letter -> index), in Stage 1 item order."""
    task_dir.mkdir(parents=True, exist_ok=True)
    rows = read_jsonl(REPO_ROOT / "data" / "splits" / "mmlu_test.jsonl")
    data = task_dir / "mmlu300.jsonl"
    data.write_text("".join(json.dumps({"id": r["id"], "subject": r["category"], "question": r["question"],
                                        "choices": r["choices"], "answer": "ABCD".index(r["answer"])}) + "\n"
                            for r in rows), encoding="utf-8")
    (task_dir / "mmlu300.yaml").write_text(MMLU300_YAML.format(data=data.as_posix()), encoding="utf-8")
    return task_dir


def samples_arg(tasks: list[str]) -> dict[str, list[int]]:
    cap = load_subsets()["capability"]
    return {t: cap[t]["indices"] for t in tasks if "indices" in cap.get(t, {})}


def lm_eval_command(model_dir: str, out: Path, task_dir: Path, tasks: list[str] = TASKS, seed: int = 42,
                    limit_override: dict | None = None) -> list[str]:
    from aftermarket.stage2.vgen import backend, gpu_mem_util, vllm_python

    samples = limit_override or samples_arg(tasks)
    if backend() == "vllm":
        model, bs = "vllm", "auto"
        model_args = (f"pretrained={model_dir},tokenizer={model_dir},dtype=float16,"
                      f"gpu_memory_utilization={gpu_mem_util()},max_model_len=4096,seed={seed}")
    else:  # fallback: HF transformers backend, fp16, fixed batch
        model, bs = "hf", "16"
        model_args = f"pretrained={model_dir},tokenizer={model_dir},dtype=float16,max_length=4096"
    return [vllm_python(), "-m", "lm_eval", "run", "--model", model, "--model_args", model_args,
            "--tasks", ",".join(tasks), "--include_path", str(task_dir), "--apply_chat_template",
            "--fewshot_as_multiturn", "--batch_size", bs, "--seed", str(seed), "--log_samples",
            "--output_path", str(out), "--samples", json.dumps(samples)]


def run(model_dir: str, out: Path, tmp: Path, tasks: list[str] = TASKS, limit_override: dict | None = None) -> int:
    task_dir = write_mmlu300_task(tmp / "lm_eval_tasks")
    if limit_override:  # smoke: tiny subsets for every task incl. the ones evaluated in full
        tasks = [t for t in tasks if t in limit_override]
    cmd = lm_eval_command(model_dir, out, task_dir, tasks, limit_override=limit_override)
    print("[capability]", " ".join(cmd[:4]), "...", flush=True)
    return subprocess.run(cmd).returncode


def collect(out: Path) -> dict:
    """Aggregate scores from lm-eval's results_*.json under `out` (latest file)."""
    files = sorted(out.rglob("results_*.json"))
    if not files:
        return {}
    res = json.loads(files[-1].read_text(encoding="utf-8"))["results"]
    keep = {"acc,none", "acc_norm,none", "prompt_level_strict_acc,none", "inst_level_strict_acc,none",
            "exact_match,strict-match", "exact_match,flexible-extract"}
    return {t: {k: v for k, v in m.items() if k in keep} for t, m in res.items()}


# Primary metric per task (STAGE2_PLAN.md §3 E1); gsm8k uses the strict-match filter.
PRIMARY = {"mmlu300": ("acc", None), "arc_challenge": ("acc_norm", None), "hellaswag": ("acc_norm", None),
           "winogrande": ("acc", None), "ifeval": ("prompt_level_strict_acc", None),
           "gsm8k": ("exact_match", "strict-match")}


def item_scores(cell_dir: Path) -> "pd.DataFrame":
    """Per-item primary scores from lm-eval's samples_<task>_*.jsonl under `cell_dir` (latest file per task).

    Returns columns task, doc_id, y (0/1 float). Benign benchmark items only, so the output is safe to commit."""
    import pandas as pd

    recs = []
    for task, (metric, filt) in PRIMARY.items():
        files = sorted(Path(cell_dir).rglob(f"samples_{task}_*.jsonl"))
        if not files:
            continue
        for line in files[-1].read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            s = json.loads(line)
            if filt is not None and s.get("filter") != filt:
                continue
            recs.append({"task": task, "doc_id": int(s["doc_id"]), "y": float(s[metric])})
    df = pd.DataFrame(recs, columns=["task", "doc_id", "y"])
    if df.duplicated(["task", "doc_id"]).any():
        raise ValueError(f"duplicate (task, doc_id) in {cell_dir}")
    return df
