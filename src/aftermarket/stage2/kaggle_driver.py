"""Kaggle 2xT4 driver for Stage 2 (used while HF org Jobs are blocked). One worker per GPU.

    PYTHONPATH=src python -m aftermarket.stage2.kaggle_driver --plan session1 --final --max-hours 11.3

session1 = smoke (translation, BASE A0 cell on tiny lm-eval subsets, fake-quant equivalence), then, ONLY if the
           smoke passes, launch 2 (12 runs x A0, A2, A3, A4q3) until the session clock runs out.
session2 = launch 2 again: resumes, skipping every cell already finished on the private Hub store.
auto     = session2 if an earlier session1 passed its smoke (store: stage2/kaggle/), else session1.

Decision rules, fixed here before any run:
  * vLLM: if the smoke cell fails on vLLM, it is retried once with the HF backend (AFTERMARKET_S2_BACKEND=hf);
    if that passes, every later cell uses HF (recorded in each manifest). If both fail, stop.
  * A4q3 fake quantization is used only if its MMLU answer agreement with real HQQ is >= min(0.95, the
    fp16 backend-noise agreement - 0.02); otherwise A4q3 cells are skipped and reported.
  * A new cell starts only if its estimated T4 time fits in what is left of --max-hours.
"""

from __future__ import annotations

import argparse
import json
import os
import queue
import subprocess
import sys
import threading
import time
from pathlib import Path

from aftermarket.stage2.subsets import REPO_ROOT

RUNS = ["BASE", "SDT", "GC", "IL", "RL", "RL_noguard", "ST", "SP", "AM", "AM_rand0", "AM_rand1", "AM_all"]
LAUNCH2 = ["A2", "A0", "A3", "A4q3"]          # longest first, so the two GPUs end together
EST_MIN_T4 = {"A0": 30, "A2": 50, "A3": 38, "A4q3": 35, "A1": 45, "A2s1": 30, "A2s2": 30}
STORE, STORE_PATH = "aftermarket-runs", "stage2/cells"


def env() -> dict:
    return {**os.environ, "PYTHONPATH": str(REPO_ROOT / "src")}


def sh(cmd: list[str], extra: dict | None = None) -> int:
    print("$", " ".join(cmd)[:300], flush=True)
    return subprocess.run(cmd, env={**env(), **(extra or {})}).returncode


def done_on_hub() -> set[str]:
    """Cells whose manifest on the private store has no error."""
    from huggingface_hub import HfApi, hf_hub_download

    from aftermarket.generate import resolve_repo_id

    repo = resolve_repo_id(STORE)
    files = [f for f in HfApi().list_repo_files(repo, repo_type="dataset")
             if f.startswith(STORE_PATH + "/") and f.endswith("/manifest.json")]
    done = set()
    for f in files:
        name = f.split("/")[-2]
        if name.endswith("__smoke"):
            continue
        m = json.loads(Path(hf_hub_download(repo, f, repo_type="dataset")).read_text(encoding="utf-8"))
        if not m.get("error"):
            done.add(name)
    return done


def previous_smoke() -> dict | None:
    """The smoke verdict of an earlier session (stage2/kaggle/kaggle_session1_*.json on the private store)."""
    from huggingface_hub import HfApi, hf_hub_download

    from aftermarket.generate import resolve_repo_id

    repo = resolve_repo_id(STORE)
    try:
        files = sorted(f for f in HfApi().list_repo_files(repo, repo_type="dataset")
                       if f.startswith("stage2/kaggle/kaggle_session1_"))
    except Exception:
        return None
    for f in reversed(files):
        s = json.loads(Path(hf_hub_download(repo, f, repo_type="dataset")).read_text(encoding="utf-8")).get("smoke")
        if s and s.get("ok"):
            return s
    return None


def smoke(final_flag: list[str]) -> dict:
    """Run the smoke; returns {'ok', 'backend', 'a4q3_ok', 'report'}."""
    rep_path = REPO_ROOT / "results" / "stage2" / "smoke" / "smoke_report.json"
    py = [sys.executable, "-m", "aftermarket.stage2.smoke_job"] + final_flag
    sh(py, {"CUDA_VISIBLE_DEVICES": "0"})
    rep = json.loads(rep_path.read_text(encoding="utf-8")) if rep_path.exists() else {"steps": {}}
    be = "vllm"
    if rep["steps"].get("cell_BASE_A0", {}).get("rc", 1) != 0:
        print("smoke cell failed on vLLM: retrying with the HF backend", flush=True)
        be = "hf"
        sh(py + ["--skip", "translate"], {"CUDA_VISIBLE_DEVICES": "0", "AFTERMARKET_S2_BACKEND": "hf"})
        rep = json.loads(rep_path.read_text(encoding="utf-8"))
    ok = rep["steps"].get("cell_BASE_A0", {}).get("rc", 1) == 0 and rep["steps"].get("translate", {"rc": 0})["rc"] == 0
    eq = rep["steps"].get("equivalence", {})
    fq = eq.get("fake_quant (HF HQQ-3bit vs vLLM fake-quant)", {}).get("mmlu_letter_agreement")
    noise = eq.get("backend_noise_fp16 (HF fp16 vs vLLM fp16)", {}).get("mmlu_letter_agreement")
    a4_ok = fq is not None and noise is not None and fq >= min(0.95, noise - 0.02)
    out = {"ok": ok, "backend": be, "a4q3_ok": a4_ok, "fake_quant_agreement": fq, "noise_agreement": noise}
    print("SMOKE:", out, flush=True)
    return out


def run_cells(cells: list[tuple[str, str]], gpus: list[str], deadline: float, final_flag: list[str],
              backend: str) -> dict:
    q: queue.Queue = queue.Queue()
    for c in cells:
        q.put(c)
    results: dict = {}
    lock = threading.Lock()

    def worker(gpu: str) -> None:
        while True:
            try:
                run_id, attack = q.get_nowait()
            except queue.Empty:
                return
            name = f"{run_id}__{attack}"
            left_min = (deadline - time.time()) / 60
            if EST_MIN_T4[attack] > left_min:
                with lock:
                    results[name] = "skipped (session time)"
                continue
            t = time.time()
            rc = sh([sys.executable, "-m", "aftermarket.stage2.cell", "--run", run_id, "--attack", attack] + final_flag,
                    {"CUDA_VISIBLE_DEVICES": gpu, "AFTERMARKET_S2_BACKEND": backend})
            with lock:
                results[name] = f"rc {rc}, {round((time.time() - t) / 60, 1)} min (GPU {gpu})"
                print(f"[driver] {name}: {results[name]}  | {q.qsize()} queued", flush=True)

    ts = [threading.Thread(target=worker, args=(g,)) for g in gpus]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    return results


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--plan", choices=["session1", "session2", "auto"], default="auto")
    ap.add_argument("--final", action="store_true")
    ap.add_argument("--gpus", default="0,1")
    ap.add_argument("--max-hours", type=float, default=11.3)
    ap.add_argument("--session-start", type=float, default=None, help="epoch seconds the Kaggle session started")
    ap.add_argument("--attacks", default=",".join(LAUNCH2))
    ap.add_argument("--runs", default=",".join(RUNS))
    args = ap.parse_args(argv)
    if not args.final:
        raise SystemExit("Stage 2 cells use test-derived items: pass --final")
    ff = ["--final"]
    start = args.session_start or time.time()
    deadline = start + args.max_hours * 3600
    status_path = REPO_ROOT / "results" / "stage2" / f"kaggle_{args.plan}_{int(start)}.json"
    status: dict = {"plan": args.plan, "time_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}

    if args.plan == "auto":
        args.plan = "session2" if (previous_smoke() or {}).get("ok") else "session1"
        print(f"[driver] plan auto -> {args.plan}", flush=True)
        status["plan"] = args.plan
        status_path = REPO_ROOT / "results" / "stage2" / f"kaggle_{args.plan}_{int(start)}.json"
    backend, attacks = os.environ.get("AFTERMARKET_S2_BACKEND", "vllm"), args.attacks.split(",")
    if args.plan == "session1":
        s = smoke(ff)
        status["smoke"] = s
        if not s["ok"]:
            save_status(status, status_path)
            raise SystemExit("smoke failed on both backends: stopping (see the log above)")
        backend = s["backend"]
        if not s["a4q3_ok"]:
            attacks = [a for a in attacks if a != "A4q3"]
            print("A4q3 fake quant did not pass the equivalence rule: A4q3 cells skipped", flush=True)
    else:
        s = previous_smoke()
        if not s or not s.get("ok"):
            raise SystemExit("no passed smoke on the store: run --plan session1 first")
        status["smoke"] = s
        backend = s["backend"]
        if not s["a4q3_ok"]:
            attacks = [a for a in attacks if a != "A4q3"]
    done = done_on_hub()
    cells = [(r, a) for a in attacks for r in args.runs.split(",") if f"{r}__{a}" not in done]
    print(f"[driver] backend={backend}; {len(done)} cells already done; {len(cells)} to run", flush=True)
    status["backend"], status["already_done"] = backend, sorted(done)
    status["cells"] = run_cells(cells, args.gpus.split(","), deadline, ff, backend)
    save_status(status, status_path)


def save_status(status: dict, path: Path) -> None:
    """Status JSON (cell names, return codes, timings; no generations) -> local + the private store."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(status, indent=1), encoding="utf-8")
    print(json.dumps(status, indent=1), flush=True)
    from huggingface_hub import HfApi

    from aftermarket.generate import ensure_private_repo

    HfApi().upload_file(path_or_fileobj=str(path), path_in_repo=f"stage2/kaggle/{path.name}",
                        repo_id=ensure_private_repo(STORE), repo_type="dataset")


if __name__ == "__main__":
    main()
