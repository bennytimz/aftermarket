"""Generation step (PLAN.md §5, §13.4).

    PYTHONPATH=src python -m aftermarket.generate --config configs/gen_a0.yaml --final

Runs (run_id, attack_id, split) jobs: load model -> apply attack -> generate -> one parquet per job in
results/gen/<run>__<attack>__<split>.parquet plus a .json manifest (git sha, config, seed,
time, throughput). Jobs whose parquet exists are skipped. Jobs are grouped by run (each
model loads once) and spread across GPUs, one worker process per GPU. After each job the
parquet and manifest are pushed to a PRIVATE HF dataset; the push refuses public repos
(PLAN.md §9). Judging happens later in a separate process (judge.py).

Attacks come from configs/attacks.yaml: weight attacks (A3 abliterate, A4 HQQ) change the
loaded model; prompt attacks (A5raw, A5bon) change the inputs. Each (run, attack) group loads a
fresh model once and runs all its splits. Attacked weights live in memory only.

Decoding: batched greedy, left padding with the pinned BASE tokenizer, batch size starting
at `batch_size` and halved on CUDA OOM. max_new_tokens 256 (harmful/XSTest), 8 (MMLU).
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from functools import lru_cache

import yaml

from aftermarket.data import load_split, verify_checksums

REPO_ROOT = Path(__file__).resolve().parents[2]
ATTACKS_YAML = REPO_ROOT / "configs" / "attacks.yaml"
IMPLEMENTED_KINDS = {"none", "abliterate", "hqq", "prompt", "finetune"}
PUSHABLE_SUFFIXES = (".parquet", ".json")  # generations + manifests only: never weights or adapters (PLAN.md §9)
TRAIN_COST = 60000.0  # rough cost units of one QLoRA run (~ several generation jobs), for GPU balancing
SPLIT_SIZES = {"harmful_test": 200, "harmful_dev": 100, "xstest_test": 250, "mmlu_test": 300, "mmlu_dev": 100,
               "benign_dev": 100}


def load_attacks(path: Path = ATTACKS_YAML) -> dict:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))["attacks"]


def attack_applies(attack: dict, split: str) -> bool:
    return not attack.get("splits") or any(split.startswith(p) for p in attack["splits"])


@dataclass(frozen=True, order=True)
class Job:
    run_id: str
    attack_id: str
    split: str

    @property
    def name(self) -> str:
        return f"{self.run_id}__{self.attack_id}__{self.split}"


# ----------------------------------------------------------------------------- planning (no torch)

def select_attacks(cfg: dict, attacks_arg: str | None) -> dict:
    """Restrict cfg["attacks"] to a comma-separated subset (must be in the config)."""
    if not attacks_arg:
        return cfg
    wanted = [a.strip() for a in attacks_arg.split(",") if a.strip()]
    unknown = [a for a in wanted if a not in cfg["attacks"]]
    if unknown:
        raise SystemExit(f"--attacks {unknown} not in config attacks {cfg['attacks']}")
    return {**cfg, "attacks": wanted}


def make_jobs(cfg: dict, attacks: dict | None = None) -> list[Job]:
    """All (run, attack, split) jobs, minus splits an attack doesn't apply to (e.g. BoN: harmful only)."""
    attacks = attacks or load_attacks()
    return [Job(r, a, s) for r in cfg["runs"] for a in cfg["attacks"] for s in cfg["splits"]
            if attack_applies(attacks[a], s)]


def parquet_path(out_dir: Path, job: Job) -> Path:
    return Path(out_dir) / f"{job.name}.parquet"


def pending_jobs(jobs: list[Job], out_dir: Path) -> list[Job]:
    """Jobs whose parquet does not exist yet (finished jobs are skipped)."""
    return [j for j in jobs if not parquet_path(out_dir, j).exists()]


def check_final(splits: list[str], final: bool) -> None:
    """Test splits are only generated with --final (CLAUDE.md test-set rule).

    Allowed: leaderboard cells (fixed attacks on others' models) and frozen AM. Never use test results to
    make a choice; tuning uses *_dev / calib_pairs only.
    """
    test = [s for s in splits if s.endswith("_test")]
    if test and not final:
        raise SystemExit(f"splits {test} are test sets: pass --final (leaderboard cells / frozen AM only; tune on dev splits)")


def check_defense_runs(runs: list[str], splits: list[str], registry: dict | None = None) -> None:
    """AM is frozen before its test run: AM-family runs on test must come from configs/am_frozen.yaml
    (pinned Hub revision), never from local dev-time folders; dev candidates (AMc_*) never touch test."""
    from aftermarket.load import load_registry

    test = [s for s in splits if s.endswith("_test")]
    if not test:
        return
    reg = registry or load_registry()
    for r in runs:
        if r.startswith("AMc_"):
            raise SystemExit(f"{r} is a dev-time AM candidate: it never runs on test splits")
        if r.startswith("AM"):
            spec = reg["models"].get(r, {})
            if spec.get("status") != "active" or spec.get("local") or not spec.get("revision"):
                raise SystemExit(f"{r} is not frozen (configs/am_frozen.yaml with a pinned revision): no test run yet")


def select_runs(cfg: dict, runs_arg: str | None) -> dict:
    """Override cfg["runs"] (e.g. a dev-time AM candidate chosen at run time)."""
    if not runs_arg:
        return cfg
    return {**cfg, "runs": [r.strip() for r in runs_arg.split(",") if r.strip()]}


def max_new_tokens(cfg: dict, split: str) -> int:
    m = cfg["max_new_tokens"]
    return m["mmlu"] if split.startswith("mmlu") else m["default"]


def _job_cost(job: Job, cfg: dict, attacks: dict | None = None) -> float:
    a = (attacks or {}).get(job.attack_id, {})
    mult = a.get("n", 1) if a.get("mode") == "bon" else 1
    return SPLIT_SIZES.get(job.split, 200) * mult * (max_new_tokens(cfg, job.split) + 32)


def group_key(job: Job, attacks: dict | None = None) -> tuple[str, str]:
    """One model load (+ one training run) per group. Fine-tunes group by the run whose adapter is trained,
    so RL / RL_noguard share BASE's A1/A2 adapter (identical weights, data and seed)."""
    if (attacks or {}).get(job.attack_id, {}).get("kind") == "finetune":
        from aftermarket.attacks.finetune import train_source_run

        return train_source_run(job.run_id), job.attack_id
    return job.run_id, job.attack_id


def assign_jobs(jobs: list[Job], n_workers: int, cfg: dict, attacks: dict | None = None) -> list[list[Job]]:
    """Group jobs (group_key: one model load + attack each), then longest-first onto the least-loaded worker."""
    groups: dict[tuple[str, str], list[Job]] = {}
    for j in sorted(jobs):
        groups.setdefault(group_key(j, attacks), []).append(j)
    loads = [0.0] * n_workers
    out: list[list[Job]] = [[] for _ in range(n_workers)]

    def cost(js):
        train = TRAIN_COST if (attacks or {}).get(js[0].attack_id, {}).get("kind") == "finetune" else 0.0
        return sum(_job_cost(j, cfg, attacks) for j in js) + 2000 + train  # + model load (+ training)

    for key, js in sorted(groups.items(), key=lambda kv: (-cost(kv[1]), kv[0])):
        w = loads.index(min(loads))
        out[w] += js
        loads[w] += cost(js)
    return out


# ----------------------------------------------------------------------------- HF push (private only)

PUSH_DISABLED_ENV = "AFTERMARKET_PUSH_DISABLED"  # set by the runner when the preflight push check fails


class PublicRepoError(RuntimeError):
    """The target repo is public: generations must never go there (PLAN.md §9)."""


@lru_cache(maxsize=None)
def resolve_repo_id(name: str) -> str:
    """'aftermarket-runs' -> '<whoami name>/aftermarket-runs'; a full 'ns/name' is kept as is.

    The HF namespace comes from the logged-in token (HF usernames can differ in case from GitHub's).
    """
    if "/" in name:
        return name
    from huggingface_hub import whoami

    return f"{whoami()['name']}/{name}"


def ensure_private_repo(repo_id: str) -> str:
    """Create the dataset repo as private if missing; raise PublicRepoError if it is public. Returns the full id."""
    from huggingface_hub import HfApi

    repo_id = resolve_repo_id(repo_id)
    api = HfApi()
    api.create_repo(repo_id, repo_type="dataset", private=True, exist_ok=True)
    if not api.repo_info(repo_id, repo_type="dataset").private:
        raise PublicRepoError(f"{repo_id} is PUBLIC: refusing to push generations (PLAN.md §9)")
    return repo_id


def check_pushable(paths: list[Path]) -> None:
    """Only generation parquet + JSON manifests may leave the machine; never adapters or weights."""
    from aftermarket.attacks.finetune import ADAPTER_ROOT

    root = Path(ADAPTER_ROOT).resolve()
    for p in map(Path, paths):
        rp = p.resolve()
        if p.suffix not in PUSHABLE_SUFFIXES or rp == root or root in rp.parents or "adapter" in p.name.lower():
            raise PermissionError(f"refusing to push {p}: only result .parquet/.json files may be pushed (PLAN.md §9)")


def push_files(paths: list[Path], repo_id: str, path_in_repo: str) -> str:
    """Push files to a PRIVATE dataset repo (privacy re-checked on every push). Raises on any failure."""
    from huggingface_hub import CommitOperationAdd, HfApi

    check_pushable(paths)
    repo_id = ensure_private_repo(repo_id)
    ops = [CommitOperationAdd(path_in_repo=f"{path_in_repo}/{p.name}", path_or_fileobj=str(p)) for p in paths]
    HfApi().create_commit(repo_id, ops, repo_type="dataset", commit_message=f"add {paths[0].stem}")
    return repo_id


def try_push(paths: list[Path], cfg: dict) -> tuple[bool, str | None]:
    """Non-fatal push: (True, None) on success, else (False, reason) after a clear warning. Files stay local."""
    push = cfg.get("push", {})
    if not push.get("enabled"):
        return False, "push disabled in config"
    if os.environ.get(PUSH_DISABLED_ENV):
        return False, os.environ[PUSH_DISABLED_ENV]
    try:
        push_files(paths, push["hf_repo"], push["path_in_repo"])
        return True, None
    except PublicRepoError as e:
        reason = str(e)
    except Exception as e:
        reason = f"{type(e).__name__}: {e}"
    print(f"  WARNING push failed for {paths[0].name} (kept locally, pushed=false): {reason}", flush=True)
    return False, reason


def pull_from_hub(cfg: dict, out_dir: Path, path_in_repo: str | None = None, want=None,
                  suffixes: tuple[str, ...] = (".parquet", ".json")) -> int:
    """Resume support: download finished results from the private Hub repo into out_dir.

    Lists <namespace>/<hf_repo>/<path_in_repo>/*.parquet|json and copies each missing file into
    out_dir (manifest before parquet, so a parquet never appears without its manifest). `want(stem)`
    filters which results to fetch (e.g. only this config's jobs). Never overwrites local files.
    Non-fatal: on any error it warns and returns what it got. Returns the number of files pulled.
    """
    push = cfg.get("push", {})
    sub = path_in_repo or push.get("path_in_repo")
    if not push.get("hf_repo") or not sub:
        return 0
    try:
        from huggingface_hub import HfApi

        repo_id = resolve_repo_id(push["hf_repo"])
        files = [f for f in HfApi().list_repo_files(repo_id, repo_type="dataset")
                 if f.startswith(sub + "/") and f.endswith(suffixes) and f.count("/") == 1]
    except Exception as e:
        print(f"WARNING resume-from-Hub skipped ({type(e).__name__}: {e}); starting from local files only", flush=True)
        return 0
    from huggingface_hub import hf_hub_download

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    todo = [f for f in files if not (out_dir / f.split("/", 1)[1]).exists()
            and (want is None or want(Path(f).stem))]
    n = 0
    for f in sorted(todo, key=lambda f: (Path(f).stem, f.endswith(".parquet"))):  # .json before .parquet
        dest = out_dir / f.split("/", 1)[1]
        try:
            src = hf_hub_download(repo_id, f, repo_type="dataset")
            tmp = dest.with_name(dest.name + ".pull")
            shutil.copyfile(src, tmp)
            tmp.replace(dest)
            n += 1
        except Exception as e:
            print(f"WARNING could not pull {f}: {type(e).__name__}: {e}", flush=True)
    print(f"resume-from-Hub: pulled {n} files from {repo_id}/{sub} ({len(files)} there)", flush=True)
    return n


def push_pending(cfg: dict, out_dir: Path) -> list[str]:
    """Push every local result whose manifest says pushed=false. Returns names still unpushed."""
    left = []
    for mpath in sorted(Path(out_dir).glob("*.json")):
        m = json.loads(mpath.read_text(encoding="utf-8"))
        pq = mpath.with_suffix(".parquet")
        if m.get("pushed") or not pq.exists():
            continue
        m.pop("push_error", None)
        m["pushed"] = True
        mpath.write_text(json.dumps(m, indent=1), encoding="utf-8")
        ok, err = try_push([pq, mpath], cfg)
        if not ok:
            m["pushed"], m["push_error"] = False, err
            mpath.write_text(json.dumps(m, indent=1), encoding="utf-8")
            left.append(mpath.stem)
        else:
            print(f"pushed {mpath.stem}", flush=True)
    return left


# ----------------------------------------------------------------------------- generation (torch)

def prepare_model(run_id: str, attack_id: str, attack: dict, cfg: dict):
    """Load the run's model and apply a weight attack. Returns (model, tok, attack_info)."""
    from aftermarket.load import load_model, resolve_run

    model_id, template = resolve_run(run_id)
    kind = attack["kind"]
    if kind not in IMPLEMENTED_KINDS:
        raise NotImplementedError(f"attack {attack_id} ({kind}) not implemented yet")
    if kind == "hqq":
        from aftermarket.attacks.compress import load_hqq

        return load_hqq(model_id, template, attack["nbits"], attack.get("group_size", 64))
    if kind == "finetune":  # trains once here; info["_cleanup"] deletes the local adapter after generation
        from aftermarket.attacks.finetune import finetune_and_load

        model, tok, info, cleanup = finetune_and_load(run_id, attack_id, attack, cfg)
        return model, tok, {**info, "_cleanup": cleanup}
    model, tok = load_model(model_id, precision=cfg["precision"], template=template)
    info = {"attack": attack_id, "kind": kind}
    if kind == "abliterate":
        from aftermarket.attacks import abliterate

        info.update(abliterate.apply(model, tok, attack))
    return model, tok, info


def _check_finite(model, tok) -> None:
    import torch

    ids = tok("Hello, how are you?", return_tensors="pt").input_ids.to(model.device)
    with torch.no_grad():
        logits = model(ids).logits
    if not torch.isfinite(logits).all():
        raise RuntimeError("non-finite logits: this model overflows in fp16; set precision: bf16")


def generate_batched(model, tok, prompt_ids: list[list[int]], max_new: int, batch_size: int) -> tuple[list[dict], int]:
    """Greedy-decode every prompt; returns per-prompt {response, n_new_tokens, finish_reason} and final batch size."""
    import torch

    from aftermarket.load import _stop_ids

    stops = _stop_ids(tok)
    pad = tok.pad_token_id
    order = sorted(range(len(prompt_ids)), key=lambda i: -len(prompt_ids[i]))  # longest first: OOM shows early
    results: list[dict | None] = [None] * len(prompt_ids)
    i, bs = 0, batch_size
    while i < len(order):
        chunk = order[i:i + bs]
        L = max(len(prompt_ids[k]) for k in chunk)
        x = torch.tensor([[pad] * (L - len(prompt_ids[k])) + prompt_ids[k] for k in chunk], device=model.device)
        mask = torch.tensor([[0] * (L - len(prompt_ids[k])) + [1] * len(prompt_ids[k]) for k in chunk], device=model.device)
        try:
            with torch.no_grad():
                y = model.generate(input_ids=x, attention_mask=mask, max_new_tokens=max_new, do_sample=False,
                                   eos_token_id=sorted(stops), pad_token_id=pad)
        except torch.cuda.OutOfMemoryError:
            del x, mask
            torch.cuda.empty_cache()
            if bs == 1:
                raise
            bs //= 2
            print(f"  OOM -> batch size {bs}", flush=True)
            continue
        for row, k in zip(y[:, L:].tolist(), chunk):
            cut = next((t for t, tok_id in enumerate(row) if tok_id in stops), None)
            new = row if cut is None else row[:cut]
            results[k] = {"response": tok.decode(new, skip_special_tokens=True).strip(),
                          "n_new_tokens": len(new), "finish_reason": "length" if cut is None else "stop"}
        i += len(chunk)
    return results, bs


def _git_sha() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=REPO_ROOT).stdout.strip()
    except OSError:
        return ""


def run_job(model, tok, job: Job, cfg: dict, out_dir: Path, attack: dict, attack_info: dict) -> dict:
    """Generate one job, write parquet + manifest, push. Returns the manifest."""
    import pandas as pd

    from aftermarket.attacks.prompt import build_inputs
    from aftermarket.load import resolve_run

    model_id, template = resolve_run(job.run_id)
    split_rows = load_split(job.split)
    expanded = build_inputs(model, tok, split_rows, attack, cfg["seed"])
    per = len(expanded) // len(split_rows)
    rows = [r for r in split_rows for _ in range(per)]
    extras = [e for e, _ in expanded]
    ids = [i for _, i in expanded]
    t0 = time.time()
    outs, final_bs = generate_batched(model, tok, ids, max_new_tokens(cfg, job.split), cfg["batch_size"])
    secs = time.time() - t0
    n_tok = sum(o["n_new_tokens"] for o in outs)
    spec = model.aftermarket_spec
    df = pd.DataFrame([{
        "run_id": job.run_id, "model_id": model_id, "template": template, "attack_id": job.attack_id,
        "split": job.split, "item_id": r["id"], "source": r.get("source"), "category": r.get("category"),
        "prompt": r["prompt"], "answer": r.get("answer"), **e, "prompt_tokens": len(i), **o,
    } for r, e, i, o in zip(rows, extras, ids, outs)])
    out_dir.mkdir(parents=True, exist_ok=True)
    pq = parquet_path(out_dir, job)
    tmp = pq.with_suffix(".parquet.tmp")
    df.to_parquet(tmp, index=False)
    manifest = {
        "job": asdict(job), "model_id": model_id, "template": template, "hf_path": spec["hf_path"],
        "revision": spec.get("revision"), "precision": cfg["precision"], "seed": cfg["seed"],
        "max_new_tokens": max_new_tokens(cfg, job.split), "batch_size_final": final_bs, "n_prompts": len(rows),
        "n_items": len(split_rows), "attack": attack, "attack_info": attack_info,
        "seconds": round(secs, 1), "prompts_per_sec": round(len(rows) / secs, 3),
        "new_tokens_per_sec": round(n_tok / secs, 1),
        "finish_length_frac": round(float((df.finish_reason == "length").mean()), 3),
        "git_sha": _git_sha(), "time_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "config": cfg,
    }
    mpath = pq.with_suffix(".json")
    # the pushed copy says pushed=true; if the push fails the local manifest is rewritten with pushed=false
    manifest["pushed"] = bool(cfg.get("push", {}).get("enabled")) and not os.environ.get(PUSH_DISABLED_ENV)
    mpath.write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    tmp.replace(pq)  # parquet appears only when complete: skip logic never sees partial files
    ok, err = try_push([pq, mpath], cfg)
    if not ok:
        manifest["pushed"], manifest["push_error"] = False, err
        mpath.write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    return manifest


def run_worker(jobs: list[Job], cfg: dict, out_dir: Path) -> None:
    """Run jobs on this process's GPU: per (run, attack), load + attack once, then every split."""
    import torch

    from aftermarket.load import free_memory, retarget

    attacks = load_attacks()
    torch.manual_seed(cfg["seed"])
    groups: dict[tuple[str, str], list[Job]] = {}
    for j in jobs:
        groups.setdefault(group_key(j, attacks), []).append(j)
    for (_, attack_id), js in groups.items():
        js = pending_jobs(js, out_dir)
        if not js:
            continue
        run_id = js[0].run_id
        shared = sorted({j.run_id for j in js})
        model = tok = None
        cleanup = None
        try:  # one failing (run, attack) must not stop the others on this GPU
            t0 = time.time()
            attack = attacks[attack_id]
            model, tok, info = prepare_model(run_id, attack_id, attack, cfg)
            cleanup = info.pop("_cleanup", None)
            _check_finite(model, tok)
            tok.padding_side = "left"
            brief = {k: v for k, v in info.items() if k != "separation"}
            print(f"[{run_id}/{attack_id}] ready in {time.time() - t0:.0f}s {brief}", flush=True)
            for job in js:
                if parquet_path(out_dir, job).exists():
                    continue
                job_info = info
                if len(shared) > 1:  # e.g. BASE / RL / RL_noguard on one adapter: only the eval template differs
                    retarget(model, tok, job.run_id)
                    job_info = {**info, "adapter_shared_with": shared,
                                "adapter_note": "template-only fixes share BASE's weights; one adapter trained "
                                                "on BASE, eval chat template per run"}
                m = run_job(model, tok, job, cfg, out_dir, attack, job_info)
                print(f"[{job.name}] {m['n_prompts']} prompts in {m['seconds']}s = {m['prompts_per_sec']} prompts/s, "
                      f"{m['new_tokens_per_sec']} tok/s, bs={m['batch_size_final']}, "
                      f"hit max_new_tokens={m['finish_length_frac']:.0%}", flush=True)
        except Exception as e:
            import traceback

            print(f"[{run_id}/{attack_id}] FAILED:", flush=True)
            traceback.print_exc(file=sys.stdout)
            # machine-readable line: the parent collects these for its end-of-stage FAILED summary
            print(FAIL_MARKER + json.dumps({"jobs": [j.name for j in js], "error": one_line_error(e)}), flush=True)
        finally:
            if cleanup:  # fine-tune adapters: deleted after the group's generations, success or not
                cleanup()
        del model, tok
        free_memory()


# ----------------------------------------------------------------------------- multi-GPU runner

FAIL_MARKER = "AFTERMARKET_FAILED "


def one_line_error(e: BaseException, limit: int = 300) -> str:
    """'ExcType: first line of the message', capped at `limit` chars."""
    lines = str(e).strip().splitlines()
    msg = f"{type(e).__name__}: {lines[0] if lines else ''}".rstrip(": ")
    return msg if len(msg) <= limit else msg[:limit - 3] + "..."


def parse_fail_line(line: str) -> dict[str, str]:
    """{job name: error} from one worker FAIL_MARKER line; {} for any other line."""
    i = line.find(FAIL_MARKER)
    if i < 0:
        return {}
    try:
        rec = json.loads(line[i + len(FAIL_MARKER):])
        return {name: rec["error"] for name in rec["jobs"]}
    except (ValueError, KeyError, TypeError):
        return {}


def failure_summary(failed: list[Job], errors: dict[str, str], codes: dict[int, int]) -> list[str]:
    """One 'FAILED <job>: <error>' line per job still pending after the stage."""
    out = []
    for j in failed:
        err = errors.get(j.name) or f"no exception reported (worker exit codes {codes}; see logs/gpu*.log)"
        out.append(f"FAILED {j.name}: {err}")
    if failed:
        out.append(f"{len(failed)} job(s) FAILED")
    return out


def _stream(proc: subprocess.Popen, prefix: str, log: Path, errors: dict[str, str] | None = None) -> None:
    with open(log, "a", encoding="utf-8") as f:
        for line in proc.stdout:
            f.write(line)
            if errors is not None:
                errors.update(parse_fail_line(line))
            print(f"{prefix} {line}", end="", flush=True)


def run_jobs(cfg: dict, config_path: str, final: bool, pull: bool = True) -> list[Job]:
    """Plan, then launch one worker process per GPU (CUDA_VISIBLE_DEVICES) and wait."""
    out_dir = REPO_ROOT / cfg["out_dir"]
    check_final(cfg["splits"], final)
    check_defense_runs(cfg["runs"], cfg["splits"])
    verify_checksums()
    attacks = load_attacks()
    for a in cfg["attacks"]:
        if attacks[a]["kind"] not in IMPLEMENTED_KINDS:
            raise NotImplementedError(f"attack {a} not implemented yet")
    all_jobs = make_jobs(cfg, attacks)
    if pull:  # a fresh session has no local results/: fetch finished jobs so they are skipped
        names = {j.name for j in all_jobs}
        pull_from_hub(cfg, out_dir, want=lambda stem: stem in names)
    todo = pending_jobs(all_jobs, out_dir)
    print(f"{len(all_jobs)} jobs, {len(todo)} pending")
    if not todo:
        return []
    push_env = {}
    if cfg.get("push", {}).get("enabled"):
        try:  # preflight: generation never depends on the push
            print(f"pushing to private dataset {ensure_private_repo(cfg['push']['hf_repo'])}")
        except Exception as e:
            reason = f"preflight failed: {type(e).__name__}: {e}"
            print(f"WARNING: {reason}\nGenerating anyway; results stay local with pushed=false. "
                  f"Fix access, then: python -m aftermarket.generate --config {config_path} --push-pending", flush=True)
            push_env = {PUSH_DISABLED_ENV: reason[:500]}
    gpus = cfg["gpus"]
    plan = assign_jobs(todo, len(gpus), cfg, attacks)
    (out_dir / "logs").mkdir(parents=True, exist_ok=True)
    procs, threads = [], []
    errors: dict[str, str] = {}
    for gpu, js in zip(gpus, plan):
        if not js:
            continue
        print(f"GPU {gpu}: {[j.name for j in js]}")
        env = {**os.environ, **push_env, "CUDA_VISIBLE_DEVICES": str(gpu), "PYTHONUNBUFFERED": "1",
               "PYTHONPATH": str(REPO_ROOT / "src") + os.pathsep + os.environ.get("PYTHONPATH", "")}
        cmd = [sys.executable, "-m", "aftermarket.generate", "--config", config_path, "--worker",
               json.dumps([asdict(j) for j in js]), "--runs", ",".join(cfg["runs"])] + (["--final"] if final else [])
        p = subprocess.Popen(cmd, env=env, cwd=REPO_ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        t = threading.Thread(target=_stream, args=(p, f"[gpu{gpu}]", out_dir / "logs" / f"gpu{gpu}.log", errors),
                             daemon=True)
        t.start()
        procs.append((gpu, p))
        threads.append(t)
    codes = {gpu: p.wait() for gpu, p in procs}
    for t in threads:
        t.join()
    failed = pending_jobs(todo, out_dir)
    print(f"worker exit codes {codes}; still pending: {[j.name for j in failed]}")
    for line in failure_summary(failed, errors, codes):
        print(line, flush=True)
    return failed


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="batched greedy generation")
    ap.add_argument("--config", required=True)
    ap.add_argument("--final", action="store_true", help="allow test splits: leaderboard cells or frozen AM only")
    ap.add_argument("--attacks", help="comma-separated subset of the config's attacks to run now (cost ordering)")
    ap.add_argument("--runs", help="comma-separated runs replacing the config's runs (e.g. an AM candidate)")
    ap.add_argument("--push-pending", action="store_true", help="only push local results with pushed=false")
    ap.add_argument("--no-pull", action="store_true", help="don't download finished results from the Hub first")
    ap.add_argument("--gpus", help="comma-separated GPU ids replacing the config's (e.g. 0 on a one-T4 Colab)")
    ap.add_argument("--worker", help=argparse.SUPPRESS)
    args = ap.parse_args(argv)
    cfg = select_runs(select_attacks(yaml.safe_load(Path(args.config).read_text(encoding="utf-8")), args.attacks), args.runs)
    if args.gpus:
        cfg = {**cfg, "gpus": [int(g) for g in args.gpus.split(",")]}
    out_dir = REPO_ROOT / cfg["out_dir"]
    if args.push_pending:
        left = push_pending(cfg, out_dir)
        print(f"still unpushed: {left}")
        raise SystemExit(1 if left else 0)
    if args.worker:
        jobs = [Job(**d) for d in json.loads(args.worker)]
        check_final([j.split for j in jobs], args.final)
        check_defense_runs([j.run_id for j in jobs], [j.split for j in jobs])
        run_worker(jobs, cfg, out_dir)
    else:
        if run_jobs(cfg, args.config, args.final, pull=not args.no_pull):
            raise SystemExit(1)


if __name__ == "__main__":
    main()
