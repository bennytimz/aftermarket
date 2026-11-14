"""Job claims on the private runs dataset, so Colab and Kaggle can share one job list without doing a job twice.

Before a worker starts a job it writes `locks/<path_in_repo>/<job>.json` (owner, time) to the Hub in one commit
whose parent is the head it read (compare-and-swap: if anyone else committed in between, the commit is refused and
the claim retried on the new head). A job is skipped when its result parquet is already on the Hub, or when another
session holds a lock younger than LOCK_TTL_S (2 h). An older lock is stale (its session died) and is taken over.
A session may always re-take its own locks (same AFTERMARKET_SESSION), e.g. after a crash in the same notebook.
Locks are released (deleted) when the worker is done with the job's group, success or not.

If the Hub cannot be reached, or pushing is disabled, there is nothing to share: the job runs unclaimed (warning).
If the claim commit keeps conflicting, the job is skipped (it stays pending for the next session).
"""

from __future__ import annotations

import json
import os
import random
import socket
import time
import uuid

LOCK_TTL_S = 2 * 3600
SESSION_ENV = "AFTERMARKET_SESSION"
ATTEMPTS = 6


def lock_path(sub: str, name: str) -> str:
    return f"locks/{sub}/{name}.json"


def session_id() -> str:
    """This session's owner id (set once per process tree: workers inherit it through the environment)."""
    if not os.environ.get(SESSION_ENV):
        where = "colab" if os.environ.get("COLAB_RELEASE_TAG") else "kaggle" if os.path.isdir("/kaggle") else "local"
        os.environ[SESSION_ENV] = f"{where}-{socket.gethostname()[:16]}-{uuid.uuid4().hex[:8]}"
    return os.environ[SESSION_ENV]


def lock_state(lock: dict | None, owner: str, now: float, ttl: float = LOCK_TTL_S) -> str:
    """'free' (no lock), 'mine', 'stale' (another session's, older than ttl) or 'held' (another session's, fresh)."""
    if not lock:
        return "free"
    if lock.get("owner") == owner:
        return "mine"
    return "stale" if now - float(lock.get("epoch") or 0) >= ttl else "held"


def lock_record(name: str, owner: str, now: float) -> dict:
    return {"job": name, "owner": owner, "epoch": now,
            "time_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(now)), "ttl_s": LOCK_TTL_S}


class HfHub:
    """The few Hub calls claims need (tests swap in a fake)."""

    def __init__(self, repo_id: str):
        from huggingface_hub import HfApi

        self.repo_id, self.api = repo_id, HfApi()

    def head(self) -> str:
        return self.api.repo_info(self.repo_id, repo_type="dataset").sha

    def files(self, rev: str) -> set[str]:
        return set(self.api.list_repo_files(self.repo_id, repo_type="dataset", revision=rev))

    def read(self, path: str, rev: str) -> dict | None:
        from huggingface_hub import hf_hub_download

        try:
            p = hf_hub_download(self.repo_id, path, repo_type="dataset", revision=rev)
            with open(p, encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return None

    def commit(self, add: dict[str, bytes], delete: list[str], parent: str | None, message: str) -> None:
        from huggingface_hub import CommitOperationAdd, CommitOperationDelete

        ops = [CommitOperationAdd(path_in_repo=p, path_or_fileobj=b) for p, b in add.items()]
        ops += [CommitOperationDelete(path_in_repo=p) for p in delete]
        self.api.create_commit(self.repo_id, ops, repo_type="dataset", commit_message=message, parent_commit=parent)


def hub_for(cfg: dict):
    """(hub, sub) for a generate config, or (None, why) when claims can't be shared."""
    from aftermarket.generate import PUSH_DISABLED_ENV, resolve_repo_id

    push = cfg.get("push", {})
    if not push.get("enabled") or not push.get("hf_repo"):
        return None, "push disabled in config"
    if os.environ.get(PUSH_DISABLED_ENV):
        return None, os.environ[PUSH_DISABLED_ENV]
    try:
        return HfHub(resolve_repo_id(push["hf_repo"])), push["path_in_repo"]
    except Exception as e:
        return None, f"{type(e).__name__}: {e}"


def claim(hub, sub: str, names: list[str], owner: str, clock=time.time, sleep=time.sleep,
          log=print) -> list[str]:
    """Claim (or refresh) every name that is not done on the Hub and not held by another session.
    Returns the names claimed. One commit for all of them, retried on conflict."""
    for attempt in range(ATTEMPTS):
        try:
            head = hub.head()
            files = hub.files(head)
        except Exception as e:
            log(f"  WARNING claims: Hub unreachable ({type(e).__name__}: {e}); running {names} unclaimed")
            return list(names)
        now = clock()
        take, skipped = [], []
        for n in names:
            if f"{sub}/{n}.parquet" in files:
                skipped.append(f"{n} (done on the Hub)")
                continue
            lp = lock_path(sub, n)
            lock = hub.read(lp, head) if lp in files else None
            st = lock_state(lock, owner, now)
            if st == "held":
                age = (now - float(lock.get("epoch") or 0)) / 60
                skipped.append(f"{n} (locked by {lock.get('owner')} {age:.0f} min ago)")
            else:
                take.append(n)
        for s in skipped:
            log(f"  skip {s}")
        if not take:
            return []
        add = {lock_path(sub, n): json.dumps(lock_record(n, owner, now)).encode() for n in take}
        try:
            hub.commit(add, [], head, f"claim {len(take)} job(s): {take[0]}" + (" ..." if len(take) > 1 else ""))
            log(f"  claimed {take} as {owner}")
            return take
        except Exception as e:  # usually: head moved (another push or claim); re-read and retry
            log(f"  claim attempt {attempt + 1} refused ({type(e).__name__}: {str(e).splitlines()[0][:120] if str(e) else ''})")
            sleep(2 + 4 * random.random() * (attempt + 1))
    log(f"  WARNING claims: could not commit a claim for {names} after {ATTEMPTS} attempts; skipping them this session")
    return []


def release(hub, sub: str, names: list[str], owner: str, log=print) -> list[str]:
    """Delete this session's locks among `names` (another session's takeover is left alone). Best effort."""
    for attempt in range(3):
        try:
            head = hub.head()
            files = hub.files(head)
            mine = [n for n in names if lock_path(sub, n) in files
                    and lock_state(hub.read(lock_path(sub, n), head), owner, 0.0) == "mine"]
            if not mine:
                return []
            hub.commit({}, [lock_path(sub, n) for n in mine], head, f"release {len(mine)} job lock(s)")
            return mine
        except Exception as e:
            if attempt == 2:
                log(f"  WARNING could not release locks {names} ({type(e).__name__}); they expire after "
                    f"{LOCK_TTL_S // 3600} h")
            time.sleep(2 + 3 * random.random())
    return []


def held_by_others(hub, sub: str, owner: str, now: float | None = None) -> dict[str, dict]:
    """{job name: lock} for every fresh lock of another session (for status printouts)."""
    head = hub.head()
    now = time.time() if now is None else now
    out = {}
    for f in sorted(hub.files(head)):
        if f.startswith(f"locks/{sub}/") and f.endswith(".json"):
            lock = hub.read(f, head)
            if lock_state(lock, owner, now) == "held":
                out[f.rsplit("/", 1)[1][:-5]] = lock
    return out
