"""Hub job claims (Colab + Kaggle sharing one job list): no torch, no network."""

import json

from aftermarket import claims
from aftermarket.attacks import a3_rerun
from aftermarket.generate import out_of_time

H = 3600.0


class FakeHub:
    """A dataset repo as {path: bytes} with a head sha; commit(parent=stale head) is refused like the real Hub."""

    def __init__(self, files=None):
        self.data = dict(files or {})
        self.rev = 0
        self.commits = 0
        self.before_commit = None  # hook: simulate another session committing in between

    def head(self):
        return str(self.rev)

    def files(self, rev):
        return set(self.data)

    def read(self, path, rev):
        return json.loads(self.data[path]) if path in self.data else None

    def commit(self, add, delete, parent, message):
        if self.before_commit:
            hook, self.before_commit = self.before_commit, None
            hook(self)
        if parent is not None and parent != self.head():
            raise RuntimeError("412 Precondition Failed: parent commit is not the head")
        for p in delete:
            del self.data[p]
        self.data.update(add)
        self.rev += 1
        self.commits += 1


def lock(owner, epoch):
    return json.dumps(claims.lock_record("x", owner, epoch)).encode()


def run_claim(hub, names, owner="colab-a", now=10 * H):
    return claims.claim(hub, "gen", names, owner, clock=lambda: now, sleep=lambda s: None, log=lambda m: None)


def test_lock_state():
    assert claims.lock_state(None, "a", 0) == "free"
    assert claims.lock_state({"owner": "a", "epoch": 0}, "a", 99 * H) == "mine"
    assert claims.lock_state({"owner": "b", "epoch": 9 * H}, "a", 10 * H) == "held"     # 1 h old
    assert claims.lock_state({"owner": "b", "epoch": 8 * H}, "a", 10 * H) == "stale"    # 2 h old


def test_claim_skips_done_and_fresh_locks_takes_stale_and_free():
    hub = FakeHub({
        "gen/BASE__A3__harmful_test.parquet": b"x",                               # done
        claims.lock_path("gen", "GC__A3__harmful_test"): lock("kaggle-b", 9.5 * H),  # fresh, other session
        claims.lock_path("gen", "IL__A3__harmful_test"): lock("kaggle-b", 7 * H),    # stale (3 h)
    })
    names = ["BASE__A3__harmful_test", "GC__A3__harmful_test", "IL__A3__harmful_test", "SP__A3__harmful_test"]
    got = run_claim(hub, names)
    assert got == ["IL__A3__harmful_test", "SP__A3__harmful_test"]
    assert hub.commits == 1
    assert json.loads(hub.data[claims.lock_path("gen", "IL__A3__harmful_test")])["owner"] == "colab-a"
    assert run_claim(hub, names, owner="kaggle-b") == ["GC__A3__harmful_test"]  # its own lock: refreshable


def test_race_second_session_sees_first_claim():
    hub = FakeHub()
    hub.before_commit = lambda h: run_claim(h, ["SP__A3__mmlu_test"], owner="kaggle-b")  # wins the race
    assert run_claim(hub, ["SP__A3__mmlu_test"], owner="colab-a") == []                  # retried, then skipped
    assert json.loads(hub.data[claims.lock_path("gen", "SP__A3__mmlu_test")])["owner"] == "kaggle-b"


def test_unreachable_hub_runs_unclaimed_and_endless_conflict_skips():
    class Down(FakeHub):
        def head(self):
            raise ConnectionError("offline")

    assert run_claim(Down(), ["a"]) == ["a"]

    class Busy(FakeHub):
        def commit(self, *a, **k):
            raise RuntimeError("412")

    assert run_claim(Busy(), ["a"]) == []


def test_release_deletes_only_own_locks():
    hub = FakeHub({claims.lock_path("gen", "a"): lock("colab-a", 0), claims.lock_path("gen", "b"): lock("kaggle-b", 0)})
    assert claims.release(hub, "gen", ["a", "b", "c"], "colab-a", log=lambda m: None) == ["a"]
    assert set(hub.data) == {claims.lock_path("gen", "b")}


def test_held_by_others():
    hub = FakeHub({claims.lock_path("gen", "a"): lock("kaggle-b", 9 * H), claims.lock_path("gen", "b"): lock("colab-a", 9 * H),
                   claims.lock_path("gen", "c"): lock("kaggle-b", 1 * H)})
    assert list(claims.held_by_others(hub, "gen", "colab-a", now=10 * H)) == ["a"]


def test_out_of_time():
    assert not out_of_time(None)
    assert out_of_time(100.0, now=100.0) and not out_of_time(100.0, now=99.0)


def test_current_results_ignores_v1_manifests(tmp_path):
    (tmp_path / "BASE__A3__mmlu_test.json").write_text(json.dumps({"attack_info": {"select_by": "bypass"}}))
    (tmp_path / "GC__A3__mmlu_test.json").write_text(json.dumps({"attack_info": {"layer": 7}}))  # v1
    names = {"BASE__A3__mmlu_test", "GC__A3__mmlu_test", "IL__A3__mmlu_test"}
    assert a3_rerun.current_results(tmp_path, names) == {"BASE__A3__mmlu_test"}


def test_a3_parts_job_counts_and_command():
    assert len(a3_rerun.part_jobs("dev")) == 6
    assert len(a3_rerun.part_jobs("team")) == 24
    assert len(a3_rerun.part_jobs("am")) == 12
    assert len(a3_rerun.part_jobs("all")) == 42
    assert {j.attack_id for j in a3_rerun.part_jobs("all")} == {"A3"}
    cmd = a3_rerun.generate_command("configs/gen_cheap_test.yaml", True, "0,1", 2.0, 123.0)
    assert all(isinstance(a, str) for a in cmd)
    assert cmd[cmd.index("--gpus") + 1] == "0,1" and "--final" in cmd and "--claim" in cmd
    assert "--final" not in a3_rerun.generate_command("configs/gen_cheap_dev.yaml", False, None, None, 1.0)
