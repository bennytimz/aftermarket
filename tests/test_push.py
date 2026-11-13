"""HF push: namespace from whoami, non-fatal failures, strict privacy (HF mocked, no network)."""

import json
import types

import pytest

import aftermarket.generate as g

CFG = {"push": {"enabled": True, "hf_repo": "aftermarket-runs", "path_in_repo": "gen"}}


@pytest.fixture(autouse=True)
def _clear(monkeypatch):
    g.resolve_repo_id.cache_clear()
    monkeypatch.delenv(g.PUSH_DISABLED_ENV, raising=False)


def test_namespace_from_whoami(monkeypatch):
    import huggingface_hub

    monkeypatch.setattr(huggingface_hub, "whoami", lambda: {"name": "bennytimz"})
    assert g.resolve_repo_id("aftermarket-runs") == "bennytimz/aftermarket-runs"
    assert g.resolve_repo_id("someone/x") == "someone/x"


def _fake_api(monkeypatch, private: bool, calls: list):
    import huggingface_hub

    class Api:
        def create_repo(self, repo_id, **kw):
            calls.append(("create", repo_id, kw.get("private")))

        def repo_info(self, repo_id, **kw):
            return types.SimpleNamespace(private=private)

        def create_commit(self, repo_id, ops, **kw):
            calls.append(("commit", repo_id, len(ops)))

    monkeypatch.setattr(huggingface_hub, "HfApi", Api)
    monkeypatch.setattr(huggingface_hub, "whoami", lambda: {"name": "bennytimz"})


def test_push_goes_to_private_repo(monkeypatch, tmp_path):
    calls = []
    _fake_api(monkeypatch, private=True, calls=calls)
    f = tmp_path / "a.parquet"; f.write_bytes(b"x")
    assert g.try_push([f], CFG) == (True, None)
    assert ("create", "bennytimz/aftermarket-runs", True) in calls
    assert ("commit", "bennytimz/aftermarket-runs", 1) in calls


def test_never_pushes_to_public_repo(monkeypatch, tmp_path):
    calls = []
    _fake_api(monkeypatch, private=False, calls=calls)
    f = tmp_path / "a.parquet"; f.write_bytes(b"x")
    with pytest.raises(g.PublicRepoError):
        g.push_files([f], "aftermarket-runs", "gen")
    ok, reason = g.try_push([f], CFG)
    assert not ok and "PUBLIC" in reason
    assert not any(c[0] == "commit" for c in calls)


def test_push_failure_is_non_fatal(monkeypatch, tmp_path):
    def boom(*a, **k):
        raise RuntimeError("403 Forbidden")

    monkeypatch.setattr(g, "push_files", boom)
    f = tmp_path / "a.parquet"; f.write_bytes(b"x")
    ok, reason = g.try_push([f], CFG)
    assert not ok and "403" in reason and f.exists()


def test_push_disabled_by_preflight_env(monkeypatch, tmp_path):
    monkeypatch.setenv(g.PUSH_DISABLED_ENV, "preflight failed: 403")
    monkeypatch.setattr(g, "push_files", lambda *a, **k: pytest.fail("must not push"))
    assert g.try_push([tmp_path / "a.parquet"], CFG) == (False, "preflight failed: 403")
    assert g.try_push([tmp_path / "a.parquet"], {"push": {"enabled": False}})[0] is False


def test_push_pending_updates_manifests(monkeypatch, tmp_path):
    for name, pushed in [("a", False), ("b", True)]:
        (tmp_path / f"{name}.parquet").write_bytes(b"x")
        (tmp_path / f"{name}.json").write_text(json.dumps({"pushed": pushed, "push_error": "403"}))
    results = iter([(False, "still 403")])
    monkeypatch.setattr(g, "try_push", lambda paths, cfg: next(results))
    assert g.push_pending(CFG, tmp_path) == ["a"]
    m = json.loads((tmp_path / "a.json").read_text())
    assert m["pushed"] is False and m["push_error"] == "still 403"


def _fake_hub(monkeypatch, tmp_path, files: dict, fail_list=False):
    """files: repo path -> bytes. Mocks list_repo_files + hf_hub_download."""
    import huggingface_hub

    store = tmp_path / "hubcache"
    store.mkdir(exist_ok=True)

    class Api:
        def list_repo_files(self, repo_id, repo_type=None):
            if fail_list:
                raise RuntimeError("401 Unauthorized")
            assert repo_id == "bennytimz/aftermarket-runs" and repo_type == "dataset"
            return list(files) + [".gitattributes"]

    def download(repo_id, filename, repo_type=None):
        p = store / filename.replace("/", "__")
        p.write_bytes(files[filename])
        return str(p)

    monkeypatch.setattr(huggingface_hub, "HfApi", Api)
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", download)
    monkeypatch.setattr(huggingface_hub, "whoami", lambda: {"name": "bennytimz"})


def test_pull_resumes_finished_jobs(monkeypatch, tmp_path):
    jobs = g.make_jobs({"runs": ["BASE", "GC"], "attacks": ["A0"], "splits": ["harmful_test"]})
    done = jobs[0].name
    _fake_hub(monkeypatch, tmp_path, {f"gen/{done}.parquet": b"P", f"gen/{done}.json": b"{}",
                                      "gen/OTHER__A0__x.parquet": b"P", "judged/foo.parquet": b"J"})
    out = tmp_path / "gen"
    n = g.pull_from_hub(CFG, out, want=lambda stem: stem in {j.name for j in jobs})
    assert n == 2
    assert sorted(p.name for p in out.iterdir()) == [f"{done}.json", f"{done}.parquet"]
    assert g.pending_jobs(jobs, out) == [jobs[1]]


def test_pull_never_overwrites_local(monkeypatch, tmp_path):
    _fake_hub(monkeypatch, tmp_path, {"gen/a.parquet": b"HUB"})
    out = tmp_path / "gen"; out.mkdir()
    (out / "a.parquet").write_bytes(b"LOCAL")
    assert g.pull_from_hub(CFG, out) == 0
    assert (out / "a.parquet").read_bytes() == b"LOCAL"


def test_pull_failure_is_non_fatal(monkeypatch, tmp_path):
    _fake_hub(monkeypatch, tmp_path, {}, fail_list=True)
    assert g.pull_from_hub(CFG, tmp_path / "gen") == 0


def test_pull_judge_paths(monkeypatch, tmp_path):
    _fake_hub(monkeypatch, tmp_path, {"judged/x.parquet": b"J", "gen/x.parquet": b"G", "gen/y.parquet": b"G"})
    judged, gen = tmp_path / "judged", tmp_path / "gen"
    assert g.pull_from_hub(CFG, judged, "judged") == 1
    # only generations that are not judged yet
    assert g.pull_from_hub(CFG, gen, "gen", want=lambda s: not (judged / f"{s}.parquet").exists()) == 1
    assert [p.name for p in gen.iterdir()] == ["y.parquet"]
