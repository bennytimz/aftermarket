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
