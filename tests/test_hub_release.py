"""The public Hub release must never carry model outputs, weights other than the frozen defense adapters, or
dev candidates."""

import json

import pytest

from aftermarket import hub_release as H


def test_dataset_check_rejects_outputs(tmp_path):
    (tmp_path / "splits").mkdir()
    (tmp_path / "splits" / "x.jsonl").write_text(json.dumps({"id": "a", "prompt": "p"}) + "\n", encoding="utf-8")
    assert H.check_dataset(tmp_path) == 1
    for name, content in [("gen.jsonl", {"id": "a", "response": "r"}), ("m.json", {"cfg": {"guard_raw": "x"}})]:
        p = tmp_path / name
        p.write_text(json.dumps(content), encoding="utf-8")
        with pytest.raises(H.UnsafeRelease, match="output fields"):
            H.check_dataset(tmp_path)
        p.unlink()
    (tmp_path / "t.csv").write_text("id,completion\n1,x\n", encoding="utf-8")
    with pytest.raises(H.UnsafeRelease, match="output columns"):
        H.check_dataset(tmp_path)
    (tmp_path / "t.csv").unlink()
    (tmp_path / "w.parquet").write_bytes(b"x")
    with pytest.raises(H.UnsafeRelease, match="not allowed"):
        H.check_dataset(tmp_path)


def test_model_check_only_frozen_defense_adapters(tmp_path):
    d = tmp_path / "AM"
    d.mkdir()
    for f in H.ADAPTER_FILES + ["README.md"]:
        (d / f).write_text("{}" if f.endswith(".json") else "x", encoding="utf-8")
    (d / "defense_meta.json").write_text(json.dumps({"id": "AM", "kind": "final"}), encoding="utf-8")
    H.check_models([d])
    (d / "defense_meta.json").write_text(json.dumps({"id": "AM", "kind": "candidate"}), encoding="utf-8")
    with pytest.raises(H.UnsafeRelease):
        H.check_models([d])
    (d / "defense_meta.json").write_text(json.dumps({"id": "AM", "kind": "final"}), encoding="utf-8")
    (d / "attack_adapter.safetensors").write_text("x")
    with pytest.raises(H.UnsafeRelease, match="unexpected files"):
        H.check_models([d])
    assert set(H.MODELS) == {"AM", "AM_rand0", "AM_rand1", "AM_all"}


def test_public_repo_names_never_hit_the_private_store():
    names = {f"{H.OWNER}/{n}".lower() for n in H.REPO_NAMES.values()}
    assert H.STORE.lower() not in names and H.DATASET.lower() != "bennytimz/aftermarket-runs"
    assert len(names) == len(H.REPO_NAMES)  # distinct even case-insensitively


def test_publish_refuses_private_or_store_targets():
    from types import SimpleNamespace

    class Api:
        def __init__(self, rid, private):
            self.info = SimpleNamespace(id=rid, private=private)

        def repo_info(self, *a, **k):
            return self.info

    with pytest.raises(H.UnsafeRelease):
        H.ensure_public_target(Api("bennytimz/aftermarket-am", True), "bennytimz/aftermarket-AM", "model")
    with pytest.raises(H.UnsafeRelease):
        H.ensure_public_target(Api("bennytimz/aftermarket-am", False), "bennytimz/aftermarket-AM", "model")
    H.ensure_public_target(Api("bennytimz/aftermarket-AM-frozen", False), "bennytimz/aftermarket-AM-frozen", "model")
