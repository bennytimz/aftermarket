"""Schema checks for configs/models.yaml (PLAN.md §3)."""

import re

import pytest

from aftermarket.load import FORMATS, STATUSES, load_registry

SHA = re.compile(r"^[0-9a-f]{40}$")
REG = load_registry()
ALL = {**REG["models"], **REG.get("optional", {})}


def test_base_pinned():
    assert REG["base_model"] == "CohereLabs/tiny-aya-global"
    assert REG["base_revision"] == "af89d219b53ed9b13b8a4645f9c8028973510324"
    assert REG["models"]["BASE"]["revision"] == REG["base_revision"]


def test_required_ids_present():
    assert {"BASE", "SDT", "GC", "IL", "RL", "ST", "SP", "AM"} <= set(REG["models"])


@pytest.mark.parametrize("mid,spec", ALL.items())
def test_common_fields(mid, spec):
    assert spec["status"] in STATUSES, mid
    assert spec.get("name"), mid
    if spec["status"] == "dropped":
        assert spec.get("drop_reason"), f"{mid}: dropped models need drop_reason"


@pytest.mark.parametrize("mid,spec", [(k, v) for k, v in ALL.items() if v["status"] == "active"])
def test_active_fields(mid, spec):
    assert spec["format"] in FORMATS, mid
    assert "/" in spec["hf_path"] and "TODO" not in spec["hf_path"], mid
    assert SHA.match(spec["revision"]), f"{mid}: revision must be a pinned 40-char commit"
    assert spec["files"], mid
    if mid != "BASE":
        assert spec.get("base") == "BASE", mid
    if spec["format"] == "gguf":
        assert spec["gguf_file"] in spec["files"]
    if spec["format"] == "deltas":
        assert spec["delta_files"] and set(spec["delta_files"]) <= set(spec["files"])
    if spec["format"] == "lora":
        assert {"adapter_config.json", "adapter_model.safetensors"} <= set(spec["files"])
    if spec.get("chat_template_file"):
        assert spec["chat_template_file"] in spec["files"]
