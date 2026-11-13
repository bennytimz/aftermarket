"""Split integrity: checksums, sizes, disjointness (PLAN.md §5, CLAUDE.md)."""

import random

import pytest

from aftermarket.data import (ALL_SPLITS, SPLITS_DIR, load_split, norm_text, read_checksums, read_jsonl,
                              sha256_file, stratified_take, verify_checksums)

SIZES = {"harmful_test": 200, "harmful_dev": 100, "xstest_test": 250, "mmlu_test": 300, "mmlu_dev": 100,
         "benign_dev": 100, "calib_pairs": 64}
S = {name: read_jsonl(SPLITS_DIR / f"{name}.jsonl") for name in ALL_SPLITS}


def test_checksums_cover_all_splits_and_match():
    assert set(read_checksums()) == {f"{n}.jsonl" for n in ALL_SPLITS}
    verify_checksums()


def test_checksum_mismatch_detected(tmp_path):
    (tmp_path / "harmful_test.jsonl").write_bytes((SPLITS_DIR / "harmful_test.jsonl").read_bytes() + b"\n")
    (tmp_path / "checksums.txt").write_text(f"{sha256_file(SPLITS_DIR / 'harmful_test.jsonl')}  harmful_test.jsonl\n")
    with pytest.raises(ValueError):
        verify_checksums(tmp_path)


@pytest.mark.parametrize("name,n", SIZES.items())
def test_sizes_and_unique_ids(name, n):
    rows = S[name]
    assert len(rows) == n
    assert len({r["id"] for r in rows}) == n


def _harmful_texts(name):
    if name == "calib_pairs":
        return {norm_text(r["harmful"]) for r in S[name]}
    return {norm_text(r["prompt"]) for r in S[name]}


def test_harmful_splits_disjoint():
    names = ["harmful_test", "harmful_dev", "calib_pairs"]
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            assert not _harmful_texts(a) & _harmful_texts(b), (a, b)
    calib_ids = {r["harmful_id"] for r in S["calib_pairs"]}
    assert not calib_ids & {r["id"] for r in S["harmful_test"] + S["harmful_dev"]}


def test_harmless_calib_not_in_any_eval_set():
    harmless = {norm_text(r["harmless"]) for r in S["calib_pairs"]}
    for name in ["harmful_test", "harmful_dev", "xstest_test", "mmlu_test", "mmlu_dev"]:
        assert not harmless & {norm_text(r["prompt"]) for r in S[name]}, name


def test_mmlu_disjoint_and_answer_format():
    assert not {norm_text(r["question"]) for r in S["mmlu_test"]} & {norm_text(r["question"]) for r in S["mmlu_dev"]}
    for r in S["mmlu_test"] + S["mmlu_dev"]:
        assert r["answer"] in "ABCD" and len(r["answer"]) == 1
        assert r["prompt"].endswith("Answer with the letter only (A, B, C or D).")
    assert len({r["category"] for r in S["mmlu_test"]}) == 57


def test_harmful_test_mix():
    assert sum(r["source"] == "harmbench" for r in S["harmful_test"]) == 100
    assert sum(r["source"] == "advbench" for r in S["harmful_test"]) == 100


def test_load_split_checks_checksum():
    assert len(load_split("xstest_test")) == 250


def test_stratified_take_deterministic_and_proportional():
    items = [{"id": f"{c}{i}", "c": c} for c in "ab" for i in range(50 if c == "a" else 150)]
    t1, r1 = stratified_take(items, 20, "c", random.Random(0))
    t2, _ = stratified_take(items, 20, "c", random.Random(0))
    assert t1 == t2 and len(r1) == 180
    assert sum(x["c"] == "a" for x in t1) == 5


def test_benign_dev_disjoint_from_everything():
    bd = {norm_text(r["prompt"]) for r in S["benign_dev"]}
    assert all(r["source"] == "orbench" for r in S["benign_dev"])
    for name in ["harmful_test", "harmful_dev", "xstest_test", "mmlu_test", "mmlu_dev"]:
        assert not bd & {norm_text(r["prompt"]) for r in S[name]}, name
    assert not bd & {norm_text(r[f]) for r in S["calib_pairs"] for f in ("harmful", "harmless")}


def _jaccard(a, b):
    ga = {a.lower()[i:i + 3] for i in range(len(a) - 2)}
    gb = {b.lower()[i:i + 3] for i in range(len(b) - 2)}
    return len(ga & gb) / len(ga | gb)


def test_benign_dev_not_near_duplicate_of_xstest():
    worst = max(_jaccard(a["prompt"], b["prompt"]) for a in S["benign_dev"] for b in S["xstest_test"])
    assert worst < 0.7
