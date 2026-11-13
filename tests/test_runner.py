"""Job runner planning: skip logic, GPU assignment, --final guard (no torch needed)."""

import pytest

from aftermarket.generate import (Job, assign_jobs, check_final, make_jobs, max_new_tokens, parquet_path,
                                  pending_jobs)

CFG = {"runs": ["BASE", "GC", "RL", "RL_noguard"], "attacks": ["A0"],
       "splits": ["harmful_test", "xstest_test", "mmlu_test"], "max_new_tokens": {"default": 256, "mmlu": 8}}


def test_make_jobs_and_names():
    jobs = make_jobs(CFG)
    assert len(jobs) == 12
    assert Job("RL_noguard", "A0", "mmlu_test").name == "RL_noguard__A0__mmlu_test"


def test_pending_skips_existing_parquet(tmp_path):
    jobs = make_jobs(CFG)
    parquet_path(tmp_path, jobs[0]).write_bytes(b"done")
    parquet_path(tmp_path, jobs[1]).with_suffix(".parquet.tmp").write_bytes(b"partial")  # unfinished
    todo = pending_jobs(jobs, tmp_path)
    assert jobs[0] not in todo and jobs[1] in todo and len(todo) == 11


def test_all_done_means_nothing_pending(tmp_path):
    jobs = make_jobs(CFG)
    for j in jobs:
        parquet_path(tmp_path, j).write_bytes(b"x")
    assert pending_jobs(jobs, tmp_path) == []


def test_assign_keeps_runs_together_and_covers_all():
    jobs = make_jobs(CFG)
    plan = assign_jobs(jobs, 2, CFG)
    assert sorted(j for w in plan for j in w) == sorted(jobs)
    for run in CFG["runs"]:
        assert sum(any(j.run_id == run for j in w) for w in plan) == 1, run
    assert all(plan)  # both GPUs used


def test_final_guard():
    with pytest.raises(SystemExit):
        check_final(["harmful_test"], final=False)
    check_final(["harmful_test"], final=True)
    check_final(["harmful_dev", "mmlu_dev"], final=False)


def test_max_new_tokens():
    assert max_new_tokens(CFG, "mmlu_test") == 8
    assert max_new_tokens(CFG, "harmful_test") == 256
