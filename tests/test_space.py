"""Space: the public bundle allowlist, and the app's data functions (gradio itself not needed)."""

import importlib.util
from pathlib import Path

import pandas as pd
import pytest

from aftermarket import space_push as S

REPO = Path(__file__).resolve().parents[1]


def app():
    spec = importlib.util.spec_from_file_location("space_app", REPO / "space" / "app.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def ci_rows():
    return pd.DataFrame([
        {"stage": "test", "run": "BASE", "attack": "A0", "metric": "asr_lg", "value": .1, "lo": .05, "hi": .15, "n": 200},
        {"stage": "test", "run": "SDT", "attack": "A3", "metric": "asr_lg", "value": .7, "lo": .6, "hi": .8, "n": 200},
        {"stage": "dev", "run": "BASE", "attack": "A0", "metric": "asr_lg", "value": .2, "lo": .1, "hi": .3, "n": 100},
        {"stage": "test", "run": "AMc_k6", "attack": "A0", "metric": "asr_lg", "value": .0, "lo": 0, "hi": 0, "n": 200}])


def tables(tmp_path, extra_cols=None):
    d = tmp_path / "tables"
    d.mkdir()
    ci = ci_rows()
    for k, v in (extra_cols or {}).items():
        ci[k] = v
    ci.to_csv(d / "cell_ci.csv", index=False)
    pd.DataFrame([{"stage": "test", "run": "BASE", "metric": "asr_lg", "retention": .5, "lo": .4, "hi": .6, "n": 200,
                   "families": "A1,A2,A3,A4,A5", "attacks": "A1", "partial": False}]).to_csv(d / "retention.csv",
                                                                                           index=False)
    pd.DataFrame([{"x": 1}]).to_csv(d / "pairwise.csv", index=False)  # not allowlisted: must not be copied
    return d


def test_bundle_keeps_test_rows_and_allowlisted_files_only(tmp_path):
    out = tmp_path / "data"
    counts = S.build_bundle(tables(tmp_path), out)
    assert counts == {"cell_ci.csv": 2, "retention.csv": 1}
    assert sorted(p.name for p in out.iterdir()) == ["cell_ci.csv", "retention.csv"]
    ci = pd.read_csv(out / "cell_ci.csv")
    assert set(ci["stage"]) == {"test"} and "AMc_k6" not in set(ci["run"])


def test_bundle_drops_unlisted_columns_and_rejects_text(tmp_path):
    out = tmp_path / "data"
    S.build_bundle(tables(tmp_path, {"response": "SECRET"}), out)  # unlisted column: dropped on copy
    assert "response" not in pd.read_csv(out / "cell_ci.csv").columns
    pd.DataFrame([{"stage": "test", "run": "BASE", "prompt": "SECRET"}]).to_csv(out / "cell_ci.csv", index=False)
    with pytest.raises(S.UnsafeBundle):
        S.check_bundle(out)  # anything that slips in is refused before upload
    (out / "cell_ci.csv").unlink()
    (out / "notes.txt").write_text("x")
    with pytest.raises(S.UnsafeBundle):
        S.check_bundle(out)


def test_bundle_needs_results(tmp_path):
    with pytest.raises(FileNotFoundError):
        S.build_bundle(tmp_path / "nothing", tmp_path / "data")


def test_app_tables_and_filters():
    a = app()
    ci = ci_rows()
    ci = ci[ci["stage"] == "test"]
    t = a.cell_table(ci, "asr_lg")
    assert list(t.columns[:1]) == ["model"] and t.iloc[0]["model"] == "BASE"
    assert t.set_index("model").loc["SDT", "A3"] == "70% [60%–80%]"
    only = a.cell_table(ci, "asr_lg", runs=["SDT"], attacks=["A3"])
    assert list(only["model"]) == ["SDT"]
    assert "No results" in a.cell_table(ci, "mmlu").iloc[0, 0]
    assert "No judge" in a.judges_table(pd.DataFrame(), pd.DataFrame()).iloc[0, 0]


def test_app_loads_missing_data_and_draws(tmp_path):
    pytest.importorskip("matplotlib")
    a = app()
    d = a.load_data(tmp_path)  # no CSVs at all: empty frames, no crash
    assert all(v.empty for v in d.values())
    assert a.heatmap(ci_rows(), "asr_lg", True) is not None
    assert a.retention_figure(d["retention"]) is not None
