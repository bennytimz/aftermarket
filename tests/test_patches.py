"""ScarTissue-style rank-one patch loading and validation (numpy only)."""

import io

import numpy as np
import pytest

from aftermarket.load import load_patch, patch_delta, validate_patch


def _npz(d, alpha, cols, proj, name="model.layers.0.mlp.down_proj.weight"):
    buf = io.BytesIO()
    np.savez(buf, d=d, alpha=np.array(alpha), modules=np.array([name]), cols__0=cols, proj__0=proj)
    return buf.getvalue()


def test_roundtrip_and_delta_matches_formula():
    d = np.array([0.6, 0.8])
    cols = np.array([0, 2])
    proj = np.array([1.0, -2.0])
    p = load_patch(_npz(d, -1.0, cols, proj))
    expected = -1.0 * np.outer(d, proj)
    np.testing.assert_allclose(patch_delta(p, p["modules"][0]), expected, rtol=1e-6)
    assert patch_delta(p, p["modules"][0]).dtype == np.float32


@pytest.mark.parametrize("d,cols", [
    (np.array([1.0, 1.0]), np.array([0, 1])),     # not a unit vector
    (np.array([0.6, 0.8]), np.array([2, 1])),     # unsorted columns
    (np.array([0.6, 0.8]), np.array([-1, 1])),    # negative column
])
def test_rejects_bad_patches(d, cols):
    with pytest.raises(ValueError):
        load_patch(_npz(d, 4.0, cols, np.array([1.0, 1.0])))


def test_validate_rejects_key_mismatch():
    with pytest.raises(ValueError):
        validate_patch({"d": np.array([1.0]), "alpha": 1.0, "modules": ["a"], "proj": {}, "cols": {}})
