"""P9.15: the dual W* bundle is a deterministic, pickle-free .npz whose fingerprint is the published one."""

from __future__ import annotations

import hashlib
import json
import sys

import numpy as np
import pytest

from src.crypto.fingerprint import fingerprint_state_dict
from src.utils.results import repo_root

ROOT = repo_root()
sys.path.insert(0, str(ROOT / "experiments"))
import p9_15_bundle_dual_w_star as bundle  # noqa: E402


def test_npz_bytes_are_deterministic_and_reload_bit_for_bit():
    arrays = {"b.weight": np.arange(6, dtype=np.float32).reshape(2, 3), "a.num_batches_tracked": np.array(7)}
    data = bundle.npz_bytes(arrays)
    assert data == bundle.npz_bytes(dict(reversed(list(arrays.items()))))
    back = bundle.load_npz(data)
    assert set(back) == set(arrays)
    for name, array in arrays.items():
        assert back[name].dtype == array.dtype and back[name].tobytes() == array.tobytes()
    assert fingerprint_state_dict(back).sha256 == fingerprint_state_dict(arrays).sha256


def test_object_arrays_are_refused():
    with pytest.raises(ValueError):
        bundle.npz_bytes({"x": np.array([object()], dtype=object)})


@pytest.mark.skipif(not (ROOT / bundle.BUNDLE).exists(), reason="bundle not written yet")
def test_committed_bundle_is_the_published_model():
    data = (ROOT / bundle.BUNDLE).read_bytes()
    published = json.loads((ROOT / bundle.PUBLICATION).read_text(encoding="utf-8"))["model_fingerprint"]
    assert fingerprint_state_dict(bundle.load_npz(data)).sha256 == published["sha256"]
    record = json.loads(sorted((ROOT / "results").glob("p9.15_dual_w_star_bundle__*.json"))[-1].read_text())
    assert record["metrics"]["bundle_sha256"] == hashlib.sha256(data).hexdigest()
