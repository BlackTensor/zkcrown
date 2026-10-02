"""Tests for P5.1: the canonical SHA-256 model fingerprint."""

from __future__ import annotations

import hashlib
import os
import struct
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from src.crypto.fingerprint import (  # noqa: E402
    FINGERPRINT_VERSION,
    canonical_chunks,
    fingerprint_file,
    fingerprint_model,
    fingerprint_state_dict,
)
from src.models import main_model, zk_model  # noqa: E402
from src.utils.seeding import set_seed  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]


def fp(state) -> str:
    return fingerprint_state_dict(state).sha256


def fixed_state() -> dict:
    """A small state dict with exactly representable values, the same on every torch build."""
    return {
        "features.0.weight": torch.arange(24, dtype=torch.float32).reshape(2, 3, 2, 2) / 8 - 1,
        "features.1.running_mean": torch.tensor([0.5, -0.25], dtype=torch.float32),
        "features.1.num_batches_tracked": torch.tensor(7, dtype=torch.int64),
        "classifier.bias": torch.tensor([1.0, -2.0, 0.0], dtype=torch.float32),
    }


@pytest.fixture
def state():
    set_seed(11)
    return main_model().state_dict()


# --- the format ---------------------------------------------------------------


def test_known_answer_against_hand_assembled_bytes():
    """The serialization, written out here with struct alone, hashes to the same digest."""
    state = {
        "b": torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32),
        "a": torch.tensor(5, dtype=torch.int64),
    }
    expected = b"zk-crown/model-fingerprint/v1\x00" + struct.pack(">I", 2)
    expected += struct.pack(">H", 1) + b"a" + struct.pack(">B", 5) + b"int64" + struct.pack(">B", 0)
    expected += struct.pack(">Q", 8) + struct.pack("<q", 5)
    expected += struct.pack(">H", 1) + b"b" + struct.pack(">B", 7) + b"float32" + struct.pack(">B", 2)
    expected += struct.pack(">QQ", 2, 2) + struct.pack(">Q", 16) + struct.pack("<4f", 1.0, 2.0, 3.0, 4.0)

    assert b"".join(canonical_chunks(state)) == expected
    result = fingerprint_state_dict(state)
    assert result.sha256 == hashlib.sha256(expected).hexdigest()
    assert (result.version, result.tensors, result.elements, result.data_bytes) == (FINGERPRINT_VERSION, 2, 5, 24)
    assert result.to_dict()["sha256"] == result.sha256


def test_pinned_digest():
    """A change to the format must be deliberate: it changes this digest and needs a new version tag."""
    assert fp(fixed_state()) == PINNED


PINNED = "8e32834b342a8ca4935e7fdc64e8ea14c50426c16e43a146a374e9b944e6f6da"


def test_summary_counts(state):
    result = fingerprint_state_dict(state)
    assert result.tensors == len(state)
    assert result.elements == sum(t.numel() for t in state.values())
    assert result.data_bytes == sum(t.numel() * t.element_size() for t in state.values())
    assert len(result.sha256) == 64


# --- stable across save and reload --------------------------------------------


def test_stable_across_save_and_reload(state, tmp_path):
    reference = fp(state)
    torch.save(state, tmp_path / "one.pt")
    assert fingerprint_file(tmp_path / "one.pt").sha256 == reference
    torch.save(torch.load(tmp_path / "one.pt", weights_only=True), tmp_path / "another_name.pt")
    assert fingerprint_file(tmp_path / "another_name.pt").sha256 == reference
    torch.save(state, tmp_path / "legacy.pt", _use_new_zipfile_serialization=False)
    assert fingerprint_file(tmp_path / "legacy.pt").sha256 == reference


def test_stable_through_a_model(state):
    model = main_model()
    model.load_state_dict(state, strict=True)
    assert fingerprint_model(model).sha256 == fp(state)
    model.train()  # mode is not part of the weights
    for p in model.parameters():
        p.requires_grad_(False)
    assert fingerprint_model(model).sha256 == fp(state)


def test_stable_through_npz_without_torch(state, tmp_path):
    np.savez(tmp_path / "w.npz", **{k: t.numpy() for k, t in state.items()})
    with np.load(tmp_path / "w.npz") as archive:
        arrays = {k: archive[k] for k in archive.files}
    assert fp(arrays) == fp(state)


def test_independent_of_dict_order(state):
    assert fp(dict(reversed(list(state.items())))) == fp(state)


def test_independent_of_memory_layout_and_byte_order():
    state = fixed_state()
    reference = fp(state)
    w = state["features.0.weight"]
    view = w.transpose(0, 1).contiguous().transpose(0, 1)
    assert not view.is_contiguous() and torch.equal(view, w)
    assert fp({**state, "features.0.weight": view}) == reference
    fortran = np.asfortranarray(w.numpy())
    assert fp({**state, "features.0.weight": fortran}) == reference
    big_endian = w.numpy().astype(">f4")
    assert fp({**state, "features.0.weight": big_endian}) == reference
    assert fp({**state, "features.0.weight": w.clone().requires_grad_(True)}) == reference


def test_does_not_modify_its_input(state):
    before = {k: t.clone() for k, t in state.items()}
    fp(state)
    assert all(torch.equal(state[k], before[k]) for k in before)


def test_same_in_a_fresh_process_with_another_hash_seed(tmp_path):
    torch.save(fixed_state(), tmp_path / "w.pt")
    env = {**os.environ, "PYTHONHASHSEED": "987"}
    done = subprocess.run([sys.executable, "-m", "src.crypto.fingerprint", str(tmp_path / "w.pt")], cwd=REPO_ROOT,
                          env=env, capture_output=True, text=True, check=True)
    assert done.stdout.split()[0] == PINNED


def test_zk_model_round_trip(tmp_path):
    set_seed(3)
    model = zk_model()
    torch.save(model.state_dict(), tmp_path / "zk.pt")
    assert fingerprint_file(tmp_path / "zk.pt").sha256 == fingerprint_model(model).sha256


# --- sensitive to every change ------------------------------------------------


def test_one_bit_in_any_tensor_changes_it(state):
    reference = fp(state)
    for name, tensor in state.items():
        changed = {k: t.clone() for k, t in state.items()}
        raw = changed[name].numpy().reshape(-1).view(np.uint8)  # shares memory, 0-d included
        raw[0] ^= 1  # lowest bit of the first element (little-endian host)
        assert fp(changed) != reference, name


def test_buffers_are_covered(state):
    reference = fp(state)
    counter = next(k for k in state if k.endswith("num_batches_tracked"))
    assert fp({**state, counter: state[counter] + 1}) != reference
    mean = next(k for k in state if k.endswith("running_mean"))
    assert fp({**state, mean: state[mean] + 1e-3}) != reference


def test_name_dtype_and_shape_are_covered():
    state = fixed_state()
    reference = fp(state)
    w = state["features.0.weight"]
    renamed = {("features.0.weightx" if k == "features.0.weight" else k): t for k, t in state.items()}
    assert fp(renamed) != reference
    assert fp({**state, "features.0.weight": w.reshape(3, 2, 2, 2)}) != reference  # same bytes, other shape
    assert fp({**state, "features.0.weight": w.double()}) != reference  # same values, other dtype
    assert fp({**state, "features.0.weight": w.half()}) != reference
    assert fp({**state, "classifier.bias": state["classifier.bias"].to(torch.int32)}) != reference


def test_added_removed_and_swapped_tensors():
    state = fixed_state()
    reference = fp(state)
    assert fp({**state, "extra": torch.zeros(1)}) != reference
    assert fp({k: t for k, t in state.items() if k != "classifier.bias"}) != reference
    a, b = torch.tensor([1.0, 2.0]), torch.tensor([3.0, 4.0])
    assert fp({"x": a, "y": b}) != fp({"x": b, "y": a})


def test_negative_zero_and_nan_payloads_are_distinct():
    assert fp({"w": torch.tensor([0.0])}) != fp({"w": torch.tensor([-0.0])})
    quiet = np.array([0x7FC00000], dtype=np.uint32).view(np.float32)
    other = np.array([0x7FC00001], dtype=np.uint32).view(np.float32)
    assert fp({"w": quiet}) != fp({"w": other})
    assert fp({"w": quiet}) == fp({"w": quiet.copy()})


def test_no_ambiguity_between_names_and_between_tensors():
    one = torch.tensor([1.0])
    assert fp({"a": one, "bc": one}) != fp({"ab": one, "c": one})
    # one tensor of two elements vs two tensors of one element
    assert fp({"a": torch.tensor([1.0, 2.0])}) != fp({"a": torch.tensor([1.0]), "b": torch.tensor([2.0])})
    assert fp({"a": torch.tensor(1.0)}) != fp({"a": torch.tensor([1.0])})  # 0-d vs 1-d


def test_sorted_by_utf8_bytes():
    one, two = torch.tensor([1.0]), torch.tensor([2.0])
    chunks = b"".join(canonical_chunks({"é": one, "z": two, "A": one}))
    assert chunks.index(b"\x00\x01A") < chunks.index(b"\x00\x01z") < chunks.index("é".encode())


def test_different_models_differ():
    set_seed(1)
    a = main_model().state_dict()
    set_seed(2)
    b = main_model().state_dict()
    assert fp(a) != fp(b)


# --- refusals -----------------------------------------------------------------


def test_refuses_what_has_no_canonical_encoding():
    with pytest.raises(ValueError, match="empty"):
        fp({})
    with pytest.raises(TypeError, match="mapping"):
        fp([torch.zeros(1)])
    with pytest.raises(TypeError, match="keys must be str"):
        fp({1: torch.zeros(1)})
    with pytest.raises(TypeError, match="not a tensor"):
        fp({"w": [1.0, 2.0]})
    with pytest.raises(TypeError, match="no canonical encoding"):
        fp({"w": torch.zeros(2, dtype=torch.bfloat16)})
    with pytest.raises(TypeError, match="no canonical encoding"):
        fp({"w": torch.zeros(2, dtype=torch.complex64)})
    with pytest.raises(TypeError, match="no canonical encoding"):
        fp({"w": np.array(["a"])})
    with pytest.raises(TypeError, match="sparse"):
        fp({"w": torch.zeros(2, 2).to_sparse()})


# --- the experiment's helpers -------------------------------------------------


@pytest.fixture
def script(monkeypatch):
    import importlib

    monkeypatch.syspath_prepend(str(REPO_ROOT / "experiments"))
    return importlib.import_module("p5_1_model_fingerprint")


def test_script_helpers(script, state):
    flipped = script.flip_lowest_bit(state)
    differing = [k for k in state if not torch.equal(flipped[k], state[k])]
    assert differing == [next(iter(state))]
    assert int((flipped[differing[0]] != state[differing[0]]).sum()) == 1
    views = script.noncontiguous(state)
    assert all(torch.equal(views[k], state[k]) for k in state)
    assert any(not t.is_contiguous() for t in views.values())
    assert set(script.MODELS) == {"clean_W", "behavioral_only_W_star", "dual_W_star", "zk_model"}


def test_script_check_model_on_a_seeded_model(script, state, tmp_path, monkeypatch):
    path = tmp_path / "w.pt"
    torch.save(state, path)
    monkeypatch.setattr(script, "repo_root", lambda: tmp_path)
    spec = {"task": "test", "file": "w.pt", "file_sha256": script.sha256_file(path), "build": main_model}
    monkeypatch.setattr(script, "fresh_process_fingerprint", lambda p: fingerprint_file(p).sha256)
    row = script.check_model("seeded", spec, tmp_path)
    assert row["fingerprint"]["sha256"] == fp(state)
    assert all(row["stable_routes"].values()) and len(row["stable_routes"]) == 8
    assert all(row["changed_weights_change_fingerprint"].values())
    with pytest.raises(SystemExit, match="Refusing"):
        script.check_model("seeded", {**spec, "file_sha256": "0" * 64}, tmp_path)


def test_committed_result_matches_local_weights(script):
    """Where the real weights are present, they still have the committed fingerprints."""
    from src.utils.results import read_result

    records = sorted((REPO_ROOT / "results").glob("p5.1_model_fingerprint__*.json"))
    if not records:
        pytest.skip("P5.1 has not been run")
    models = read_result(records[-1])["metrics"]["models"]
    checked = 0
    for name, row in models.items():
        path = REPO_ROOT / row["file"]
        if path.exists():
            assert fingerprint_file(path).sha256 == row["fingerprint"]["sha256"], name
            checked += 1
    if not checked:
        pytest.skip("no local weights files")
