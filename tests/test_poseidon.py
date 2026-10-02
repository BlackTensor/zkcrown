"""Tests for P5.2: Poseidon over BN254, checked against circomlibjs and published vectors.

Stdlib only. The circomlibjs vectors in ``tests/data`` were produced by
``experiments/p5_2_make_poseidon_vectors.mjs`` with Node, not by Python.
"""

from __future__ import annotations

import importlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from src.crypto import poseidon as module
from src.crypto.poseidon import (
    BN254_SCALAR_FIELD as P,
    FULL_ROUNDS,
    MAX_INPUTS,
    PARTIAL_ROUNDS,
    grain_parameters,
    partial_rounds,
    poseidon,
    poseidon_permutation,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
VECTORS = json.loads((REPO_ROOT / "tests" / "data" / "poseidon_circomlibjs_vectors.json").read_text(encoding="utf-8"))


@pytest.fixture
def script(monkeypatch):
    monkeypatch.syspath_prepend(str(REPO_ROOT / "experiments"))
    return importlib.import_module("p5_2_validate_poseidon")


# --- known vectors ------------------------------------------------------------


def test_reference_permutation_vector_t3():
    """The Poseidon authors' test vector for the x^5, 254-bit, t = 3 permutation."""
    assert poseidon_permutation([0, 1, 2]) == [
        0x115CC0F5E7D690413DF64C6B9662E9CF2A3617F2743245519E19607A4417189A,
        0x0FCA49B798923AB0239DE1C9E7A4A9A2210312B6A2F616D18B5A87F9B628AE29,
        0x0E7AE82E40091E63CBD4F16A6D16310B3729D4B6E138FCF54110E2867045A30C,
    ]


@pytest.mark.parametrize("inputs, expected", [
    ([1, 2], 7853200120776062878684798364095072458815029376092732009249414926327459813530),
    ([1, 2, 3, 4], 18821383157269793795438455681495246036402687001665670618754263018637548127333),
    (list(range(1, 17)), 9989051620750914585850546081941653841776809718687451684622678807385399211877),
])
def test_values_asserted_by_circomlibjs_own_tests(inputs, expected):
    assert poseidon(inputs) == expected


def test_vector_file_covers_every_input_count():
    assert VECTORS["source"] == "circomlibjs" and int(VECTORS["field"]) == P
    assert VECTORS["implementations_agreeing"] == ["buildPoseidon", "buildPoseidonReference"]
    counts = {n: sum(v["n"] == n for v in VECTORS["vectors"]) for n in range(1, MAX_INPUTS + 1)}
    assert all(count == 8 for count in counts.values()) and len(VECTORS["vectors"]) == 128
    assert all(len(v["inputs"]) == v["n"] for v in VECTORS["vectors"])


@pytest.mark.parametrize("n", range(1, MAX_INPUTS + 1))
def test_matches_circomlibjs_vectors(n):
    vectors = [v for v in VECTORS["vectors"] if v["n"] == n]
    for vector in vectors:
        assert poseidon([int(x) for x in vector["inputs"]]) == int(vector["output"]), vector["kind"]


@pytest.mark.parametrize("t", range(2, MAX_INPUTS + 2))
def test_generated_constants_equal_circomlibjs_tables(script, t):
    assert script.constant_digests(t) == VECTORS["constant_digests"][str(t)]


def test_vector_inputs_follow_the_stated_rule(script):
    for vector in VECTORS["vectors"]:
        assert [int(x) for x in vector["inputs"]] == script.expected_inputs(VECTORS, vector["n"], vector["kind"])


# --- the instance -------------------------------------------------------------


def test_instance_parameters():
    assert P == 21888242871839275222246405745257275088548364400416034343698204186575808495617
    assert P.bit_length() == 254 and FULL_ROUNDS == 8 and MAX_INPUTS == 16
    assert PARTIAL_ROUNDS == (56, 57, 56, 60, 60, 63, 64, 63, 60, 66, 60, 65, 70, 60, 64, 68)
    assert partial_rounds(2) == 56 and partial_rounds(4) == 56 and partial_rounds(17) == 68
    assert (P - 1) % 5 != 0  # x^5 is a permutation of the field
    for bad in (1, 18, 0, -1, 3.0, True, "3"):
        with pytest.raises(ValueError):
            partial_rounds(bad)


@pytest.mark.parametrize("t", [2, 3, 4, 17])
def test_generated_parameters_are_well_formed(t):
    constants, mds = grain_parameters(t)
    assert len(constants) == (FULL_ROUNDS + partial_rounds(t)) * t
    assert all(0 <= c < P for c in constants)
    assert len(mds) == t and all(len(row) == t for row in mds)
    assert all(0 < m < P for row in mds for m in row)
    assert grain_parameters(t) is grain_parameters(t)  # cached, and immutable tuples


def test_hash_is_the_first_element_of_the_permutation_with_zero_capacity():
    assert poseidon([7, 8, 9]) == poseidon_permutation([0, 7, 8, 9])[0]
    assert poseidon((7, 8, 9)) == poseidon([7, 8, 9])  # tuple or list


# --- behaviour a commitment relies on -----------------------------------------


def test_sensitive_to_order_value_and_length():
    base = poseidon([1, 2, 3])
    assert poseidon([3, 2, 1]) != base and poseidon([2, 1, 3]) != base
    assert poseidon([1, 2, 4]) != base
    assert poseidon([1, 2, 3, 0]) != base  # a trailing zero is a different hash, not padding
    assert poseidon([0]) != poseidon([0, 0])
    assert len({poseidon([i, 2, 3]) for i in range(50)}) == 50


def test_output_is_a_field_element_and_deterministic():
    out = poseidon([P - 1, 0, 12345])
    assert isinstance(out, int) and 0 <= out < P
    assert poseidon([P - 1, 0, 12345]) == out


def test_does_not_modify_its_input():
    inputs = [1, 2, 3]
    poseidon(inputs)
    state = [0, 1, 2]
    poseidon_permutation(state)
    assert inputs == [1, 2, 3] and state == [0, 1, 2]


def test_refuses_out_of_field_and_non_integer_inputs():
    """No silent reduction: p and 0 must not hash alike."""
    for bad in ([P], [-1], [1, P + 5], [2**256]):
        with pytest.raises(ValueError, match="outside the field"):
            poseidon(bad)
    for bad in ([1.0], ["1"], [True], [None], [b"\x01"]):
        with pytest.raises(TypeError, match="must be an int"):
            poseidon(bad)
    with pytest.raises(ValueError, match="1 to 16 inputs"):
        poseidon([])
    with pytest.raises(ValueError, match="1 to 16 inputs"):
        poseidon([1] * 17)
    with pytest.raises(TypeError, match="list or tuple"):
        poseidon(5)
    with pytest.raises(TypeError, match="list or tuple"):
        poseidon(b"abc")


def test_same_in_a_fresh_process_with_another_hash_seed():
    code = "from src.crypto.poseidon import poseidon; print(poseidon([1, 2]))"
    env = {**os.environ, "PYTHONHASHSEED": "31337"}
    done = subprocess.run([sys.executable, "-c", code], cwd=REPO_ROOT, env=env, capture_output=True, text=True,
                          check=True)
    assert int(done.stdout) == 7853200120776062878684798364095072458815029376092732009249414926327459813530


# --- the checks can fail ------------------------------------------------------


def test_a_transposed_mds_matrix_is_caught(monkeypatch):
    real = grain_parameters

    def transposed(t):
        constants, mds = real(t)
        return constants, tuple(zip(*mds))

    monkeypatch.setattr(module, "grain_parameters", transposed)
    assert poseidon([1, 2]) != 7853200120776062878684798364095072458815029376092732009249414926327459813530


def test_a_wrong_round_count_is_caught(monkeypatch):
    grain_parameters.cache_clear()
    monkeypatch.setattr(module, "PARTIAL_ROUNDS", (58,) * 16)  # t = 3 really has 57
    try:
        assert poseidon([1, 2]) != 7853200120776062878684798364095072458815029376092732009249414926327459813530
    finally:
        grain_parameters.cache_clear()


def test_validation_stops_on_a_wrong_vector(script):
    assert script.validate(VECTORS)["mismatches"] == 0
    broken = json.loads(json.dumps(VECTORS))
    broken["vectors"][10]["output"] = str((int(broken["vectors"][10]["output"]) + 1) % P)
    with pytest.raises(SystemExit, match="differs from circomlibjs"):
        script.validate(broken)
    broken = json.loads(json.dumps(VECTORS))
    broken["vectors"][3]["inputs"][0] = "5"
    with pytest.raises(SystemExit, match="generator's rule"):
        script.validate(broken)
    broken = json.loads(json.dumps(VECTORS))
    broken["constant_digests"]["4"]["M_sha256"] = "0" * 64
    with pytest.raises(SystemExit, match="width 4"):
        script.validate(broken)
