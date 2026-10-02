"""Tests for P5.3: the commitment `C = Poseidon(DOMAIN, K_hi, K_lo, S, nonce)`.

Stdlib only. All keys, signatures and nonces here are test or public demo
values, never the owner's.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from src.crypto import commitment as module
from src.crypto.commitment import (
    COMMITMENT_VERSION,
    DOMAIN_ELEMENT,
    INPUT_BITS,
    INPUT_NAMES,
    NONCE_BYTES,
    Commitment,
    commit,
    commitment_inputs,
    new_nonce,
)
from src.crypto.poseidon import BN254_SCALAR_FIELD as P
from src.crypto.poseidon import poseidon
from src.watermark.signature import derive_signature

REPO_ROOT = Path(__file__).resolve().parents[1]
VECTOR = json.loads((REPO_ROOT / "tests" / "data" / "commitment_circomlibjs_vector.json").read_text(encoding="utf-8"))

KEY = bytes(range(32))
SIG = bytes(range(100, 116))
NONCE = bytes(range(200, 231))


# --- the layout ---------------------------------------------------------------


def test_matches_the_circomlibjs_vector():
    """JavaScript re-did the byte slicing from the specification; both must agree."""
    key, sig, nonce = (bytes.fromhex(VECTOR[k]) for k in ("key_hex", "signature_hex", "nonce_hex"))
    assert VECTOR["commitment_version"] == COMMITMENT_VERSION and VECTOR["input_names"] == list(INPUT_NAMES)
    assert [str(x) for x in commitment_inputs(key, sig, nonce)] == VECTOR["inputs"]
    c = commit(key, sig, nonce)
    assert c.decimal() == VECTOR["commitment_decimal"] and c.hex() == VECTOR["commitment_hex"]


def test_vector_uses_public_demo_values_only():
    phrases = VECTOR["phrases"]
    assert "DEMO" in VECTOR["warning"] and all("DEMO" in p for p in phrases.values())
    assert hashlib.sha256(phrases["key"].encode()).hexdigest() == VECTOR["key_hex"]
    assert hashlib.sha256(phrases["signature"].encode()).digest()[:16].hex() == VECTOR["signature_hex"]
    assert hashlib.sha256(phrases["nonce"].encode()).digest()[:31].hex() == VECTOR["nonce_hex"]


def test_field_layout_written_out_by_hand():
    inputs = commitment_inputs(KEY, SIG, NONCE)
    assert inputs == (
        0x7A6B2D63726F776E2F636F6D6D69746D656E742F7631,
        0x000102030405060708090A0B0C0D0E0F,
        0x101112131415161718191A1B1C1D1E1F,
        int.from_bytes(SIG, "big"),
        int.from_bytes(NONCE, "big"),
    )
    assert commit(KEY, SIG, NONCE).value == poseidon(list(inputs))
    assert DOMAIN_ELEMENT == int.from_bytes(b"zk-crown/commitment/v1", "big")
    assert INPUT_NAMES == ("DOMAIN", "K_hi", "K_lo", "S", "nonce") and INPUT_BITS == (175, 128, 128, 128, 248)


def test_pinned_commitment():
    """A change to the layout must be deliberate: it changes this value and needs a new version."""
    assert commit(KEY, SIG, NONCE).decimal() == PINNED


PINNED = "3495897459136643063190850786010973179789732169561139198693747600224527128823"


def test_every_input_fits_the_field_at_its_maximum():
    inputs = commitment_inputs(b"\xff" * 32, b"\xff" * 16, b"\xff" * 31)
    assert inputs[1:] == (2**128 - 1, 2**128 - 1, 2**128 - 1, 2**248 - 1)
    assert all(0 <= x < P for x in inputs)
    assert all(x.bit_length() <= bits for x, bits in zip(inputs, INPUT_BITS))
    assert 0 <= commit(b"\xff" * 32, b"\xff" * 16, b"\xff" * 31).value < P
    assert commitment_inputs(bytes(32), bytes(16), bytes(31))[1:] == (0, 0, 0, 0)


def test_inputs_are_one_to_one_with_the_opening():
    """The limbs reassemble to exactly K, S and the nonce: nothing is reduced or dropped."""
    key = hashlib.sha256(b"k").digest()
    _, hi, lo, s, n = commitment_inputs(key, SIG, NONCE)
    assert hi.to_bytes(16, "big") + lo.to_bytes(16, "big") == key
    assert s.to_bytes(16, "big") == SIG and n.to_bytes(31, "big") == NONCE


def test_accepts_an_ownership_signature_object():
    signature = derive_signature(KEY, "test-owner")
    assert commit(KEY, signature, NONCE) == commit(KEY, signature.value, NONCE)
    assert commitment_inputs(KEY, signature, NONCE)[3] == signature.as_int
    assert commit(KEY, signature, NONCE) != commit(KEY, derive_signature(KEY, "other-owner"), NONCE)


# --- binding and hiding, as far as a test can see -----------------------------


def test_any_single_bit_of_the_opening_changes_the_commitment():
    base = commit(KEY, SIG, NONCE)
    seen = {base.value}
    for which, blob in (("key", KEY), ("sig", SIG), ("nonce", NONCE)):
        for bit in range(len(blob) * 8):
            flipped = bytearray(blob)
            flipped[bit // 8] ^= 0x80 >> (bit % 8)
            parts = {"key": KEY, "sig": SIG, "nonce": NONCE, which: bytes(flipped)}
            seen.add(commit(parts["key"], parts["sig"], parts["nonce"]).value)
    assert len(seen) == 1 + 256 + 128 + 248


def test_limbs_and_fields_are_not_interchangeable():
    hi, lo = KEY[:16], KEY[16:]
    assert commit(lo + hi, SIG, NONCE) != commit(KEY, SIG, NONCE)  # swapped limbs
    assert commit(hi + SIG, lo, NONCE) != commit(KEY, SIG, NONCE)  # K_lo and S swapped
    # the domain element separates it from a plain four-input hash of the same values
    assert commit(KEY, SIG, NONCE).value != poseidon(list(commitment_inputs(KEY, SIG, NONCE)[1:]))


def test_nonce_makes_commitments_unlinkable_and_is_never_drawn_inside_commit():
    assert commit(KEY, SIG, NONCE) == commit(KEY, SIG, NONCE)  # deterministic given the nonce
    values = {commit(KEY, SIG, new_nonce()).value for _ in range(8)}
    assert len(values) == 8


def test_new_nonce():
    nonces = {new_nonce() for _ in range(64)}
    assert len(nonces) == 64 and all(isinstance(n, bytes) and len(n) == NONCE_BYTES == 31 for n in nonces)


def test_same_in_a_fresh_process_with_another_hash_seed():
    code = ("from src.crypto.commitment import commit; "
            "print(commit(bytes(range(32)), bytes(range(100, 116)), bytes(range(200, 231))).decimal())")
    env = {**os.environ, "PYTHONHASHSEED": "777"}
    done = subprocess.run([sys.executable, "-c", code], cwd=REPO_ROOT, env=env, capture_output=True, text=True,
                          check=True)
    assert done.stdout.strip() == PINNED


# --- refusals -----------------------------------------------------------------


@pytest.mark.parametrize("key, sig, nonce, error", [
    (KEY[:31], SIG, NONCE, ValueError), (KEY + b"\0", SIG, NONCE, ValueError),
    (KEY, SIG[:15], NONCE, ValueError), (KEY, SIG + b"\0", NONCE, ValueError),
    (KEY, SIG, NONCE[:30], ValueError), (KEY, SIG, NONCE + b"\0", ValueError),
    (KEY.hex(), SIG, NONCE, TypeError), (KEY, int.from_bytes(SIG, "big"), NONCE, TypeError),
    (KEY, SIG, 5, TypeError), (None, SIG, NONCE, TypeError),
])
def test_refuses_wrong_lengths_and_types(key, sig, nonce, error):
    with pytest.raises(error):
        commit(key, sig, nonce)


def test_no_padding_or_truncation_of_short_values():
    """A short nonce is refused, not left-padded: b'\\x01' and 31 bytes ending in 1 must not collide silently."""
    with pytest.raises(ValueError, match="exactly 31 bytes"):
        commitment_inputs(KEY, SIG, b"\x01")
    assert commitment_inputs(bytearray(KEY), bytearray(SIG), bytearray(NONCE)) == commitment_inputs(KEY, SIG, NONCE)


# --- the published value ------------------------------------------------------


def test_commitment_encodings_round_trip():
    c = commit(KEY, SIG, NONCE)
    assert Commitment.from_decimal(c.decimal()) == c and Commitment.from_bytes(c.to_bytes()) == c
    assert len(c.to_bytes()) == 32 and int(c.hex(), 16) == c.value == int(c.decimal())
    assert c.to_dict() == {"version": COMMITMENT_VERSION, "decimal": c.decimal(), "hex": c.hex()}
    assert Commitment(0).decimal() == "0" and Commitment.from_decimal("0").value == 0


def test_commitment_refuses_non_canonical_values():
    for bad in (P, P + 1, -1):
        with pytest.raises(ValueError):
            Commitment(bad)
    for bad in (1.0, "1", True):
        with pytest.raises(TypeError):
            Commitment(bad)
    for bad in ("01", "", "+1", "-1", "1 ", "0x10", "１２", str(P), 12):
        with pytest.raises(ValueError):
            Commitment.from_decimal(bad)
    with pytest.raises(ValueError):
        Commitment.from_bytes(b"\xff" * 32)  # >= p
    with pytest.raises(ValueError):
        Commitment.from_bytes(b"\x01" * 31)
    with pytest.raises(ValueError):
        Commitment(1, version="zk-crown/commitment/v0")


def test_module_has_no_secret_defaults():
    """Nothing in the module holds a key or nonce: every secret is an argument."""
    assert not [n for n in vars(module) if n.lower() in {"key", "nonce", "k", "secret"}]
