"""Tests for the ownership signature `S` (P2.1).

Every key and owner id here is TEST data, public by construction.
"""

from __future__ import annotations

import hashlib
import hmac
import os
import subprocess
import sys
from pathlib import Path

import pytest

from src.watermark.keygen import DOMAIN, KeyStream
from src.watermark.signature import (
    SIGNATURE_BITS,
    SIGNATURE_BYTES,
    OwnershipSignature,
    check_owner_id,
    derive_signature,
    signature_label,
)
from src.watermark.triggers import BASE_INDEX_LABEL, PERTURBATION_SIGN_LABEL

TEST_KEY = bytes(range(32))
"""TEST KEY ONLY."""

TEST_OWNER = "zk-crown test owner"

KAT_S_HEX = "45a5e6685192dc95decafc229f80e773"
"""S for TEST_KEY and TEST_OWNER. Cross-checked outside Python with
``openssl dgst -sha256 -mac HMAC -macopt hexkey:<TEST_KEY>`` over the documented
message; S is the first 16 bytes of that MAC."""

BN254_SCALAR_FIELD = 21888242871839275222246405745257275088548364400416034343698204186575808495617
REPO_ROOT = Path(__file__).resolve().parents[1]


# --- derivation -------------------------------------------------------------


def test_known_answer_vector():
    assert derive_signature(TEST_KEY, TEST_OWNER).hex() == KAT_S_HEX


def test_matches_the_documented_formula():
    label = f"signature/v1/owner:{TEST_OWNER}".encode()
    message = DOMAIN + len(label).to_bytes(2, "big") + label + (0).to_bytes(8, "big")
    expected = hmac.new(TEST_KEY, message, hashlib.sha256).digest()[:16]
    assert derive_signature(TEST_KEY, TEST_OWNER).value == expected


def test_is_deterministic():
    a = derive_signature(TEST_KEY, TEST_OWNER)
    b = derive_signature(bytearray(TEST_KEY), TEST_OWNER)
    assert a == b and a.value == b.value


@pytest.mark.parametrize("hashseed", ["0", "777"])
def test_identical_in_a_fresh_process(hashseed):
    code = (
        "from src.watermark.signature import derive_signature;"
        f"print(derive_signature(bytes(range(32)), {TEST_OWNER!r}).hex())"
    )
    out = subprocess.run(
        [sys.executable, "-c", code],
        cwd=REPO_ROOT,
        env={**os.environ, "PYTHONHASHSEED": hashseed},
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    assert out == KAT_S_HEX


def test_different_key_gives_different_signature():
    flipped = bytearray(TEST_KEY)
    flipped[0] ^= 0x80
    assert derive_signature(bytes(flipped), TEST_OWNER).value != derive_signature(TEST_KEY, TEST_OWNER).value


def test_different_owner_gives_different_signature():
    owners = [TEST_OWNER, TEST_OWNER + "2", "Zk-crown test owner", "other"]
    values = {derive_signature(TEST_KEY, o).value for o in owners}
    assert len(values) == len(owners)


def test_is_not_a_plain_encoding_of_the_owner():
    s = derive_signature(TEST_KEY, TEST_OWNER)
    encoded = TEST_OWNER.encode()
    assert s.value != encoded[:SIGNATURE_BYTES]
    assert s.value != hashlib.sha256(encoded).digest()[:SIGNATURE_BYTES]


def test_single_bit_key_flips_change_about_half_the_bits():
    """Over all 256 single-bit key flips, pooled 32,768 bits: expect half to change."""
    reference = derive_signature(TEST_KEY, TEST_OWNER).as_int
    changed = 0
    for bit in range(256):
        key = bytearray(TEST_KEY)
        key[bit // 8] ^= 1 << (bit % 8)
        changed += bin(reference ^ derive_signature(bytes(key), TEST_OWNER).as_int).count("1")
    total = 256 * SIGNATURE_BITS
    # Standard deviation is sqrt(total) / 2, about 90.5. The threshold is 6 sd.
    assert abs(changed - total / 2) < 6 * (total**0.5) / 2, changed


def test_shares_no_bytes_with_the_trigger_streams():
    s = derive_signature(TEST_KEY, TEST_OWNER).value
    for label in (BASE_INDEX_LABEL, PERTURBATION_SIGN_LABEL):
        assert s not in KeyStream(TEST_KEY, label).read(4_096)


def test_owner_id_cannot_forge_another_namespace():
    """An owner id cannot turn the signature label into a trigger label."""
    assert not signature_label("x/../../triggers/v1").startswith("triggers/")
    assert signature_label(TEST_OWNER).startswith("signature/v1/owner:")


# --- representations --------------------------------------------------------


def test_bits_signs_and_int_agree():
    s = derive_signature(TEST_KEY, TEST_OWNER)
    assert len(s.bits) == len(s.signs) == SIGNATURE_BITS
    assert int("".join(map(str, s.bits)), 2) == s.as_int
    assert all((b == 1) == (x == 1) and x in (-1, 1) for b, x in zip(s.bits, s.signs))
    assert s.as_int.to_bytes(SIGNATURE_BYTES, "big") == s.value


def test_bits_are_msb_first():
    s = OwnershipSignature(owner_id="o", value=bytes([0x80]) + bytes(15))
    assert s.bits[0] == 1 and sum(s.bits) == 1
    assert s.signs[0] == 1 and s.signs[1] == -1


def test_fits_in_one_bn254_field_element():
    assert (1 << SIGNATURE_BITS) - 1 < BN254_SCALAR_FIELD
    assert derive_signature(TEST_KEY, TEST_OWNER).as_int < BN254_SCALAR_FIELD


def test_repr_hides_the_value():
    s = derive_signature(TEST_KEY, TEST_OWNER)
    assert KAT_S_HEX not in repr(s) and KAT_S_HEX not in str(s)


# --- validation -------------------------------------------------------------


@pytest.mark.parametrize(
    "owner, exc",
    [
        ("", ValueError),
        (" owner", ValueError),
        ("owner\n", ValueError),
        ("e\u0301", ValueError),  # decomposed e-acute, not NFC
        ("x" * 257, ValueError),
        (b"owner", TypeError),
        (None, TypeError),
    ],
)
def test_bad_owner_ids_are_rejected(owner, exc):
    with pytest.raises(exc):
        check_owner_id(owner)
    with pytest.raises(exc):
        derive_signature(TEST_KEY, owner)


def test_nfc_and_non_ascii_owner_ids_are_accepted():
    assert check_owner_id("\u00e9quipe z\u00fcrich") == "\u00e9quipe z\u00fcrich"
    assert check_owner_id("x" * 256)


def test_bad_key_is_rejected():
    with pytest.raises(ValueError):
        derive_signature(bytes(31), TEST_OWNER)


@pytest.mark.parametrize("value", [bytes(15), bytes(17), "00" * 16])
def test_signature_value_must_be_16_bytes(value):
    with pytest.raises(ValueError):
        OwnershipSignature(owner_id=TEST_OWNER, value=value)
