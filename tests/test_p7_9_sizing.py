"""P7.9: the sizing arithmetic for in-circuit trigger derivation."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "experiments"))
try:
    import p7_9_sizing_estimate as est  # noqa: E402
finally:
    sys.path.remove(str(REPO_ROOT / "experiments"))


def test_sha256_padding_block_counts():
    assert [est.compressions(n) for n in (0, 55, 56, 64, 119, 120)] == [1, 1, 2, 2, 2, 3]


def test_hmac_block_costs_follow_v1_labels():
    # 22-byte keystream domain + u16 + label + u64 counter, then 1 outer compression
    assert est.hmac_block_compressions("triggers/v1/base-index") == 2      # 54-byte inner message
    assert est.hmac_block_compressions("triggers/v1/perturbation-sign") == 3  # 61 bytes
    assert est.hmac_block_compressions("responses/v1/target-class") == 3   # 57 bytes


def test_bundle_digest_covers_all_pixels():
    assert est.bundle_digest_bytes() > est.N * est.PIXELS + 3 * est.N * 8


def test_estimate_arithmetic_with_fixed_counts():
    c = {"sha256_compression": 30000, "poseidon_2": 500, "poseidon_16": 2000, "num2bits_8": 9, "lessthan_10": 14}
    e = est.estimate(c)
    assert e["compressions"]["perturbation_sign"] == 1200 * 3
    assert e["parts"]["keystream_sha256"] == (2 + 3600 + 50 + 75) * 30000
    assert e["total_estimate"] == sum(e["parts"].values())
    assert not e["fits_2_15"]
