"""Checking a revealed secret against a published commitment: the non-ZK baseline (P5.6).

The owner published `C` (P5.4). The plain way to show they know what is behind
it is to **open** it: hand the verifier `K`, `S` and the nonce. The verifier
recomputes the hash and compares. This module is that verifier. It exists so
the zero-knowledge version (P7) has something to be compared against: both
establish "the owner knows an opening of `C`", and they differ in what the
verifier learns.

What the verifier is given
--------------------------
- the publication record, as read from ``provenance/commitment.json``;
- an `Opening`: `K` (32 bytes), `S` (16 bytes) and the nonce (31 bytes).

Nothing else. It reads no file under ``secrets/`` and needs no model.

What it checks
--------------
1. ``publication_well_formed``: the record passes `validate_publication`.
2. ``commitment_matches``: ``Poseidon(DOMAIN, K_hi, K_lo, S, nonce)``,
   recomputed from the opening with the P5.3 layout, equals the published `C`.
3. ``signature_derives_from_key``: `S` is the signature that `K` gives for the
   published owner id (P2.1). The commitment itself binds `S` as an opaque
   value. Only this check ties the owner id in the artifact to the committed
   key, and it is possible only because `K` has been revealed.

``valid`` is True only if all three hold. A mismatch is a verdict, not an
exception. Malformed openings (wrong lengths or types) raise, because they are
not openings at all.

What a valid opening shows, and what it costs
---------------------------------------------
It shows that whoever produced the opening knows a ``(K, S, nonce)`` that
hashes to `C`, and that `S` belongs to the published owner id under that `K`.
Finding a second opening of the same `C` would be a Poseidon collision.

It does **not** show when `C` was published (P5.5), that the opener is the
person who published it, or anything about a model. Whether a suspect model
carries the watermark of this `K` is the P2.8 and P3.7 tests, which need `K`.

The cost is total: **the verifier now holds `K`.** From `K` anyone can derive
every trigger and target (P1.2, P2.2), the projection `P_K` and `S` (P3.1,
P2.1). They can then run the detection tests themselves, but they can also
remove the weight watermark precisely, train the triggers away, or claim the
key as their own in a later dispute over a different model. An opening is a
one-time act, and after it the key should be treated as burned. A ZK proof of
the same statement reveals none of the 79 secret bytes. That difference is the
reason for P7, and P9.6 reports it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from src.crypto.commitment import NONCE_BYTES, Commitment, commit, commitment_inputs
from src.crypto.publication import validate_publication
from src.watermark.keygen import KEY_BYTES
from src.watermark.signature import SIGNATURE_BYTES, derive_signature

OPENING_SCHEMA = "zk-crown/commitment-opening/v1"
REVEALED_BYTES = KEY_BYTES + SIGNATURE_BYTES + NONCE_BYTES
"""Secret bytes a non-ZK opening hands to the verifier: 32 + 16 + 31 = 79."""


@dataclass(frozen=True, repr=False)
class Opening:
    """A revealed ``(K, S, nonce)``. **Secret until deliberately shown to a verifier.**"""

    key: bytes
    signature: bytes
    nonce: bytes

    def __post_init__(self) -> None:
        commitment_inputs(self.key, self.signature, self.nonce)  # raises on wrong types or lengths

    def __repr__(self) -> str:
        return "Opening(<hidden>)"

    def to_dict(self) -> dict[str, str]:
        """The opening as hex, for handing to a verifier. Writing this anywhere reveals `K`."""
        return {"schema": OPENING_SCHEMA, "key_hex": self.key.hex(), "signature_hex": self.signature.hex(),
                "nonce_hex": self.nonce.hex()}

    @classmethod
    def from_dict(cls, data: Any) -> "Opening":
        if not isinstance(data, dict) or set(data) != {"schema", "key_hex", "signature_hex", "nonce_hex"}:
            raise ValueError("an opening holds schema, key_hex, signature_hex and nonce_hex")
        if data["schema"] != OPENING_SCHEMA:
            raise ValueError(f"unknown opening schema {data['schema']!r}")
        try:
            parts = [bytes.fromhex(data[name]) for name in ("key_hex", "signature_hex", "nonce_hex")]
        except (TypeError, ValueError) as error:
            raise ValueError("opening fields must be hex strings") from error
        return cls(*parts)


@dataclass(frozen=True)
class OpeningVerdict:
    """The outcome of checking one opening against one publication. Holds no secret."""

    publication_well_formed: bool
    commitment_matches: bool
    signature_derives_from_key: bool
    published_commitment: str | None
    owner_id: str | None
    problem: str | None = None

    @property
    def valid(self) -> bool:
        return self.publication_well_formed and self.commitment_matches and self.signature_derives_from_key

    def to_dict(self) -> dict[str, Any]:
        return {
            "valid": self.valid,
            "publication_well_formed": self.publication_well_formed,
            "commitment_matches": self.commitment_matches,
            "signature_derives_from_key": self.signature_derives_from_key,
            "published_commitment": self.published_commitment,
            "owner_id": self.owner_id,
            "problem": self.problem,
            "secret_bytes_revealed_to_verifier": REVEALED_BYTES,
        }


def verify_opening(publication: Any, opening: Opening) -> OpeningVerdict:
    """Check `opening` against the published record. Never raises on a mismatch."""
    if not isinstance(opening, Opening):
        raise TypeError("opening must be an Opening")
    try:
        validate_publication(publication)
    except (ValueError, TypeError) as error:
        return OpeningVerdict(False, False, False, None, None, problem=f"publication is malformed: {error}")

    published = Commitment.from_decimal(publication["commitment"]["decimal"])
    owner_id = publication["owner_id"]
    matches = commit(opening.key, opening.signature, opening.nonce) == published
    derives = derive_signature(opening.key, owner_id).value == opening.signature
    problem = None
    if not matches:
        problem = "the opening does not hash to the published commitment"
    elif not derives:
        problem = "S is not the signature this K gives for the published owner id"
    return OpeningVerdict(True, matches, derives, published.decimal(), owner_id, problem=problem)
