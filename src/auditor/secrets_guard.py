"""The auditor's no-secrets guard (P9.1).

A verdict is meant to be shown to people who must not learn the owner's
secrets. The rule "a check reports aggregates only" is therefore enforced in
code, in two layers, on everything a check returns:

1. **Structural.** A check's ``statistic`` is a flat mapping from short
   snake_case names to scalars: ``None``, ``bool``, ``int`` with
   ``|x| <= 2^53``, a finite ``float``, or a ``str`` of at most 120
   characters. Lists, arrays, tensors, bytes and nested mappings are refused,
   so a trigger image, a target list, the fingerprint ``y`` or a sign vector
   cannot be returned at all. Integers above ``2^53`` are refused, so `K`'s
   limbs, `S` and the nonce (128 to 248 bits) cannot be returned as numbers.
   Any string, in the statistic or in the check's ``reason``, is refused if it
   holds a run of 16 or more hexadecimal characters, which covers every hex or
   decimal encoding of those values.
2. **Value scan.** When the owner's secrets are supplied to the audit, every
   string a check returns is also searched, case-insensitively, for `K`, its
   two halves, `S` and the nonce in hex, decimal (as the P5.3 field elements
   and as whole integers) and base64. This catches encodings the structural
   rule lets through, such as base64.

A check whose output breaks either rule is not reported with that output. Its
result is replaced by status ``error`` and a reason naming the rule, never the
offending value. Finally the engine scans the whole built verdict with the
same needles and refuses to return it if any appears, which also covers what
the caller passed in (a suspect label, for example).

What this does not catch: a single small number that happens to be secret,
such as one target class (0 to 9) or one bit of `S`. Those cannot be told
apart from a count by their form. Nor does the value scan catch a fragment of
a secret in an encoding other than hex or decimal (a truncated base64 string,
for example), or any encoding it does not list. Checks are owner code; the guard stops
whole secrets and bulk secret material from leaving through a verdict, and
the checks themselves must still report aggregates only.
"""

from __future__ import annotations

import base64
import math
import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from src.crypto.commitment import NONCE_BYTES, commitment_inputs
from src.watermark.keygen import KEY_BYTES
from src.watermark.signature import SIGNATURE_BYTES

MAX_SAFE_INT = 2**53
MAX_STATISTIC_STRING = 120
MAX_REASON = 1000
MAX_STATISTIC_ENTRIES = 64
HEX_RUN = re.compile(r"[0-9a-fA-F]{16,}")
STATISTIC_NAME = re.compile(r"^[a-z][a-z0-9_]{0,63}$")
SECRET_NAMES = ("K", "S", "nonce")


class SecretLeak(ValueError):
    """A check's output, or a built verdict, broke the no-secrets rule.

    The message names the rule only, never the value.
    """


@dataclass(frozen=True)
class OwnerSecrets:
    """The owner's secrets for one audit: `K`, `S` and the commitment nonce.

    Held so checks that need them can get them, and so the guard knows what to
    look for. ``repr`` hides every value.
    """

    key: bytes
    signature: bytes
    nonce: bytes

    def __post_init__(self) -> None:
        for name, value, length in (("key", self.key, KEY_BYTES), ("signature", self.signature, SIGNATURE_BYTES),
                                    ("nonce", self.nonce, NONCE_BYTES)):
            if not isinstance(value, bytes) or len(value) != length:
                raise ValueError(f"{name} must be exactly {length} bytes")

    def __repr__(self) -> str:
        return "OwnerSecrets(key=<hidden>, signature=<hidden>, nonce=<hidden>)"

    def get(self, name: str) -> bytes:
        return {"K": self.key, "S": self.signature, "nonce": self.nonce}[name]

    def needles(self) -> tuple[str, ...]:
        """Every encoding of a secret the guard searches for, lower case."""
        k_hi, k_lo, s, nonce = commitment_inputs(self.key, self.signature, self.nonce)[1:]
        values: list[str] = []
        for element in (k_hi, k_lo, s, nonce, int.from_bytes(self.key, "big")):
            values += [str(element), format(element, "x")]
        for raw in (self.key, self.key[:16], self.key[16:], self.signature, self.nonce):
            values += [raw.hex(), base64.b64encode(raw).decode().rstrip("=")]
        return tuple(sorted({v.lower() for v in values}))


def check_text(where: str, text: str, limit: int) -> None:
    """Structural rule for one string a check returns."""
    if len(text) > limit:
        raise SecretLeak(f"{where} is longer than {limit} characters")
    if HEX_RUN.search(text):
        raise SecretLeak(f"{where} holds a run of 16 or more hexadecimal characters")


def check_statistic(statistic: Any) -> dict[str, Any]:
    """Structural rule for a check's statistic. Returns a plain dict copy."""
    if not isinstance(statistic, Mapping):
        raise SecretLeak("statistic must be a mapping")
    if len(statistic) > MAX_STATISTIC_ENTRIES:
        raise SecretLeak(f"statistic has more than {MAX_STATISTIC_ENTRIES} entries")
    clean: dict[str, Any] = {}
    for name, value in statistic.items():
        if not isinstance(name, str) or not STATISTIC_NAME.match(name):
            raise SecretLeak("statistic names must be short snake_case strings")
        where = f"statistic[{name!r}]"
        if value is None or isinstance(value, bool):
            pass
        elif isinstance(value, int):
            if abs(value) > MAX_SAFE_INT:
                raise SecretLeak(f"{where} is an integer above 2^53")
        elif isinstance(value, float):
            if not math.isfinite(value):
                raise SecretLeak(f"{where} is not finite")
        elif isinstance(value, str):
            check_text(where, value, MAX_STATISTIC_STRING)
        else:
            raise SecretLeak(f"{where} is a {type(value).__name__}, not a scalar")
        clean[name] = value
    return clean


def find_needle(text: str, needles: tuple[str, ...]) -> bool:
    lowered = text.lower()
    return any(needle in lowered for needle in needles)
