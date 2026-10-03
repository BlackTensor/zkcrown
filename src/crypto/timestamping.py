"""Independent timestamps for the commitment publication artifact (P5.5).

``created_utc`` inside ``provenance/commitment.json`` is the owner's own clock.
"I committed before the dispute" needs evidence the owner could not have made
up afterwards. Two mechanisms are used, and they are not the same kind of
evidence:

**OpenTimestamps** (the time evidence). The artifact's SHA-256 is sent to
public calendar servers, which aggregate many digests into one Bitcoin
transaction. The resulting proof, ``provenance/commitment.json.ots``, is a
chain of hash operations from the file's digest to a Bitcoin block header.
Once the block is mined, the proof shows the file existed before that block,
and anyone can check it without trusting the owner or the calendars.

- Only the hash leaves the machine. The artifact holds nothing secret anyway.
- A fresh proof is **pending**: it holds the calendars' promises, not a
  Bitcoin attestation. That takes a few hours. `upgrade` then fetches the
  rest of the chain. Until then the proof shows only that calendars received
  the digest, on their word.
- The time proved is the block's, so it is a few hours later than the stamp
  and accurate to within Bitcoin's block-time tolerance, roughly hours.

**A GPG-signed git tag** (who, not when). The tag binds the commit holding
the artifact to the owner's signing key. The date inside a GPG signature is
the signer's own clock, so the tag alone is **not** independent evidence of
time. It shows that the holder of that key vouched for exactly this commit.
It becomes time evidence only through a third party: a host that records when
the tag was pushed, or an OpenTimestamps proof.

A Bitcoin attestation is checked by `check_bitcoin_attestations`: the
attested block's header is fetched from two public explorers, hashed here,
checked for proof of work, and compared with the Merkle root the proof
requires. That trusts the explorers' view of which block sits at a height
(two must agree); it is not a full node.

The ``ots`` command line client does not start on this Windows machine: its
`python-bitcoinlib` dependency fails to load OpenSSL at import. The
`opentimestamps` library underneath works, so the stamp, upgrade and
inspection steps are done with it here. The proof file format is the standard
one, and any OpenTimestamps client can verify it.
"""

from __future__ import annotations

import hashlib
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

DEFAULT_CALENDARS = (
    "https://a.pool.opentimestamps.org",
    "https://b.pool.opentimestamps.org",
    "https://a.pool.eternitywall.com",
)
"""The aggregators the reference client submits to by default."""
MIN_CALENDARS = 2
"""A stamp needs this many calendars to accept it, as in the reference client."""
OTS_SUFFIX = ".ots"


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _load(ots_path: Path):
    from opentimestamps.core.serialize import StreamDeserializationContext
    from opentimestamps.core.timestamp import DetachedTimestampFile

    with open(ots_path, "rb") as handle:
        return DetachedTimestampFile.deserialize(StreamDeserializationContext(handle))


def _save(detached, ots_path: Path, *, replace: bool) -> None:
    from opentimestamps.core.serialize import StreamSerializationContext

    ots_path = Path(ots_path)
    if replace:
        tmp = ots_path.with_suffix(ots_path.suffix + ".tmp")
        with open(tmp, "wb") as handle:
            detached.serialize(StreamSerializationContext(handle))
        os.replace(tmp, ots_path)
        return
    fd = os.open(ots_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0), 0o644)
    with os.fdopen(fd, "wb") as handle:
        detached.serialize(StreamSerializationContext(handle))


def _remote_calendar(url: str):
    from opentimestamps.calendar import RemoteCalendar

    return RemoteCalendar(url, user_agent="zk-crown-p5.5")


def stamp_file(path: Path | str, calendars=DEFAULT_CALENDARS, *, timeout: float = 20.0,
               calendar_factory=_remote_calendar) -> dict[str, Any]:
    """Create ``<path>.ots`` by submitting the file's SHA-256 to the calendars.

    As in the reference client, a random 16-byte nonce is appended and hashed
    before submission, so the calendars see a digest that does not reveal the
    file's own hash. Refuses to replace an existing proof. Raises
    `RuntimeError` if fewer than ``MIN_CALENDARS`` accept.
    """
    from opentimestamps.core.op import OpAppend, OpSHA256
    from opentimestamps.core.timestamp import DetachedTimestampFile

    path = Path(path)
    ots_path = path.with_name(path.name + OTS_SUFFIX)
    if ots_path.exists():
        raise FileExistsError(f"{ots_path} already exists; a timestamp proof is not replaced")
    with open(path, "rb") as handle:
        detached = DetachedTimestampFile.from_fd(OpSHA256(), handle)
    tip = detached.timestamp.ops.add(OpAppend(os.urandom(16))).ops.add(OpSHA256())

    accepted, failed = [], {}
    for url in calendars:
        try:
            tip.merge(calendar_factory(url).submit(tip.msg, timeout=timeout))
            accepted.append(url)
        except Exception as error:  # noqa: BLE001 - one calendar being down must not lose the others
            failed[url] = f"{type(error).__name__}: {error}"
    if len(accepted) < MIN_CALENDARS:
        raise RuntimeError(f"only {len(accepted)} calendar(s) accepted the digest, need {MIN_CALENDARS}: {failed}")
    _save(detached, ots_path, replace=False)
    return {"ots_path": str(ots_path), "calendars_accepted": accepted, "calendars_failed": failed}


def describe_proof(ots_path: Path | str, file_path: Path | str | None = None) -> dict[str, Any]:
    """What a proof file says, without contacting anyone.

    Returns the digest the proof is for, whether it equals `file_path`'s
    SHA-256 (when given), the calendars still pending, and any Bitcoin block
    attestations with their height and the Merkle root the block header must
    have (hex, in the byte order block explorers display).

    This parses the proof. It does **not** check a Bitcoin attestation against
    the blockchain; ``status`` says ``"bitcoin-attested (unverified)"`` for
    that reason.
    """
    from opentimestamps.core.notary import BitcoinBlockHeaderAttestation, PendingAttestation

    detached = _load(Path(ots_path))
    pending, bitcoin, other = [], [], []
    for message, attestation in detached.timestamp.all_attestations():
        if isinstance(attestation, PendingAttestation):
            pending.append(attestation.uri)
        elif isinstance(attestation, BitcoinBlockHeaderAttestation):
            bitcoin.append({"height": attestation.height, "merkle_root": bytes(message)[::-1].hex()})
        else:
            other.append(type(attestation).__name__)
    digest = detached.file_digest.hex()
    return {
        "hash_op": type(detached.file_hash_op).__name__,
        "file_digest": digest,
        "matches_file": None if file_path is None else digest == _file_sha256(Path(file_path)),
        "pending_calendars": sorted(pending),
        "bitcoin_attestations": sorted(bitcoin, key=lambda a: a["height"]),
        "other_attestations": sorted(other),
        "status": "bitcoin-attested (unverified)" if bitcoin else "pending" if pending else "no attestation",
        "ots_sha256": _file_sha256(Path(ots_path)),
        "ots_bytes": Path(ots_path).stat().st_size,
    }


def _pending_nodes(timestamp):
    from opentimestamps.core.notary import PendingAttestation

    for attestation in timestamp.attestations:
        if isinstance(attestation, PendingAttestation):
            yield timestamp, attestation
    for child in timestamp.ops.values():
        yield from _pending_nodes(child)


def upgrade_proof(ots_path: Path | str, *, timeout: float = 20.0, calendar_factory=_remote_calendar) -> dict[str, Any]:
    """Ask each pending calendar for the rest of the proof and merge what it has.

    Only calendars on the library's default whitelist are contacted. The file
    is rewritten only if something was added. Returns what happened per
    calendar; "not ready" is the normal answer in the first hours.
    """
    from opentimestamps.calendar import DEFAULT_CALENDAR_WHITELIST

    ots_path = Path(ots_path)
    detached = _load(ots_path)
    before = describe_proof(ots_path)
    outcome = {}
    for node, attestation in list(_pending_nodes(detached.timestamp)):
        if attestation.uri not in DEFAULT_CALENDAR_WHITELIST:
            outcome[attestation.uri] = "skipped: not on the calendar whitelist"
            continue
        try:
            node.merge(calendar_factory(attestation.uri).get_timestamp(node.msg, timeout=timeout))
            outcome[attestation.uri] = "merged"
        except Exception as error:  # noqa: BLE001 - "not found yet" and network errors are both normal here
            outcome[attestation.uri] = f"not ready: {type(error).__name__}: {error}"
    changed = "merged" in outcome.values()
    if changed:
        _save(detached, ots_path, replace=True)
    after = describe_proof(ots_path)
    if after["file_digest"] != before["file_digest"]:
        raise RuntimeError("the upgraded proof is for another digest")
    return {"calendars": outcome, "changed": changed, "status": after["status"]}


# --- checking a Bitcoin attestation against the chain -------------------------

DEFAULT_EXPLORERS = ("https://blockstream.info/api", "https://mempool.space/api")
"""Two independent public block explorers with the same Esplora-style API."""
MIN_EXPLORERS = 2
"""A block counts as checked only if this many explorers answered and agreed."""
_BLOCK_HASH = re.compile(r"[0-9a-f]{64}")


def parse_block_header(header: bytes) -> dict[str, Any]:
    """Decode an 80-byte Bitcoin block header and check its proof of work.

    The block hash is SHA-256d of the header, shown byte-reversed as explorers
    do. The Merkle root is header bytes 36 to 68, also shown reversed. The
    work check is that the hash, as a number, is at most the target encoded in
    the header's ``bits`` field.
    """
    if not isinstance(header, (bytes, bytearray)) or len(header) != 80:
        raise ValueError("a block header is exactly 80 bytes")
    header = bytes(header)
    block_hash = hashlib.sha256(hashlib.sha256(header).digest()).digest()[::-1]
    bits = int.from_bytes(header[72:76], "little")
    exponent, mantissa = bits >> 24, bits & 0x007FFFFF
    target = mantissa << (8 * (exponent - 3)) if exponent >= 3 else mantissa >> (8 * (3 - exponent))
    return {
        "block_hash": block_hash.hex(),
        "merkle_root": header[36:68][::-1].hex(),
        "time": int.from_bytes(header[68:72], "little"),
        "bits": f"{bits:08x}",
        "meets_target": 0 < target and int.from_bytes(block_hash, "big") <= target,
    }


def _http_get(url: str, timeout: float) -> str:
    from urllib.request import Request, urlopen

    with urlopen(Request(url, headers={"User-Agent": "zk-crown-p5.5"}), timeout=timeout) as response:  # noqa: S310
        return response.read().decode("ascii").strip()


def check_bitcoin_attestations(attestations, explorers=DEFAULT_EXPLORERS, *, timeout: float = 20.0,
                               fetch=_http_get) -> dict[str, Any]:
    """Check each attested block's Merkle root against block headers fetched from several explorers.

    `attestations` is ``describe_proof(...)["bitcoin_attestations"]``. For each
    distinct height, every explorer is asked for the block hash at that height
    and for the raw 80-byte header. The header is hashed here, so it must hash
    to the block hash the explorer named, meet its own proof-of-work target,
    and carry the Merkle root the proof requires. A block is ``verified`` only
    if at least ``MIN_EXPLORERS`` explorers were asked, every one answered and
    passed those checks, and all named the same block.

    Only block heights and block hashes are sent. A failed lookup is recorded,
    not raised.

    What this does not do: validate the chain itself. It relies on the
    explorers' view of which block is at each height. Requiring two of them to
    agree, and checking the header's work, makes that harder to fake, but this
    is not a full node.
    """
    blocks = []
    expected: dict[int, set] = {}
    for attestation in attestations:
        expected.setdefault(int(attestation["height"]), set()).add(attestation["merkle_root"])
    for height in sorted(expected):
        if len(expected[height]) != 1:
            blocks.append({"height": height, "verified": False, "problem": "the proof requires two roots for one block"})
            continue
        root = next(iter(expected[height]))
        answers = {}
        for base in explorers:
            try:
                block_hash = fetch(f"{base}/block-height/{height}", timeout).lower()
                if not _BLOCK_HASH.fullmatch(block_hash):
                    raise ValueError(f"not a block hash: {block_hash[:80]!r}")
                header = parse_block_header(bytes.fromhex(fetch(f"{base}/block/{block_hash}/header", timeout)))
                answers[base] = {
                    "block_hash": block_hash,
                    "header_hashes_to_block_hash": header["block_hash"] == block_hash,
                    "meets_target": header["meets_target"],
                    "merkle_root": header["merkle_root"],
                    "merkle_root_matches": header["merkle_root"] == root,
                    "header_time": header["time"],
                }
            except Exception as error:  # noqa: BLE001 - one explorer failing is a finding, not a crash
                answers[base] = {"error": f"{type(error).__name__}: {error}"}
        good = [a for a in answers.values() if "error" not in a and a["header_hashes_to_block_hash"]
                and a["meets_target"] and a["merkle_root_matches"]]
        agreed = len({a["block_hash"] for a in good}) == 1
        verified = len(good) == len(answers) and len(good) >= MIN_EXPLORERS and agreed
        blocks.append({
            "height": height,
            "expected_merkle_root": root,
            "verified": verified,
            "block_hash": good[0]["block_hash"] if verified else None,
            "header_time_utc": (datetime.fromtimestamp(good[0]["header_time"], timezone.utc).isoformat(timespec="seconds")
                                if verified else None),
            "explorers": answers,
        })
    verified_blocks = [b for b in blocks if b["verified"]]
    return {
        "explorers": list(explorers),
        "min_explorers": MIN_EXPLORERS,
        "blocks": blocks,
        "all_verified": bool(blocks) and len(verified_blocks) == len(blocks),
        "earliest_verified_block": (None if not verified_blocks else
                                    {k: verified_blocks[0][k] for k in ("height", "block_hash", "header_time_utc")}),
    }


# --- the GPG-signed tag -------------------------------------------------------

_VALIDSIG = re.compile(r"^\[GNUPG:\] VALIDSIG ([0-9A-F]{40}) (\S+) (\d+)", re.MULTILINE)


def parse_verify_tag(status_output: str) -> dict[str, Any]:
    """Read gpg's machine-readable status lines, as printed by ``git verify-tag --raw``.

    ``good`` is True only with both a GOODSIG and a VALIDSIG line and no
    BADSIG, ERRSIG, EXPKEYSIG or REVKEYSIG line. ``signed_unix`` is the time
    the signer's machine wrote into the signature. It is self-asserted.
    """
    valid = _VALIDSIG.search(status_output)
    bad = any(f"[GNUPG:] {word} " in status_output for word in ("BADSIG", "ERRSIG", "EXPKEYSIG", "REVKEYSIG"))
    good = bool(valid) and "[GNUPG:] GOODSIG " in status_output and not bad
    return {
        "good": good,
        "fingerprint": valid.group(1) if valid else None,
        "signed_date": valid.group(2) if valid else None,
        "signed_unix": int(valid.group(3)) if valid else None,
    }
