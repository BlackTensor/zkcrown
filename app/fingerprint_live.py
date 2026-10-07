"""The live fingerprint check for the bundled model (P9.15), the one app module that imports numpy.

The hosted app bundles exactly one model, the dual-watermarked W* (P3.6), as
``results/p9.15_dual_W_star.npz`` (owner decision, 2026-10-07). This module
reads that file through the data layer (so its bytes are hash-checked against
the manifest), loads it with ``allow_pickle=False``, and computes its P5.1
fingerprint with the repo's own code (``src/crypto/fingerprint.py``, loaded by
path through `app.repo_code`). The result is compared with the fingerprint the
provenance record names, and with the one the recorded audit measured for the
verbatim-copy suspect.

numpy is a dependency of Streamlit itself, so this adds nothing to the host's
requirements. The import test allows numpy in this module only.
"""

from __future__ import annotations

import io
from dataclasses import dataclass

import numpy as np

from app import repo_code

BUNDLE = "results/p9.15_dual_W_star.npz"
BUNDLE_RECORD = "results/p9.15_dual_w_star_bundle__"


@dataclass(frozen=True)
class LiveFingerprint:
    bundle: str
    bundle_sha256: str
    fingerprint: str
    tensors: int
    record_fingerprint: str
    audited_fingerprint: str | None
    matches_record: bool
    matches_audit: bool

    @property
    def status(self) -> str:
        return "passed" if self.matches_record else "failed"


def live_fingerprint(store, record: dict, audited_fingerprint: str | None) -> LiveFingerprint:
    """Fingerprint the bundled weights now. Raises `DataIntegrityError` if the bundle fails its hash check."""
    data = store.read_bytes(BUNDLE)
    with np.load(io.BytesIO(data), allow_pickle=False) as archive:
        arrays = {name: archive[name] for name in archive.files}
    fp = repo_code.fingerprint().fingerprint_state_dict(arrays)
    expected = record["model"]["fingerprint"]["sha256"]
    return LiveFingerprint(
        bundle=BUNDLE, bundle_sha256=store.recorded_sha256(BUNDLE), fingerprint=fp.sha256, tensors=fp.tensors,
        record_fingerprint=expected, audited_fingerprint=audited_fingerprint,
        matches_record=fp.sha256 == expected, matches_audit=fp.sha256 == audited_fingerprint)
