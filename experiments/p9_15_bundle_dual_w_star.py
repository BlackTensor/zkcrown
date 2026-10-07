"""P9.15: bundle the dual-watermarked W* (P3.6) for the hosted dashboard, as a numpy ``.npz``.

The dashboard's live fingerprint check needs the weights, and the host has no
torch. So the P3.6 state dict is written as ``results/p9.15_dual_W_star.npz``:
one ``.npy`` member per state-dict entry, in a zip with fixed timestamps and no
compression, so the same weights always give the same bytes.

What the script checks before it keeps the file:

- the source ``.pt`` has the SHA-256 P3.6 recorded (it is gitignored, so it is
  loaded by hash);
- the bundle reloads with ``np.load(..., allow_pickle=False)`` and every array
  is bit-identical to the torch tensor it came from;
- the bundle's P5.1 fingerprint equals the fingerprint the published commitment
  (``provenance/commitment.json``) names, ``c0995109…``;
- it is not the behavioral-only P2.3 model or clean ``W`` (their fingerprints
  are compared too). Bundling P2.3 next to the dual model would publish their
  difference, which is exactly the owner's weight-watermark change.

This is the only model the hosted app bundles (owner decision, 2026-10-07).
Nothing is measured; no owner secret is read.

    python experiments/p9_15_bundle_dual_w_star.py
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import time
import zipfile

import numpy as np

import _bootstrap  # noqa: F401  -- puts the repo root on sys.path

from src.crypto.fingerprint import fingerprint_state_dict
from src.utils.results import git_info, repo_root, write_result
from src.utils.seeding import DEFAULT_SEED

SOURCE = "results/p3.6_dual_wm_W_star.pt"
SOURCE_SHA256 = "7a9a9f141b55c7894b208c1f338f4c7385362f969b82b319ce5893fb4bb434c4"
BUNDLE = "results/p9.15_dual_W_star.npz"
PUBLICATION = "provenance/commitment.json"
P5_1_RESULT_PREFIX = "p5.1_model_fingerprint__"
NOT_THESE = ("behavioral_only_W_star", "clean_W")
"""P5.1 model keys whose fingerprint the bundle must not have."""
ZIP_DATE = (1980, 1, 1, 0, 0, 0)


def npz_bytes(arrays: dict[str, np.ndarray]) -> bytes:
    """A deterministic ``.npz``: sorted members, fixed timestamps, stored (no compression)."""
    out = io.BytesIO()
    with zipfile.ZipFile(out, "w", compression=zipfile.ZIP_STORED) as zf:
        for name in sorted(arrays):
            member = io.BytesIO()
            np.lib.format.write_array(member, np.asarray(arrays[name]), allow_pickle=False)
            info = zipfile.ZipInfo(f"{name}.npy", date_time=ZIP_DATE)
            info.external_attr = 0o644 << 16
            zf.writestr(info, member.getvalue())
    return out.getvalue()


def load_npz(data: bytes) -> dict[str, np.ndarray]:
    with np.load(io.BytesIO(data), allow_pickle=False) as z:
        return {name: z[name] for name in z.files}


def main(argv: list[str] | None = None) -> dict:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="recorded only")
    parser.add_argument("--out-dir", default=None)
    args = parser.parse_args(argv)
    import torch

    git_snapshot = git_info()
    started = time.perf_counter()
    root = repo_root()
    source = root / SOURCE
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    if digest != SOURCE_SHA256:
        raise SystemExit(f"{SOURCE} has SHA-256 {digest}, not P3.6's {SOURCE_SHA256}")
    state = torch.load(source, map_location="cpu", weights_only=True)
    arrays = {name: tensor.detach().cpu().numpy() for name, tensor in state.items()}
    data = npz_bytes(arrays)
    if npz_bytes(arrays) != data:
        raise SystemExit("the bundle bytes are not deterministic")

    reloaded = load_npz(data)
    if set(reloaded) != set(arrays):
        raise SystemExit("the bundle does not hold exactly the state dict's entries")
    for name, array in arrays.items():
        back = reloaded[name]
        if back.dtype != array.dtype or back.shape != array.shape or back.tobytes() != array.tobytes():
            raise SystemExit(f"entry {name} does not round-trip bit for bit")

    published = json.loads((root / PUBLICATION).read_text(encoding="utf-8"))["model_fingerprint"]
    bundle_fp = fingerprint_state_dict(reloaded)
    if bundle_fp.sha256 != published["sha256"] or bundle_fp.tensors != published["tensors"]:
        raise SystemExit(f"bundle fingerprint {bundle_fp.sha256} is not the published {published['sha256']}")
    p5_1 = sorted((root / "results").glob(P5_1_RESULT_PREFIX + "*.json"))[-1]
    models = json.loads(p5_1.read_text(encoding="utf-8"))["metrics"]["models"]
    others = {key: models[key]["fingerprint"]["sha256"] for key in NOT_THESE}
    if models["dual_W_star"]["fingerprint"]["sha256"] != published["sha256"]:
        raise SystemExit("P5.1's dual W* fingerprint is not the published one")
    if bundle_fp.sha256 in others.values():
        raise SystemExit("the bundle is one of the models that must not be bundled")

    bundle = root / BUNDLE
    if bundle.exists() and bundle.read_bytes() != data:
        raise SystemExit(f"{BUNDLE} exists with other bytes; refusing to replace it")
    bundle.write_bytes(data)
    bundle_sha256 = hashlib.sha256(bundle.read_bytes()).hexdigest()

    path = write_result(
        name="p9.15_dual_w_star_bundle",
        seed=args.seed,
        task="P9.15",
        params={"source": SOURCE, "source_sha256": SOURCE_SHA256, "bundle": BUNDLE,
                "format": "numpy .npz: one .npy per state-dict entry, sorted, stored, fixed 1980-01-01 timestamps, "
                          "allow_pickle=False",
                "publication": PUBLICATION, "p5_1_result": p5_1.relative_to(root).as_posix(),
                "must_not_equal": list(NOT_THESE)},
        metrics={"bundle_sha256": bundle_sha256, "bundle_bytes": len(data), "entries": len(arrays),
                 "fingerprint": bundle_fp.to_dict(), "equals_published_fingerprint": True,
                 "round_trip_bit_identical": True, "deterministic": True,
                 "differs_from": {key: True for key in NOT_THESE}},
        duration_seconds=time.perf_counter() - started,
        out_dir=args.out_dir,
        git=git_snapshot,
        notes="The only model bundled for the hosted dashboard. Its fingerprint is the one the published commitment "
              "names. No owner secret was read.",
    )
    print(f"wrote {BUNDLE}: {len(data):,} bytes, SHA-256 {bundle_sha256}")
    print(f"fingerprint {bundle_fp.sha256} equals the published one")
    print("wrote", path)
    return {"path": path, "bundle_sha256": bundle_sha256}


if __name__ == "__main__":
    main()
