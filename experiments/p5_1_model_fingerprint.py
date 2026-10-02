"""P5.1: fingerprint the project's models and check the fingerprint survives save and reload.

    python experiments/p5_1_model_fingerprint.py

CPU only, a few seconds. Needs the local weights files, no key and no data.

The fingerprint is `src/crypto/fingerprint.py`: SHA-256 over a canonical
serialization of the state dict. For each of the four named models (clean `W`,
the P2.3 behavioral model, the P3.6 dual `W*`, and `zk_model`) the script:

1. checks the weights file against its recorded file SHA-256 and fingerprints
   the state dict loaded from it. That is the reference.
2. repeats the fingerprint through each of these routes and requires the
   reference every time:

   - ``model``: strict load into a fresh model, then ``model.state_dict()``;
   - ``resave``: `torch.save` under another file name, reload;
   - ``resave_twice``: save and reload that a second time;
   - ``reversed_keys``: save with the dict order reversed, reload;
   - ``legacy_format``: `torch.save` in the pre-zip format, reload;
   - ``npz``: through a numpy ``.npz`` archive, with no torch on the way back;
   - ``noncontiguous``: every tensor of 2 or more dimensions replaced by a
     non-contiguous view with the same values;
   - ``fresh_process``: computed by a new Python process, with a different
     ``PYTHONHASHSEED``, from the re-saved file.

   It also records whether the re-saved file's own SHA-256 differs from the
   original's, which is the reason a file hash is not used.
3. changes the weights and requires a *different* fingerprint each time: the
   lowest bit of one element of the first tensor flipped, the same values as
   float64, and one tensor renamed.

Any failed requirement stops the run before a result is written.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import _bootstrap  # noqa: F401  -- puts the repo root on sys.path
import numpy as np
import torch
from p2_4_measure_wdr import W_CLEAN_SHA256, W_STAR_SHA256, sha256_file

from src.crypto.fingerprint import FINGERPRINT_VERSION, fingerprint_model, fingerprint_state_dict
from src.models import main_model, zk_model
from src.utils.results import git_info, repo_root, write_result
from src.utils.seeding import DEFAULT_SEED, set_seed

W_DUAL_SHA256 = "7a9a9f141b55c7894b208c1f338f4c7385362f969b82b319ce5893fb4bb434c4"
"""Dual `W*` weights file, from results/p3.6_dual_wm__seed1337__20260914T085541+0000.json."""
ZK_MODEL_SHA256 = "6bc298d04edfe5af136349904c53ccc9a82e80dfb03729f616e552a3df9ddec4"
"""`zk_model` weights file, from results/p0.7_zk_model__seed1337__20260913T075832+0000.json."""

MODELS = {
    "clean_W": {"task": "P0.5", "file": "results/p0.5_clean_baseline_W.pt", "file_sha256": W_CLEAN_SHA256,
                "build": main_model},
    "behavioral_only_W_star": {"task": "P2.3", "file": "results/p2.3_behavioral_wm_W_star.pt",
                               "file_sha256": W_STAR_SHA256, "build": main_model},
    "dual_W_star": {"task": "P3.6", "file": "results/p3.6_dual_wm_W_star.pt", "file_sha256": W_DUAL_SHA256,
                    "build": main_model},
    "zk_model": {"task": "P0.7", "file": "results/p0.7_zk_model.pt", "file_sha256": ZK_MODEL_SHA256,
                 "build": zk_model},
}
FRESH_PROCESS_HASHSEED = "4242"


def load_state(path: Path) -> dict:
    return torch.load(path, map_location="cpu", weights_only=True)


def flip_lowest_bit(state: dict) -> dict:
    """A copy of `state` with the lowest bit of one element of its first tensor flipped."""
    changed = {name: tensor.clone() for name, tensor in state.items()}
    name = next(iter(changed))
    raw = changed[name].numpy().reshape(-1).view(np.uint8)  # shares memory with the clone
    raw[0] ^= 1
    return changed


def noncontiguous(state: dict) -> dict:
    """The same values, with every tensor of 2+ dimensions held as a non-contiguous view."""
    out = {}
    for name, tensor in state.items():
        if tensor.dim() >= 2:
            view = tensor.transpose(0, 1).contiguous().transpose(0, 1)
            if view.is_contiguous() and view.numel() > 1 and min(view.shape[:2]) > 1:
                raise SystemExit(f"{name}: expected a non-contiguous view")
            out[name] = view
        else:
            out[name] = tensor
    return out


def fresh_process_fingerprint(path: Path) -> str:
    """The fingerprint of `path` as a new interpreter computes it, under another PYTHONHASHSEED."""
    env = {**os.environ, "PYTHONHASHSEED": FRESH_PROCESS_HASHSEED}
    done = subprocess.run([sys.executable, "-m", "src.crypto.fingerprint", str(path)], cwd=repo_root(), env=env,
                          capture_output=True, text=True, check=True)
    return done.stdout.split()[0]


def check_model(name: str, spec: dict, tmp: Path) -> dict:
    path = repo_root() / spec["file"]
    file_sha = sha256_file(path)
    if file_sha != spec["file_sha256"]:
        raise SystemExit(f"{path} has SHA-256 {file_sha}, expected {spec['file_sha256']}. Refusing to fingerprint it.")
    state = load_state(path)
    reference = fingerprint_state_dict(state)

    model = spec["build"]()
    model.load_state_dict(state, strict=True)

    resaved = tmp / f"{name}_resaved.pt"
    torch.save(state, resaved)
    twice = tmp / f"{name}_resaved_twice.pt"
    torch.save(load_state(resaved), twice)
    reversed_path = tmp / f"{name}_reversed.pt"
    torch.save(dict(reversed(list(state.items()))), reversed_path)
    legacy = tmp / f"{name}_legacy.pt"
    torch.save(state, legacy, _use_new_zipfile_serialization=False)
    npz = tmp / f"{name}.npz"
    np.savez(npz, **{key: tensor.numpy() for key, tensor in state.items()})
    with np.load(npz) as archive:
        from_npz = {key: archive[key] for key in archive.files}

    routes = {
        "model": fingerprint_model(model).sha256,
        "resave": fingerprint_state_dict(load_state(resaved)).sha256,
        "resave_twice": fingerprint_state_dict(load_state(twice)).sha256,
        "reversed_keys": fingerprint_state_dict(load_state(reversed_path)).sha256,
        "legacy_format": fingerprint_state_dict(load_state(legacy)).sha256,
        "npz": fingerprint_state_dict(from_npz).sha256,
        "noncontiguous": fingerprint_state_dict(noncontiguous(state)).sha256,
        "fresh_process": fresh_process_fingerprint(resaved),
    }
    for route, digest in routes.items():
        if digest != reference.sha256:
            raise SystemExit(f"{name}: route {route} gave {digest}, reference is {reference.sha256}")

    first = next(iter(state))
    changes = {
        "one_bit_flipped": fingerprint_state_dict(flip_lowest_bit(state)).sha256,
        "cast_to_float64": fingerprint_state_dict(
            {k: t.double() if t.is_floating_point() else t for k, t in state.items()}).sha256,
        "one_tensor_renamed": fingerprint_state_dict(
            {(k + "_" if k == first else k): t for k, t in state.items()}).sha256,
    }
    for change, digest in changes.items():
        if digest == reference.sha256:
            raise SystemExit(f"{name}: change {change} left the fingerprint unchanged")

    return {
        "task": spec["task"],
        "file": spec["file"],
        "file_sha256": file_sha,
        "fingerprint": reference.to_dict(),
        "dtypes": sorted({str(t.dtype).replace("torch.", "") for t in state.values()}),
        "stable_routes": {route: True for route in routes},
        "resaved_file_sha256_differs": sha256_file(resaved) != file_sha,
        "changed_weights_change_fingerprint": {change: True for change in changes},
    }


def main(argv: list[str] | None = None) -> dict:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--out-dir", type=Path, default=None)
    args = parser.parse_args(argv)

    git = git_info()
    backends = set_seed(args.seed)  # nothing here is random; seeded for the record's sake
    started = time.perf_counter()
    with tempfile.TemporaryDirectory() as tmp:
        models = {name: check_model(name, spec, Path(tmp)) for name, spec in MODELS.items()}

    digests = [m["fingerprint"]["sha256"] for m in models.values()]
    if len(set(digests)) != len(digests):
        raise SystemExit("two different models share a fingerprint")

    metrics = {
        "models": models,
        "all_routes_stable": True,
        "all_changes_detected": True,
        "fingerprints_distinct": True,
    }
    path = write_result(
        name="p5.1_model_fingerprint",
        seed=args.seed,
        task="P5.1",
        params={
            "fingerprint_version": FINGERPRINT_VERSION,
            "hash": "SHA-256",
            "covers": "every state_dict entry, parameters and buffers, sorted by name",
            "routes": ["model", "resave", "resave_twice", "reversed_keys", "legacy_format", "npz", "noncontiguous",
                       "fresh_process"],
            "fresh_process_pythonhashseed": FRESH_PROCESS_HASHSEED,
            "changes": ["one_bit_flipped", "cast_to_float64", "one_tensor_renamed"],
            "device": "cpu",
        },
        metrics=metrics,
        seeded_backends=backends,
        duration_seconds=time.perf_counter() - started,
        out_dir=args.out_dir,
        git=git,
        notes=(
            "Fingerprints identify exact weights: an equality check, not a watermark, and not robust to any "
            "modification. Stability is checked on this machine and torch build only; no second platform was run."
        ),
    )

    for name, m in models.items():
        f = m["fingerprint"]
        print(f"{name:24s} {f['sha256']}  {f['tensors']} tensors, {f['elements']:,} elements, "
              f"{f['data_bytes']:,} bytes  re-saved file hash differs: {m['resaved_file_sha256_differs']}")
    print("all routes stable, all changes detected")
    print("wrote", path)
    return {"path": path, "metrics": metrics}


if __name__ == "__main__":
    main()
