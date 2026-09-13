"""P2.3, local step: build the secret trigger bundle from `K` for the Colab run.

    python experiments/p2_3_make_trigger_bundle.py

Reads `secrets/K.bin`, generates the N = 100 trigger set on the CIFAR-10
training split (P1.2) and its per-trigger targets (P2.2), and writes
`secrets/trigger_bundle.npz`. Prints the bundle's SHA-256 digest, which the
Colab notebook checks before training.

CPU only, a few seconds. `K` stays on this machine; the bundle is what gets
uploaded. The bundle is secret too, so it goes in `secrets/` and nowhere in
git.

Re-running is safe: if the bundle already exists, it is kept only if it is
identical to what `K` gives now, and the script fails otherwise.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import _bootstrap  # noqa: F401  -- puts the repo root on sys.path
from make_master_key import DEFAULT_KEY_PATH, load_key

from src.watermark.bundle import load_bundle, make_bundle, save_bundle
from src.watermark.responses import cifar10_trigger_responses
from src.watermark.triggers import DEFAULT_AMPLITUDE, DEFAULT_N, generate_cifar10_triggers

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BUNDLE_PATH = REPO_ROOT / "secrets" / "trigger_bundle.npz"


def build(key: bytes, data_root: Path, n: int, amplitude: int, download: bool):
    triggers = generate_cifar10_triggers(key, data_root, n=n, amplitude=amplitude, download=download)
    responses = cifar10_trigger_responses(key, triggers, data_root)
    return make_bundle(triggers, responses, key_kind="owner")


def main(argv: list[str] | None = None) -> str:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--key", type=Path, default=DEFAULT_KEY_PATH)
    parser.add_argument("--out", type=Path, default=DEFAULT_BUNDLE_PATH)
    parser.add_argument("--data-root", type=Path, default=REPO_ROOT / "data")
    parser.add_argument("--n", type=int, default=DEFAULT_N)
    parser.add_argument("--amplitude", type=int, default=DEFAULT_AMPLITUDE)
    parser.add_argument("--download", action="store_true", help="download CIFAR-10 if missing")
    args = parser.parse_args(argv)

    bundle = build(load_key(args.key), args.data_root, args.n, args.amplitude, args.download)
    digest = bundle.digest()

    if args.out.exists():
        existing = load_bundle(args.out)
        if existing.digest() != digest:
            raise SystemExit(
                f"{args.out} exists with digest {existing.digest()}, but K now gives {digest}. "
                "Refusing to overwrite. Did K change?"
            )
        print(f"{args.out} already exists and matches K.")
    else:
        save_bundle(bundle, args.out)
        load_bundle(args.out, expected_digest=digest)  # read back before trusting it
        print(f"wrote {args.out}")

    print(f"triggers   : {len(bundle)}  amplitude {bundle.amplitude}  key_kind {bundle.key_kind}")
    print(f"digest     : {digest}")
    return digest


if __name__ == "__main__":
    main()
