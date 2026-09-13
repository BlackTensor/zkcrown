"""Create the owner's master key `K`, once, locally.

    python experiments/make_master_key.py            # writes secrets/K.bin

`K` is 32 bytes from the OS CSPRNG (`src.watermark.keygen.generate_key`),
written raw to `secrets/K.bin`, which is gitignored. The script refuses to
overwrite an existing key: every trigger, target, signature and commitment is
derived from `K`, so replacing it after a watermarked model exists orphans that
model.

Nothing about `K` is printed. Back the file up somewhere that is not this repo.
It never goes to Colab: training uses the trigger bundle instead
(`experiments/p2_3_make_trigger_bundle.py`).
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import _bootstrap  # noqa: F401  -- puts the repo root on sys.path

from src.watermark.keygen import KEY_BYTES, generate_key

DEFAULT_KEY_PATH = Path(__file__).resolve().parents[1] / "secrets" / "K.bin"


def create_key(path: Path) -> Path:
    """Write a fresh key to `path`. Raises `FileExistsError` if it exists."""
    path.parent.mkdir(parents=True, exist_ok=True)
    # O_EXCL: fail rather than overwrite, even if two runs race.
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0), 0o600)
    with os.fdopen(fd, "wb") as f:
        f.write(generate_key())
    if path.stat().st_size != KEY_BYTES:
        raise RuntimeError(f"{path} is not {KEY_BYTES} bytes after writing")
    return path


def load_key(path: Path = DEFAULT_KEY_PATH) -> bytes:
    """Read `K` from `path`, checking its length."""
    data = Path(path).read_bytes()
    if len(data) != KEY_BYTES:
        raise ValueError(f"{path} holds {len(data)} bytes, expected {KEY_BYTES}")
    return data


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--path", type=Path, default=DEFAULT_KEY_PATH)
    args = parser.parse_args(argv)
    try:
        create_key(args.path)
    except FileExistsError:
        raise SystemExit(f"{args.path} already exists. Refusing to overwrite the master key.")
    print(f"wrote a new {KEY_BYTES}-byte master key to {args.path}")
    print("Back it up outside this repo. Do not commit it, upload it, or paste it anywhere.")


if __name__ == "__main__":
    main()
