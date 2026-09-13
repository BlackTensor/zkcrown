"""Writing run artifacts safely on Colab.

Shared by `[GPU]` entry points from P0.7 on. P0.5 carries its own copies of
these two functions. They are left there unchanged so the committed P0.5
result still matches the code that produced it.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

COLAB_DRIVE_MOUNT = Path("/content/drive")


def require_mounted_drive(root: Path | str) -> None:
    """Fail if `root` points into Colab's Drive mount point but Drive is not mounted.

    `mkdir(parents=True)` would otherwise create /content/drive/MyDrive/... on
    the ephemeral local disk, and every checkpoint would be written there and
    lost with the session, with nothing printed to say so.
    """
    root = Path(root)
    if root.as_posix().startswith(COLAB_DRIVE_MOUNT.as_posix() + "/") and not os.path.ismount(
        COLAB_DRIVE_MOUNT
    ):
        raise RuntimeError(
            f"--drive-root {root} is under {COLAB_DRIVE_MOUNT}, but Drive is not mounted. "
            "Run the notebook's Drive mount cell first; otherwise checkpoints go to "
            "local disk and vanish with the session."
        )


def save_state_dict(model, path: Path | str) -> str:
    """Write `model`'s state_dict atomically. Returns the file's SHA-256."""
    import torch

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(model.state_dict(), tmp)
    os.replace(tmp, path)
    return hashlib.sha256(path.read_bytes()).hexdigest()
