"""Per-epoch checkpointing that survives a Colab session dying.

CLAUDE.md 0.5 and 2.1: Colab wipes its filesystem between sessions and times
out at roughly 90 minutes idle, so every `[GPU]` run must checkpoint to Drive
every epoch and be resumable from the last checkpoint.

Two things make this harder than `torch.save(model.state_dict(), path)`:

1. **A single checkpoint file is a single point of failure.** If the session
   dies partway through writing it, the only good checkpoint is now a truncated
   file and the whole run is lost. `os.replace` is atomic on a local
   filesystem, but Drive is a FUSE mount and does not promise that. So writes
   alternate between two slots and a manifest records which one is newer. A
   corrupted write can therefore only ever destroy the *older* checkpoint, and
   `load_latest` falls back to it.

2. **Resuming is not just weights.** Optimizer momentum, the LR schedule
   position and the RNG states all have to come back, or a resumed run is not
   the run that would have happened without the interruption. The global RNG
   states are saved too (dropout, and augmentation when it runs in the main
   process). The DataLoader's shuffle generator is not in the checkpoint;
   `fit` reseeds it per epoch from the run seed, which is what makes data
   order and worker augmentation continue rather than restart.
"""

from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Any

import torch

MANIFEST_NAME = "manifest.json"
SLOT_NAMES = ("ckpt_a.pt", "ckpt_b.pt")
BEST_NAME = "best.pt"


def _rng_state() -> dict[str, Any]:
    state: dict[str, Any] = {
        "python": random.getstate(),
        "torch": torch.get_rng_state(),
    }
    try:
        import numpy as np

        state["numpy"] = np.random.get_state()
    except ImportError:
        state["numpy"] = None
    state["cuda"] = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    return state


def _restore_rng_state(state: dict[str, Any] | None) -> None:
    if not state:
        return
    if state.get("python") is not None:
        random.setstate(state["python"])
    if state.get("torch") is not None:
        torch.set_rng_state(state["torch"].cpu() if hasattr(state["torch"], "cpu") else state["torch"])
    if state.get("numpy") is not None:
        try:
            import numpy as np

            np.random.set_state(state["numpy"])
        except ImportError:
            pass
    if state.get("cuda") is not None and torch.cuda.is_available():
        try:
            torch.cuda.set_rng_state_all(state["cuda"])
        except (RuntimeError, ValueError):
            # Different GPU count than the run that saved it. Not fatal.
            pass


def _read_manifest(directory: Path) -> dict[str, Any]:
    path = directory / MANIFEST_NAME
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def save_checkpoint(
    directory: Path | str,
    *,
    epoch: int,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: Any | None = None,
    history: list[dict[str, Any]] | None = None,
    best: dict[str, Any] | None = None,
    config: dict[str, Any] | None = None,
    seed: int | None = None,
    is_best: bool = False,
) -> Path:
    """Write a checkpoint for `epoch`. Returns the path written.

    Alternates between two slots so an interrupted write cannot destroy the
    last good checkpoint. The manifest is written last and is the only thing
    that promotes a slot to "current".
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)

    payload = {
        "epoch": epoch,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "history": history or [],
        "best": best or {},
        "config": config or {},
        "seed": seed,
        "rng": _rng_state(),
    }

    slot = SLOT_NAMES[epoch % len(SLOT_NAMES)]
    path = directory / slot
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp)
    os.replace(tmp, path)

    if is_best:
        best_tmp = directory / (BEST_NAME + ".tmp")
        torch.save(payload, best_tmp)
        os.replace(best_tmp, directory / BEST_NAME)

    # Written last: until this lands, the previous slot is still "current".
    manifest = {"current": slot, "epoch": epoch, "previous": _read_manifest(directory).get("current")}
    manifest_tmp = directory / (MANIFEST_NAME + ".tmp")
    manifest_tmp.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    os.replace(manifest_tmp, directory / MANIFEST_NAME)

    return path


def load_latest(directory: Path | str, map_location: str = "cpu") -> dict[str, Any] | None:
    """Load the newest loadable checkpoint, or None if there is none.

    Tries the slot the manifest names, then the other slot, then any slot file
    present. A truncated file raises on load, which is exactly the case the
    fallback exists for.
    """
    directory = Path(directory)
    if not directory.is_dir():
        return None

    manifest = _read_manifest(directory)
    order = []
    if manifest.get("current"):
        order.append(manifest["current"])
    order += [slot for slot in SLOT_NAMES if slot not in order]

    for slot in order:
        path = directory / slot
        if not path.is_file():
            continue
        try:
            return torch.load(path, map_location=map_location, weights_only=False)
        except Exception as exc:  # noqa: BLE001 - any failure means "try the other slot"
            print(f"checkpoint {path.name} is unusable ({exc}); trying the next slot")
    return None


def clear_checkpoints(directory: Path | str) -> list[str]:
    """Delete the slots, manifest and best checkpoint in `directory`.

    For starting a run over. Leaving an old slot in place is not harmless: if
    the new run's first save is interrupted, `load_latest` falls back to that
    slot and a later resume silently continues the *old* run. Returns the names
    removed. Other files in the directory are left alone.
    """
    directory = Path(directory)
    if not directory.is_dir():
        return []
    removed = []
    for name in (*SLOT_NAMES, MANIFEST_NAME, BEST_NAME):
        for path in (directory / name, directory / (name + ".tmp")):
            if path.is_file():
                path.unlink()
                removed.append(path.name)
    return removed


def restore(
    checkpoint: dict[str, Any],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: Any | None = None,
    *,
    restore_rng: bool = True,
) -> int:
    """Restore state in place. Returns the epoch the checkpoint was saved at.

    Training should continue from the returned epoch + 1.
    """
    model.load_state_dict(checkpoint["model"])
    if optimizer is not None and checkpoint.get("optimizer"):
        optimizer.load_state_dict(checkpoint["optimizer"])
    if scheduler is not None and checkpoint.get("scheduler"):
        scheduler.load_state_dict(checkpoint["scheduler"])
    if restore_rng:
        _restore_rng_state(checkpoint.get("rng"))
    return int(checkpoint["epoch"])
