"""Global seed control.

Every experiment in this project takes a seed (CLAUDE.md 0.3), so seeding has
to happen in exactly one place and has to report honestly what it managed to
seed. `torch` and `numpy` are optional imports here: the crypto, Circom and
auditor work runs in environments where neither is installed, and importing
this module must not fail there.

Typical use at the top of an experiment::

    from src.utils.seeding import set_seed
    backends = set_seed(1337)
"""

from __future__ import annotations

import os
import random
from typing import Any

DEFAULT_SEED = 1337
"""Used when a script has no reason to prefer a particular seed."""


def _try_import(name: str) -> Any | None:
    try:
        return __import__(name)
    except ImportError:
        return None


def set_seed(seed: int = DEFAULT_SEED, *, deterministic: bool = True) -> dict[str, bool]:
    """Seed every RNG this project might touch.

    Args:
        seed: the seed. Must be a non-negative int.
        deterministic: if True, also ask cuDNN for deterministic kernel
            selection. This costs some throughput.

    Returns:
        A mapping of backend name to whether it was actually seeded, e.g.
        ``{"python": True, "numpy": False, "torch": True, "cuda": False}``.
        Store this in the result record so a reader can tell which RNGs were
        under control when the number was produced.

    Note:
        ``PYTHONHASHSEED`` is set here, but CPython reads it at interpreter
        startup, so setting it now only affects *subprocesses* we spawn. It is
        set anyway because it costs nothing and makes spawned workers
        consistent. If hash randomisation ever matters to a result, the
        interpreter must be launched with the variable already set.

        ``torch.use_deterministic_algorithms(True)`` is deliberately *not*
        enabled. It raises on operations with no deterministic CUDA kernel,
        which would break training runs mid-epoch. cuDNN determinism plus a
        fixed seed is the practical compromise; results are reproducible on the
        same hardware and library versions, not bit-identical across machines.
    """
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise TypeError(f"seed must be an int, got {type(seed).__name__}")
    if seed < 0:
        raise ValueError(f"seed must be non-negative, got {seed}")

    seeded = {"python": False, "numpy": False, "torch": False, "cuda": False}

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    seeded["python"] = True

    np = _try_import("numpy")
    if np is not None:
        np.random.seed(seed)
        seeded["numpy"] = True

    torch = _try_import("torch")
    if torch is not None:
        torch.manual_seed(seed)
        seeded["torch"] = True
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            seeded["cuda"] = True
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    return seeded


def torch_generator(seed: int = DEFAULT_SEED):
    """Return a seeded `torch.Generator`.

    Pass this to `DataLoader(generator=...)`. Without it, shuffling draws from
    the global RNG and the data order depends on whatever else consumed random
    numbers first, which quietly breaks reproducibility.

    Raises:
        RuntimeError: if torch is not installed.
    """
    torch = _try_import("torch")
    if torch is None:
        raise RuntimeError("torch is not installed; torch_generator() is unavailable")
    generator = torch.Generator()
    generator.manual_seed(seed)
    return generator


def seed_worker(worker_id: int) -> None:
    """`DataLoader(worker_init_fn=...)` hook.

    Each worker process inherits a fresh torch seed from the parent; this
    derives the `random` and `numpy` seeds from it so augmentations that use
    those libraries are reproducible too.
    """
    torch = _try_import("torch")
    if torch is None:
        return
    worker_seed = torch.initial_seed() % 2**32
    random.seed(worker_seed)
    np = _try_import("numpy")
    if np is not None:
        np.random.seed(worker_seed)
