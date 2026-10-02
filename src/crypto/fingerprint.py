"""SHA-256 model fingerprint over a canonical serialization of the weights (P5.1).

Why not hash the weights file
-----------------------------
Every model so far is identified by the SHA-256 of its ``.pt`` file. That
identifies a *file*, not a set of weights: `torch.save` writes a zip archive
that embeds the file name (found in P3.6), and its layout is free to change
between torch versions. The same weights saved under another name, by another
torch, or in another container (ONNX, npz) give a different file hash. The
provenance record (P5.4, P6.1) needs an identifier of the weights themselves.

Canonical serialization, ``zk-crown/model-fingerprint/v1``
---------------------------------------------------------
The fingerprint is the SHA-256 of this byte string. All length and count
fields are unsigned big-endian integers. Tensor data is little-endian.

    "zk-crown/model-fingerprint/v1" || 0x00
    u32  number of tensors
    then, for each tensor, sorted by the UTF-8 bytes of its name:
        u16  len(name)      || name, UTF-8
        u8   len(dtype)     || dtype, ASCII numpy name such as "float32"
        u8   number of dimensions
        u64  each dimension
        u64  number of data bytes
        data: the elements in C (row-major) order, little-endian

Decisions:

- **Every state_dict entry is covered**, buffers included: BatchNorm running
  statistics and ``num_batches_tracked`` as well as parameters. A model whose
  BN statistics were recalibrated is a different model and gets a different
  fingerprint.
- **Sorted by name**, so the fingerprint does not depend on dict order.
- **Names, dtypes and shapes are hashed**, each length-prefixed, so no two
  different state dicts serialize to the same bytes. The same values stored as
  float16 or float64 are a different fingerprint.
- **Bit-exact.** Elements are hashed as their IEEE-754 or integer bytes. So
  ``-0.0`` differs from ``0.0``, and two NaNs with different payloads differ.
  Nothing is rounded.
- **Independent of** the container file, its name, the torch version, the
  device, memory layout (strides, contiguity), ``requires_grad``, the host's
  byte order and ``PYTHONHASHSEED``.

What it is and is not
---------------------
It identifies one exact set of weights. It is an equality check: any change at
all, a single bit in one weight, gives an unrelated fingerprint. So it does
**not** survive pruning, quantization, fine-tuning or any other attack in
Phase 4, and it is not a watermark. In an audit it answers "is this suspect
bit-for-bit the model in the provenance record?" and nothing more. A mismatch
says nothing about ownership; that is what the watermark tests are for.

It also does not canonicalize function-preserving changes: conv-BN fusion,
permuting channels or rescaling across a ReLU all leave the outputs unchanged
and change the fingerprint. The arch (for example ``width``) is covered only
through the tensor names and shapes.

SHA-256 here is plain `hashlib`. It is collision resistant, not keyed, and
needs no secret. The fingerprint of a secret model is safe to publish in the
sense that it does not reveal the weights, but anyone holding the weights can
confirm the match.
"""

from __future__ import annotations

import hashlib
import struct
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

FINGERPRINT_VERSION = "zk-crown/model-fingerprint/v1"
_MAGIC = FINGERPRINT_VERSION.encode("ascii") + b"\x00"

SUPPORTED_DTYPES = (
    "bool",
    "int8", "int16", "int32", "int64",
    "uint8", "uint16", "uint32", "uint64",
    "float16", "float32", "float64",
)
"""numpy dtype names accepted. Anything else is refused, not converted."""


@dataclass(frozen=True)
class ModelFingerprint:
    """The fingerprint of one state dict, with the counts that went into it.

    Attributes:
        sha256: hex digest of the canonical serialization.
        version: the serialization version tag.
        tensors: number of state_dict entries hashed.
        elements: total number of elements over all tensors.
        data_bytes: total number of tensor data bytes hashed.
    """

    sha256: str
    version: str
    tensors: int
    elements: int
    data_bytes: int

    def to_dict(self) -> dict[str, Any]:
        return {"sha256": self.sha256, "version": self.version, "tensors": self.tensors,
                "elements": self.elements, "data_bytes": self.data_bytes}


def _as_array(name: str, value: Any) -> np.ndarray:
    """`value` as a numpy array, without changing its dtype or values."""
    if isinstance(value, np.ndarray):
        array = value
    elif hasattr(value, "detach") and hasattr(value, "numpy"):  # a torch tensor, without importing torch
        if getattr(value, "is_sparse", False) or getattr(value, "is_quantized", False):
            raise TypeError(f"tensor {name!r} is sparse or quantized; fingerprint dense tensors only")
        try:
            array = value.detach().cpu().numpy()
        except (TypeError, RuntimeError) as error:
            raise TypeError(f"tensor {name!r} has dtype {value.dtype}, which has no canonical encoding") from error
    else:
        raise TypeError(f"entry {name!r} is a {type(value).__name__}, not a tensor or array")
    if array.dtype.name not in SUPPORTED_DTYPES:
        raise TypeError(f"tensor {name!r} has dtype {array.dtype.name}, which has no canonical encoding")
    return array


def canonical_chunks(state_dict: Mapping[str, Any]):
    """Yield the canonical serialization of `state_dict` piece by piece.

    The concatenation of the pieces is the byte string the module docstring
    defines. Raises on anything that has no canonical encoding.
    """
    if not isinstance(state_dict, Mapping):
        raise TypeError(f"state_dict must be a mapping of names to tensors, got {type(state_dict).__name__}")
    if not state_dict:
        raise ValueError("cannot fingerprint an empty state dict")
    for name in state_dict:
        if not isinstance(name, str):
            raise TypeError(f"state_dict keys must be str, got {type(name).__name__}")
    encoded = sorted((name.encode("utf-8"), name) for name in state_dict)

    yield _MAGIC
    yield struct.pack(">I", len(encoded))
    for name_bytes, name in encoded:
        if len(name_bytes) > 0xFFFF:
            raise ValueError(f"tensor name of {len(name_bytes)} bytes is too long")
        array = _as_array(name, state_dict[name])
        dtype = array.dtype.name.encode("ascii")
        if array.ndim > 0xFF:
            raise ValueError(f"tensor {name!r} has too many dimensions")
        # C order, little-endian, whatever the memory layout or host byte order.
        data = np.ascontiguousarray(array, dtype=array.dtype.newbyteorder("<")).tobytes(order="C")
        yield struct.pack(">H", len(name_bytes)) + name_bytes
        yield struct.pack(">B", len(dtype)) + dtype
        yield struct.pack(">B", array.ndim) + b"".join(struct.pack(">Q", int(d)) for d in array.shape)
        yield struct.pack(">Q", len(data))
        yield data


def fingerprint_state_dict(state_dict: Mapping[str, Any]) -> ModelFingerprint:
    """Fingerprint a state dict (names to torch tensors or numpy arrays)."""
    h = hashlib.sha256()
    for chunk in canonical_chunks(state_dict):
        h.update(chunk)
    arrays = [_as_array(name, value) for name, value in state_dict.items()]
    return ModelFingerprint(
        sha256=h.hexdigest(),
        version=FINGERPRINT_VERSION,
        tensors=len(arrays),
        elements=sum(int(a.size) for a in arrays),
        data_bytes=sum(int(a.nbytes) for a in arrays),
    )


def fingerprint_model(model) -> ModelFingerprint:
    """Fingerprint an `nn.Module` through its ``state_dict()``."""
    return fingerprint_state_dict(model.state_dict())


def fingerprint_file(path: Path | str) -> ModelFingerprint:
    """Fingerprint the state dict stored in a `torch.save` file.

    Loaded with ``weights_only=True`` on CPU. The result does not depend on
    the file's name or bytes, only on the tensors in it.
    """
    import torch

    return fingerprint_state_dict(torch.load(Path(path), map_location="cpu", weights_only=True))


if __name__ == "__main__":  # python -m src.crypto.fingerprint <weights.pt>...
    import sys

    for argument in sys.argv[1:]:
        print(fingerprint_file(argument).sha256, argument)
