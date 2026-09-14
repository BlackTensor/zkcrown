"""Post-training quantization: FP16 and static INT8 (P4.4).

The thief converts the model for cheaper deployment and ships it without any
retraining. Three attacks are registered. The swept strength is the bit width,
and each attack accepts only its own.

``ptq_fp16`` (16)
    Every floating-point tensor, weights, BatchNorm parameters and running
    statistics, is cast to float16. This is what ``model.half()`` ships. No
    fusion. Inference runs in float32 on the rounded values, because float16
    convolution is not generally available on CPU. Activations are therefore
    not rounded to float16. At this model's activation scale that difference
    is well inside float16's range and precision, but it is not measured.
``ptq_fused_fp32`` (32)
    The control for INT8. Each conv-BatchNorm-ReLU triple is fused exactly as
    INT8 deployment does, with no rounding. It separates what fusion alone does
    (to the weights the extractor reads) from what quantization adds.
``ptq_int8_static`` (8)
    Standard static post-training quantization, **simulated in plain
    PyTorch** (owner decision, P4.4), rather than run on a quantized backend.
    The eager ``torch.ao.quantization`` API is deprecated and slated for
    removal. The only local engine is oneDNN on ARM, which is not the x86
    backend a typical deployment would use. A simulation is deterministic and
    backend-independent, and it follows the eager-mode recipe step by step:

    1. Fuse conv-BN-ReLU (below).
    2. Calibrate on attacker data: the thief's own images, drawn from the
       attacker holdout (``calibration_images`` of the 5,000, seeded by the
       run seed, no augmentation). Record the per-tensor min and max of the
       input, of every fused conv-ReLU output and of the classifier output, as
       ``MinMaxObserver`` does.
    3. Activations are ``quint8`` [0, 255], per-tensor affine, no reduced
       range: ``scale = (max(max, 0) - min(min, 0)) / 255``,
       ``zero_point = clamp(0 - round(min(min, 0) / scale), 0, 255)``.
    4. Weights are ``qint8`` [-128, 127], per-output-channel symmetric:
       ``scale_c = max|W_c| / 127.5``, zero point 0, as
       ``PerChannelMinMaxObserver`` with ``per_channel_symmetric``.
    5. Biases are rounded to int32 at scale ``input_scale * weight_scale_c``,
       which is what quantized conv and linear kernels do internally.
    6. Inference runs on the integer codes with exact arithmetic and float
       requantization (`FusedRuntime`), so results do not depend on batch
       size. Max-pooling and flattening commute with per-tensor quantization,
       so they need no observer, and dropout is the identity in eval mode.

    What is not simulated: the fixed-point requantization and saturation of
    real integer kernels. On CPUs without VNNI instructions, full-range u8 x s8
    kernels can saturate intermediate int16 sums. oneDNN warns about this, and
    the x86 default qconfig uses 7-bit activations to avoid it. On this
    project's ARM machine, oneDNN's full-range kernels were seen to saturate on
    a seeded untrained model, so they are not a trustworthy reference. With
    7-bit activations, `tests/test_quantize.py` checks the simulation against
    ``torch.ao``'s converted INT8 model: identical qparams, logits within a few
    quantization steps and the same top-1. That check covers a seeded model and
    random inputs only. The scored attack uses the full 8-bit range.

Fusion
------
For conv weight `W`, BN affine pair `gamma`, `beta` and running statistics
`mean`, `var`: ``s = gamma / sqrt(var + eps)``, ``W_f = W * s`` per output
channel, ``b_f = beta - mean * s``. It is computed in float64 and stored in
float32.

What the weight extractor sees
------------------------------
A fused model has no BatchNorm, so the owner's carrier layout is kept by
writing it back into `main_model`'s shapes: conv ``weight = W_f`` (for INT8,
the dequantized integers), BN ``bias = b_f``, ``running_mean = 0``,
``running_var = 1`` and ``weight = sqrt(1 + eps)``. That BN layer then only
adds `b_f`, up to float32 rounding. Weight extraction reads the conv and
classifier weights **as shipped**: fused, and for INT8 dequantized. It does not
undo fusion with the owner's BN statistics (owner decision, P4.4). Fusion
rescales every output channel by `s`, which changes the watermark as well as
the host, so ``ptq_fused_fp32`` is scored too.

Accuracy and the behavioral watermark are scored on the runtime model
(`AttackOutput.runtime_model`): the fused network, with simulated INT8 for
``ptq_int8_static``. `info` holds aggregates only.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Iterable, Mapping

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from src.attacks.harness import AttackOutput, register_attack
from src.models import main_model
from src.watermark.carrier import carrier_layout

QUANTIZE_VERSION = "ptq/v1"
QINT8_RANGE = (-128, 127)
QUINT8_RANGE = (0, 255)
INT32_RANGE = (-(2**31), 2**31 - 1)
BN_STATE_SUFFIXES = ("weight", "bias", "running_mean", "running_var")


# --- plan and fusion ---------------------------------------------------------------------


def _plan(model: nn.Module) -> list[dict]:
    """The forward pass of `main_model` as fusable steps. Refuses any other shape."""
    steps: list[dict] = []
    features = list(model.features.named_children())
    i = 0
    while i < len(features):
        name, module = features[i]
        if isinstance(module, nn.Conv2d):
            names = [n for n, _ in features[i : i + 3]]
            kinds = [type(m) for _, m in features[i : i + 3]]
            if kinds != [nn.Conv2d, nn.BatchNorm2d, nn.ReLU]:
                raise ValueError(f"features.{name}: expected conv-BN-ReLU, got {[k.__name__ for k in kinds]}")
            steps.append({"kind": "conv_relu", "conv": f"features.{names[0]}", "bn": f"features.{names[1]}",
                          "stride": module.stride, "padding": module.padding, "dilation": module.dilation,
                          "groups": module.groups, "eps": features[i + 1][1].eps})
            i += 3
        elif isinstance(module, nn.MaxPool2d):
            steps.append({"kind": "maxpool", "kernel_size": module.kernel_size, "stride": module.stride,
                          "padding": module.padding, "dilation": module.dilation, "ceil_mode": module.ceil_mode})
            i += 1
        else:
            raise ValueError(f"features.{name}: unsupported module {type(module).__name__}")
    for name, module in model.classifier.named_children():
        if isinstance(module, nn.Flatten):
            steps.append({"kind": "flatten"})
        elif isinstance(module, nn.Dropout):
            continue  # identity in eval mode
        elif isinstance(module, nn.Linear):
            steps.append({"kind": "linear", "linear": f"classifier.{name}"})
        else:
            raise ValueError(f"classifier.{name}: unsupported module {type(module).__name__}")
    return steps


def fuse_conv_bn(state_dict: Mapping[str, torch.Tensor], arch: Mapping) -> tuple[list[dict], dict]:
    """Fused float32 weights for every weighted step, and per-layer aggregate facts about the BN scale."""
    model = main_model(**dict(arch))
    carrier_layout(model).check(state_dict)
    steps = _plan(model)
    info = {}
    for step in steps:
        if step["kind"] == "conv_relu":
            conv, bn = step["conv"], step["bn"]
            w = state_dict[f"{conv}.weight"].detach().cpu().double()
            gamma, beta, mean, var = (state_dict[f"{bn}.{k}"].detach().cpu().double() for k in BN_STATE_SUFFIXES)
            scale = gamma / torch.sqrt(var + step["eps"])
            step["weight"] = (w * scale.view(-1, 1, 1, 1)).float()
            step["bias"] = (beta - mean * scale).float()
            info[conv] = {"bn_scale_min": float(scale.min()), "bn_scale_max": float(scale.max()),
                          "bn_scale_negative": int((scale < 0).sum())}
        elif step["kind"] == "linear":
            step["weight"] = state_dict[f"{step['linear']}.weight"].detach().cpu().float().clone()
            step["bias"] = state_dict[f"{step['linear']}.bias"].detach().cpu().float().clone()
    return steps, info


def fused_state_dict(source: Mapping[str, torch.Tensor], steps: list[dict]) -> dict[str, torch.Tensor]:
    """The fused (or quantized) weights written back into `main_model`'s shapes. See the module docstring."""
    out = {name: tensor.detach().cpu().clone() for name, tensor in source.items()}
    for step in steps:
        if step["kind"] == "conv_relu":
            conv, bn = step["conv"], step["bn"]
            out[f"{conv}.weight"] = step["weight"].clone()
            channels = step["bias"].numel()
            out[f"{bn}.weight"] = torch.full((channels,), math.sqrt(1.0 + step["eps"]), dtype=torch.float32)
            out[f"{bn}.bias"] = step["bias"].clone()
            out[f"{bn}.running_mean"] = torch.zeros(channels, dtype=torch.float32)
            out[f"{bn}.running_var"] = torch.ones(channels, dtype=torch.float32)
        elif step["kind"] == "linear":
            out[f"{step['linear']}.weight"] = step["weight"].clone()
            out[f"{step['linear']}.bias"] = step["bias"].clone()
    return out


# --- runtime --------------------------------------------------------------------------------


class FusedRuntime(nn.Module):
    """The fused network, in float32 or as simulated INT8.

    Without ``activation_qparams`` the steps' float weights and biases are
    used, and `observe` records activation ranges for calibration.

    With ``activation_qparams`` (observation point ``"input"``, then
    ``"step<i>"`` for each conv-ReLU and the linear, mapped to ``(scale,
    zero_point, quant_min, quant_max)``) the steps must carry ``weight_int``,
    ``weight_scale`` and ``bias_int``. Inference then works on integer codes as
    an integer kernel does. Each conv or linear sums ``(q - zero_point) *
    weight_int`` plus ``bias_int`` in float64. Every partial sum is an integer
    far below 2**53, so it is exact whatever the summation order, and the
    result does not depend on batch size or backend. The sum is then
    requantized: ``clamp(round(sum * in_scale * weight_scale / out_scale) +
    out_zero_point)``, with the lower clamp at the zero point after a ReLU.
    The output logits are dequantized.
    """

    def __init__(self, steps: list[dict], activation_qparams: Mapping[str, tuple[float, int, int, int]] | None = None):
        super().__init__()
        self.activation_qparams = dict(activation_qparams or {})
        self.kinds = [step["kind"] for step in steps]
        tensors = ("weight", "bias", "weight_int", "weight_scale", "bias_int")
        self.attrs = [{k: v for k, v in step.items() if k not in tensors} for step in steps]
        wanted = ("weight_int", "weight_scale", "bias_int") if self.activation_qparams else ("weight", "bias")
        for i, step in enumerate(steps):
            if "weight" in step:
                for name in wanted:
                    self.register_buffer(f"{name}{i}", step[name].clone())
        self._observed: dict[str, list[float]] | None = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._forward_int8(x) if self.activation_qparams else self._forward_float(x)

    def _observe(self, name: str, x: torch.Tensor) -> torch.Tensor:
        if self._observed is not None:
            low, high = float(x.min()), float(x.max())
            seen = self._observed.setdefault(name, [low, high])
            seen[0], seen[1] = min(seen[0], low), max(seen[1], high)
        return x

    @staticmethod
    def _pool(x: torch.Tensor, attrs: dict) -> torch.Tensor:
        return F.max_pool2d(x, attrs["kernel_size"], attrs["stride"], attrs["padding"], attrs["dilation"], attrs["ceil_mode"])

    def _forward_float(self, x: torch.Tensor) -> torch.Tensor:
        x = self._observe("input", x)
        for i, (kind, attrs) in enumerate(zip(self.kinds, self.attrs)):
            if kind == "conv_relu":
                x = F.conv2d(x, getattr(self, f"weight{i}"), getattr(self, f"bias{i}"), attrs["stride"],
                             attrs["padding"], attrs["dilation"], attrs["groups"])
                x = self._observe(f"step{i}", F.relu(x))
            elif kind == "maxpool":
                x = self._pool(x, attrs)
            elif kind == "flatten":
                x = x.flatten(1)
            else:
                x = self._observe(f"step{i}", F.linear(x, getattr(self, f"weight{i}"), getattr(self, f"bias{i}")))
        return x

    def _forward_int8(self, x: torch.Tensor) -> torch.Tensor:
        scale, zero_point, qmin, qmax = self.activation_qparams["input"]
        q = torch.clamp(torch.round(x.double() / scale) + zero_point, qmin, qmax)
        for i, (kind, attrs) in enumerate(zip(self.kinds, self.attrs)):
            if kind in ("conv_relu", "linear"):
                w, b = getattr(self, f"weight_int{i}"), getattr(self, f"bias_int{i}")
                if kind == "conv_relu":
                    total = F.conv2d(q - zero_point, w, b, attrs["stride"], attrs["padding"], attrs["dilation"], attrs["groups"])
                    shape = (1, -1, 1, 1)
                else:
                    total = F.linear(q - zero_point, w, b)
                    shape = (1, -1)
                out_scale, out_zero, out_min, out_max = self.activation_qparams[f"step{i}"]
                multiplier = (scale * getattr(self, f"weight_scale{i}") / out_scale).view(shape)
                low = max(out_zero, out_min) if kind == "conv_relu" else out_min
                q = torch.clamp(torch.round(total * multiplier) + out_zero, low, out_max)
                scale, zero_point = out_scale, out_zero
            elif kind == "maxpool":
                q = self._pool(q, attrs)
            else:
                q = q.flatten(1)
        return ((q - zero_point) * scale).float()

    @torch.no_grad()
    def observe(self, batches: Iterable[torch.Tensor]) -> dict[str, tuple[float, float]]:
        """Min and max at every observation point over `batches`, with float activations."""
        if self.activation_qparams:
            raise RuntimeError("observe() runs on the float runtime only")
        self.eval()
        self._observed = {}
        try:
            for batch in batches:
                self(batch)
            return {name: (low, high) for name, (low, high) in self._observed.items()}
        finally:
            self._observed = None


# --- quantization parameters --------------------------------------------------------------


def affine_qparams(low: float, high: float, reduce_range: bool = False) -> tuple[float, int, int, int]:
    """`MinMaxObserver` qparams ``(scale, zero_point, quant_min, quant_max)`` for per-tensor affine quint8.

    The attack uses the full 8-bit range. ``reduce_range=True`` gives the
    7-bit range [0, 127] of the x86 default qconfig; the tests use it to
    compare with real kernels that would otherwise saturate.
    """
    qmin, qmax = (0, 127) if reduce_range else QUINT8_RANGE
    low_t = torch.tensor(min(low, 0.0), dtype=torch.float32)
    high_t = torch.tensor(max(high, 0.0), dtype=torch.float32)
    scale = torch.clamp((high_t - low_t) / float(qmax - qmin), min=torch.finfo(torch.float32).eps)
    zero_point = int(torch.clamp(qmin - torch.round(low_t / scale).to(torch.int), qmin, qmax))
    return float(scale), zero_point, qmin, qmax


def symmetric_channel_scales(weight: torch.Tensor) -> torch.Tensor:
    """`PerChannelMinMaxObserver` scales for per-channel symmetric qint8 on axis 0."""
    qmin, qmax = QINT8_RANGE
    flat = weight.detach().float().reshape(weight.shape[0], -1)
    low = torch.clamp(flat.min(dim=1).values, max=0.0)
    high = torch.clamp(flat.max(dim=1).values, min=0.0)
    bound = torch.maximum(-low, high)
    return torch.clamp(bound / ((qmax - qmin) / 2), min=torch.finfo(torch.float32).eps)


def quantize_steps(steps: list[dict], observed: Mapping[str, tuple[float, float]], reduce_range: bool = False
                   ) -> tuple[list[dict], dict, dict]:
    """INT8 steps, activation qparams, and aggregate info.

    Each weighted step gets ``weight_int`` (qint8 codes), ``weight_scale`` (per
    channel) and ``bias_int`` (int32 codes at ``input_scale * weight_scale``),
    all float64, and its ``weight`` and ``bias`` are replaced by their
    dequantized float32 values: the weights as shipped.
    """
    activation = {name: affine_qparams(*observed[name], reduce_range=reduce_range) for name in observed}
    quantized, per_layer = [], {}
    input_point = "input"
    for i, step in enumerate(steps):
        step = dict(step)
        if "weight" in step:
            weight, bias = step["weight"], step["bias"]
            scales = symmetric_channel_scales(weight).double()
            view = (-1,) + (1,) * (weight.dim() - 1)
            weight_int = torch.clamp(torch.round(weight.double() / scales.view(view)), *QINT8_RANGE)
            bias_scale = activation[input_point][0] * scales
            bias_int = torch.clamp(torch.round(bias.double() / bias_scale), *INT32_RANGE)
            weight_q, bias_q = (weight_int * scales.view(view)).float(), (bias_int * bias_scale).float()
            step.update(weight_int=weight_int, weight_scale=scales, bias_int=bias_int, weight=weight_q, bias=bias_q)
            name = step.get("conv") or step["linear"]
            per_layer[name] = {
                "weight_scale_min": float(scales.min()), "weight_scale_max": float(scales.max()),
                "weight_rounding_rel_error": float(torch.linalg.vector_norm((weight_q - weight).double())
                                                   / torch.linalg.vector_norm(weight.double())),
                "int8_zero_fraction": float((weight_int == 0).double().mean()),
                "bias_rounding_max_abs": float((bias_q.double() - bias.double()).abs().max()),
                "input_scale": activation[input_point][0], "input_zero_point": activation[input_point][1],
                "output_scale": activation[f"step{i}"][0], "output_zero_point": activation[f"step{i}"][1],
                "output_range": [observed[f"step{i}"][0], observed[f"step{i}"][1]],
            }
            input_point = f"step{i}"
        quantized.append(step)
    return quantized, activation, per_layer


# --- attacks ----------------------------------------------------------------------------------


def _carrier_change(source: Mapping[str, torch.Tensor], out: Mapping[str, torch.Tensor], arch: Mapping) -> dict:
    layout = carrier_layout(main_model(**dict(arch)))
    before, after = layout.flatten(source), layout.flatten(out)
    return {"carrier_rel_change": float(np.linalg.norm(after - before) / np.linalg.norm(before)),
            "carrier_zero_fraction": float(np.mean(after == 0))}


def _check_bits(strength: float, expected: int) -> None:
    if strength != expected:
        raise ValueError(f"this attack's strength is its bit width and must be {expected}, got {strength}")


def fp16_cast(state_dict: Mapping[str, torch.Tensor], arch: Mapping) -> tuple[dict, dict]:
    """Every floating tensor cast to float16. Refuses if anything overflows."""
    out, cast = {}, 0
    for name, tensor in state_dict.items():
        if tensor.is_floating_point():
            out[name] = tensor.detach().cpu().to(torch.float16)
            if not torch.isfinite(out[name]).all():
                raise ValueError(f"{name} overflows float16")
            cast += 1
        else:
            out[name] = tensor.detach().cpu().clone()
    layout = carrier_layout(main_model(**dict(arch)))
    source64, cast64 = layout.flatten(state_dict), layout.flatten(out)
    info = {
        "version": QUANTIZE_VERSION, "bits": 16, "scheme": "float16 cast of every floating tensor, no fusion",
        "tensors_cast": cast,
        "carrier_rounding_max_abs": float(np.abs(cast64 - source64).max()),
        "carrier_underflow_to_zero": int(np.sum((cast64 == 0) & (source64 != 0))),
        **_carrier_change(state_dict, out, arch),
        "inference": "float32 on the float16-rounded tensors",
        "fused": False, "fine_tuned": False, "calibration_images": 0,
    }
    return out, info


def fused_fp32(state_dict: Mapping[str, torch.Tensor], arch: Mapping) -> tuple[dict, FusedRuntime, dict]:
    steps, bn_info = fuse_conv_bn(state_dict, arch)
    out = fused_state_dict(state_dict, steps)
    info = {
        "version": QUANTIZE_VERSION, "bits": 32, "scheme": "conv-BN-ReLU fusion, float32, no rounding (control)",
        "per_layer": bn_info, **_carrier_change(state_dict, out, arch),
        "fused": True, "fine_tuned": False, "calibration_images": 0,
    }
    return out, FusedRuntime(steps), info


def static_int8(state_dict: Mapping[str, torch.Tensor], arch: Mapping, calibration: Iterable[torch.Tensor],
                reduce_range: bool = False) -> tuple[dict, FusedRuntime, dict]:
    """Simulated static INT8 PTQ, calibrated on `calibration` batches (normalised images)."""
    steps, bn_info = fuse_conv_bn(state_dict, arch)
    batches = list(calibration)
    if not batches or sum(len(b) for b in batches) == 0:
        raise ValueError("static INT8 needs at least one calibration image")
    float_runtime = FusedRuntime(steps)
    observed = float_runtime.observe(batches)
    q_steps, activation, per_layer = quantize_steps(steps, observed, reduce_range)
    runtime = FusedRuntime(q_steps, activation).eval()
    with torch.no_grad():
        agree = sum(int((runtime(b).argmax(1) == float_runtime(b).argmax(1)).sum()) for b in batches)
    for name, facts in bn_info.items():
        per_layer[name].update(facts)
    out = fused_state_dict(state_dict, q_steps)
    n = sum(len(b) for b in batches)
    info = {
        "version": QUANTIZE_VERSION, "bits": 8,
        "scheme": ("simulated static PTQ: conv-BN-ReLU fusion; qint8 per-channel symmetric weights; quint8 "
                   "per-tensor affine activations, min/max calibration, no reduced range; int32 biases"),
        "calibration_images": n,
        "calibration_top1_agreement_with_fused_fp32": agree / n,
        "input_qparams": list(activation["input"][:2]),
        "activation_range": list(activation["input"][2:]),
        "per_layer": per_layer,
        **_carrier_change(state_dict, out, arch),
        "not_simulated": "fixed-point requantization and saturation of real integer kernels",
        "fused": True, "fine_tuned": False,
    }
    return out, runtime, info


def attacker_calibration_batches(data_root, count: int, seed: int, batch_size: int = 250) -> tuple[list[torch.Tensor], dict]:
    """`count` attacker-holdout images, chosen by `seed`, normalised, no augmentation."""
    from src.data.cifar10 import cifar10_datasets

    if not isinstance(count, int) or isinstance(count, bool) or count < 1:
        raise ValueError(f"calibration_images must be a positive int, got {count!r}")
    holdout = cifar10_datasets(data_root, augment=False, download=False)["holdout"]
    if count > len(holdout):
        raise ValueError(f"calibration_images {count} exceeds the {len(holdout)}-image attacker holdout")
    picks = sorted(torch.randperm(len(holdout), generator=torch.Generator().manual_seed(seed))[:count].tolist())
    images = torch.stack([holdout[i][0] for i in picks])
    global_indices = [int(holdout.indices[i]) for i in picks]
    digest = hashlib.sha256(json.dumps(global_indices).encode()).hexdigest()
    return list(images.split(batch_size)), {"source": "attacker holdout (5,000 CIFAR-10 train images)", "seed": seed,
                                            "indices_sha256": digest}


@register_attack(
    "ptq_fp16",
    strength="bit width; must be 16",
    description="Half-precision cast of every floating tensor, no fusion, no retraining (P4.4).",
)
def ptq_fp16(state_dict, arch, strength, params, context) -> AttackOutput:
    _check_bits(strength, 16)
    if params:
        raise ValueError(f"ptq_fp16 takes no params, got {sorted(params)}")
    state, info = fp16_cast(state_dict, arch)
    return AttackOutput(state_dict=state, arch=arch, info=info)


@register_attack(
    "ptq_fused_fp32",
    strength="bit width; must be 32 (float, the fusion-only control)",
    description="Conv-BN-ReLU fusion in float32 with no rounding: the control for static INT8 (P4.4).",
)
def ptq_fused_fp32(state_dict, arch, strength, params, context) -> AttackOutput:
    _check_bits(strength, 32)
    if params:
        raise ValueError(f"ptq_fused_fp32 takes no params, got {sorted(params)}")
    state, runtime, info = fused_fp32(state_dict, arch)
    return AttackOutput(state_dict=state, arch=arch, info=info, runtime_model=runtime)


@register_attack(
    "ptq_int8_static",
    strength="bit width; must be 8",
    description="Simulated static INT8 PTQ (fused, per-channel weights, calibrated activations), no retraining (P4.4).",
)
def ptq_int8_static(state_dict, arch, strength, params, context) -> AttackOutput:
    _check_bits(strength, 8)
    unknown = set(params) - {"calibration_images"}
    if unknown or "calibration_images" not in params:
        raise ValueError(f"ptq_int8_static takes exactly 'calibration_images', got {sorted(params)}")
    batches, calibration = attacker_calibration_batches(context.data_root, params["calibration_images"], context.seed)
    state, runtime, info = static_int8(state_dict, arch, batches)
    info["calibration"] = calibration
    return AttackOutput(state_dict=state, arch=arch, info=info, runtime_model=runtime)
