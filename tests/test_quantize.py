"""Tests for P4.4: post-training quantization (FP16, fused FP32, simulated static INT8).

Seeded untrained models with non-trivial BatchNorm state, random inputs and a
TEST key; no trained weights, no `K`, no CIFAR-10. Covered: fusion is exact,
the fused weights load back into `main_model`, the INT8 grids and qparam
formulas, batch independence and determinism, a cross-check against
``torch.ao``'s real INT8 kernels (7-bit activations, where available), the
FP16 cast, validation, the harness's runtime-model path, apply mode refusing
runtime models, and the configs.

All keys and owner ids here are TEST data, public by construction.
"""

from __future__ import annotations

import copy
import importlib
import json
import warnings
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
from torch import nn  # noqa: E402
from torch.utils.data import DataLoader, TensorDataset  # noqa: E402

from src.attacks import evaluation, harness, quantize  # noqa: E402
from src.attacks.harness import AttackConfig, AttackContext, apply_attack, expand_config, load_model  # noqa: E402
from src.models import main_model  # noqa: E402
from src.utils.seeding import set_seed  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
ARCH = {"width": 32}
TEST_KEY = bytes(range(32, 64))
"""TEST KEY ONLY."""
CONVS = ["features.0", "features.3", "features.7", "features.10", "features.14", "features.17"]


@pytest.fixture(scope="module")
def model():
    set_seed(11)
    m = main_model(**ARCH).eval()
    with torch.no_grad():
        for mod in m.modules():
            if isinstance(mod, nn.BatchNorm2d):
                mod.weight.uniform_(0.5, 1.5)
                mod.bias.uniform_(-0.5, 0.5)
                mod.running_mean.uniform_(-0.5, 0.5)
                mod.running_var.uniform_(0.5, 2)
    return m


@pytest.fixture(scope="module")
def state(model):
    return model.state_dict()


@pytest.fixture(scope="module")
def data():
    g = torch.Generator().manual_seed(3)
    return {"calib": [torch.randn(100, 3, 32, 32, generator=g) for _ in range(3)],
            "test": torch.randn(200, 3, 32, 32, generator=g)}


@pytest.fixture(scope="module")
def int8(state, data):
    return quantize.static_int8(state, ARCH, data["calib"])


# --- fusion ------------------------------------------------------------------------------------


def test_fused_runtime_matches_the_source_model(model, state, data):
    _, runtime, info = quantize.fused_fp32(state, ARCH)
    with torch.no_grad():
        a, b = model(data["test"]), runtime(data["test"])
    assert torch.allclose(a, b, atol=1e-5, rtol=1e-5)
    assert torch.equal(a.argmax(1), b.argmax(1))
    assert set(info["per_layer"]) == set(CONVS)


def test_fused_state_dict_loads_into_main_model_and_matches_the_runtime(state, data):
    fused, runtime, info = quantize.fused_fp32(state, ARCH)
    loaded = load_model(fused, ARCH)
    with torch.no_grad():
        assert torch.allclose(loaded(data["test"]), runtime(data["test"]), atol=1e-5, rtol=1e-5)
    for conv, bn in zip(CONVS, ["features.1", "features.4", "features.8", "features.11", "features.15", "features.18"]):
        scale = state[f"{bn}.weight"].double() / torch.sqrt(state[f"{bn}.running_var"].double() + 1e-5)
        expected = (state[f"{conv}.weight"].double() * scale.view(-1, 1, 1, 1)).float()
        assert torch.equal(fused[f"{conv}.weight"], expected)
        assert torch.equal(fused[f"{bn}.running_mean"], torch.zeros_like(scale, dtype=torch.float32))
    assert torch.equal(fused["classifier.2.weight"], state["classifier.2.weight"])
    assert info["carrier_rel_change"] > 0.01  # fusion rescales channels, so the carrier changes


def test_plan_refuses_an_unsupported_shape():
    class Odd(nn.Module):
        def __init__(self):
            super().__init__()
            self.features = nn.Sequential(nn.Conv2d(3, 4, 3), nn.ReLU())
            self.classifier = nn.Sequential(nn.Flatten(), nn.Linear(4, 10))

    with pytest.raises(ValueError):
        quantize._plan(Odd())


# --- INT8 -------------------------------------------------------------------------------------


def test_affine_qparams_formula():
    assert quantize.affine_qparams(0.0, 2.55) == pytest.approx((0.01, 0, 0, 255), rel=1e-6)
    scale, zp, qmin, qmax = quantize.affine_qparams(-1.0, 3.0)
    assert scale == pytest.approx(4 / 255, rel=1e-6) and zp == round(1 / (4 / 255)) and (qmin, qmax) == (0, 255)
    scale, zp, _, qmax = quantize.affine_qparams(0.5, 1.27, reduce_range=True)  # min is widened to 0
    assert scale == pytest.approx(0.01, rel=1e-6) and zp == 0 and qmax == 127


def test_symmetric_channel_scales_formula():
    w = torch.tensor([[1.0, -2.55], [0.0, 0.0], [0.3, 0.1]])
    scales = quantize.symmetric_channel_scales(w)
    assert scales[0] == pytest.approx(2.55 / 127.5)
    assert scales[1] == torch.finfo(torch.float32).eps
    assert scales[2] == pytest.approx(0.3 / 127.5)


def test_int8_codes_lie_on_their_grids_and_match_the_shipped_weights(state, int8):
    fused, runtime, info = int8
    weighted = [(i, runtime.attrs[i].get("conv") or runtime.attrs[i]["linear"])
                for i, kind in enumerate(runtime.kinds) if kind in ("conv_relu", "linear")]
    assert [name for _, name in weighted] == CONVS + ["classifier.2"]
    for i, name in weighted:
        w_int, scale, b_int = (getattr(runtime, f"{k}{i}") for k in ("weight_int", "weight_scale", "bias_int"))
        assert torch.equal(w_int, torch.round(w_int)) and w_int.min() >= -128 and w_int.max() <= 127
        assert torch.equal(b_int, torch.round(b_int))
        view = (-1,) + (1,) * (w_int.dim() - 1)
        assert torch.equal(fused[f"{name}.weight"], (w_int * scale.view(view)).float())
        layer = info["per_layer"][name]
        assert scale.min() == pytest.approx(layer["weight_scale_min"]) and scale.max() == pytest.approx(layer["weight_scale_max"])
        assert 0 < layer["weight_rounding_rel_error"] < 0.05
    # ReLU outputs never go below zero, so their zero point is 0
    assert all(info["per_layer"][c]["output_zero_point"] == 0 for c in CONVS)


def _dequantized_reference(activation, q_steps, x):
    """Independent float64 reference: dequantized weights and int32 biases, quantize-dequantize at each point."""
    def fq(name, t):
        scale, zp, qmin, qmax = activation[name]
        return (torch.clamp(torch.round(t / scale) + zp, qmin, qmax) - zp) * scale

    x, previous = fq("input", x.double()), "input"
    for i, step in enumerate(q_steps):
        if step["kind"] in ("conv_relu", "linear"):
            view = (-1,) + (1,) * (step["weight_int"].dim() - 1)
            w = step["weight_int"] * step["weight_scale"].view(view)
            b = step["bias_int"] * activation[previous][0] * step["weight_scale"]
            if step["kind"] == "conv_relu":
                x = fq(f"step{i}", torch.relu(nn.functional.conv2d(x, w, b, step["stride"], step["padding"])))
            else:
                x = fq(f"step{i}", nn.functional.linear(x, w, b))
            previous = f"step{i}"
        elif step["kind"] == "maxpool":
            x = nn.functional.max_pool2d(x, step["kernel_size"], step["stride"])
        else:
            x = x.flatten(1)
    return x


def test_int8_runtime_matches_an_independent_dequantized_reference(state, data):
    """Same quantized model computed a second way, in float64: the integer path must give the same codes."""
    steps, _ = quantize.fuse_conv_bn(state, ARCH)
    observed = quantize.FusedRuntime(steps).observe(data["calib"])
    q_steps, activation, per_layer = quantize.quantize_steps(steps, observed)
    runtime = quantize.FusedRuntime(q_steps, activation)
    with torch.no_grad():
        ours = runtime(data["test"]).double()
        reference = _dequantized_reference(activation, q_steps, data["test"])
    step = per_layer["classifier.2"]["output_scale"]
    assert float((ours - reference).abs().max()) <= 1e-4 * step  # float32 output conversion only



def test_int8_is_close_to_fused_fp32_and_reports_agreement(state, data, int8):
    _, runtime, info = int8
    _, float_runtime, _ = quantize.fused_fp32(state, ARCH)
    with torch.no_grad():
        agree = float((runtime(data["test"]).argmax(1) == float_runtime(data["test"]).argmax(1)).float().mean())
        calib_agree = sum(int((runtime(b).argmax(1) == float_runtime(b).argmax(1)).sum()) for b in data["calib"]) / 300
    assert agree > 0.9
    assert info["calibration_top1_agreement_with_fused_fp32"] == calib_agree
    assert info["calibration_images"] == 300 and info["activation_range"] == [0, 255]


def test_int8_is_exactly_batch_independent_and_deterministic(state, data, int8):
    _, runtime, info = int8
    x = data["test"][:32]
    with torch.no_grad():
        whole = runtime(x)
        pieces = torch.cat([runtime(x[i : i + 1]) for i in range(len(x))])
        halves = torch.cat([runtime(x[:13]), runtime(x[13:])])
    assert torch.equal(whole, pieces) and torch.equal(whole, halves)
    fused2, runtime2, info2 = quantize.static_int8(state, ARCH, data["calib"])
    assert all(torch.equal(int8[0][k], fused2[k]) for k in fused2)
    assert info == info2
    with torch.no_grad():
        assert torch.equal(runtime2(x), whole)



def test_int8_fused_state_dict_loads_and_is_close_to_the_runtime(int8, data):
    fused, runtime, _ = int8
    loaded = load_model(fused, ARCH)  # float activations on the quantized weights
    with torch.no_grad():
        a, b = loaded(data["test"]), runtime(data["test"])
    assert float((a - b).abs().max()) < 0.2 * float(a.abs().max())


def test_int8_needs_calibration_images(state):
    with pytest.raises(ValueError):
        quantize.static_int8(state, ARCH, [])


def test_simulation_matches_torch_ao_kernels_with_reduced_range(model, state, data):
    """Real INT8 kernels vs the simulation, on 7-bit activations so the kernels cannot saturate."""
    tq = pytest.importorskip("torch.ao.quantization")
    engines = [e for e in ("x86", "fbgemm", "onednn", "qnnpack") if e in torch.backends.quantized.supported_engines]
    if not engines:
        pytest.skip("no quantized engine")
    previous = torch.backends.quantized.engine
    torch.backends.quantized.engine = engines[0]
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            class Wrap(nn.Module):
                def __init__(self, m):
                    super().__init__()
                    self.q, self.m, self.d = tq.QuantStub(), m, tq.DeQuantStub()

                def forward(self, x):
                    return self.d(self.m(self.q(x)))

            fused_model = tq.fuse_modules(copy.deepcopy(model).eval(),
                                          [[f"features.{i}", f"features.{i + 1}", f"features.{i + 2}"] for i in (0, 3, 7, 10, 14, 17)])
            wrapped = Wrap(fused_model).eval()
            wrapped.qconfig = tq.QConfig(activation=tq.MinMaxObserver.with_args(dtype=torch.quint8, reduce_range=True),
                                         weight=tq.default_per_channel_weight_observer)
            prepared = tq.prepare(wrapped)
            with torch.no_grad():
                for batch in data["calib"]:
                    prepared(batch)
            converted = tq.convert(prepared)
            with torch.no_grad():
                real = converted(data["test"])
    finally:
        torch.backends.quantized.engine = previous
    _, runtime, info = quantize.static_int8(state, ARCH, data["calib"], reduce_range=True)
    assert converted.q.scale.item() == pytest.approx(info["input_qparams"][0], rel=1e-6)
    assert converted.q.zero_point.item() == info["input_qparams"][1]
    for index, conv in zip((0, 3, 7, 10, 14, 17), CONVS):
        layer = converted.m.features[index]
        assert layer.scale == pytest.approx(info["per_layer"][conv]["output_scale"], rel=1e-6)
        assert float(layer.weight().q_per_channel_scales().max()) == pytest.approx(info["per_layer"][conv]["weight_scale_max"], rel=1e-5)
    step = info["per_layer"]["classifier.2"]["output_scale"]
    assert converted.m.classifier[2].scale == pytest.approx(step, rel=1e-6)
    with torch.no_grad():
        simulated = runtime(data["test"])
    assert float((simulated - real).abs().max()) <= 6 * step
    assert float((simulated - real).abs().mean()) <= 1.5 * step
    assert torch.equal(simulated.argmax(1), real.argmax(1))


# --- FP16 ---------------------------------------------------------------------------------------


def test_fp16_casts_every_floating_tensor(state, data):
    out, info = quantize.fp16_cast(state, ARCH)
    for name, tensor in state.items():
        if tensor.is_floating_point():
            assert out[name].dtype == torch.float16 and torch.equal(out[name], tensor.half())
        else:
            assert torch.equal(out[name], tensor)
    assert info["tensors_cast"] == sum(1 for t in state.values() if t.is_floating_point())
    assert 0 < info["carrier_rounding_max_abs"] < 1e-3
    loaded = load_model(out, ARCH)
    with torch.no_grad():
        assert torch.allclose(loaded(data["test"]), load_model(state, ARCH)(data["test"]), atol=1e-2)


def test_fp16_refuses_overflow(state):
    bad = dict(state)
    bad["features.0.weight"] = state["features.0.weight"].clone()
    bad["features.0.weight"][0, 0, 0, 0] = 1e6
    with pytest.raises(ValueError):
        quantize.fp16_cast(bad, ARCH)


# --- through the harness ---------------------------------------------------------------------------


@pytest.mark.parametrize("name,bits", [("ptq_fp16", 16), ("ptq_fused_fp32", 32), ("ptq_int8_static", 8)])
def test_strength_must_be_the_bit_width(state, tmp_path, name, bits):
    context = AttackContext(torch.device("cpu"), 1, tmp_path)
    params = {"calibration_images": 10} if bits == 8 else {}
    with pytest.raises(ValueError):
        apply_attack(AttackConfig(name, 8 if bits != 8 else 16, params), state, ARCH, context)
    if bits != 8:
        with pytest.raises(ValueError):
            apply_attack(AttackConfig(name, bits, {"calibration_images": 10}), state, ARCH, context)


@pytest.mark.parametrize("params", [{}, {"calibration_images": 10, "observer": "histogram"}])
def test_int8_params_are_validated(state, tmp_path, params):
    with pytest.raises(ValueError):
        apply_attack(AttackConfig("ptq_int8_static", 8, params), state, ARCH, AttackContext(torch.device("cpu"), 1, tmp_path))


@pytest.mark.parametrize("count", [0, -1, 1.5, True])
def test_calibration_count_is_validated(tmp_path, count):
    with pytest.raises(ValueError):
        quantize.attacker_calibration_batches(tmp_path, count, 1)


def test_all_three_run_through_the_harness(state, data, tmp_path, monkeypatch):
    calls = []

    def fake_calibration(root, count, seed, batch_size=250):
        calls.append((count, seed))
        return data["calib"], {"source": "TEST", "seed": seed, "indices_sha256": "0" * 64}

    monkeypatch.setattr(quantize, "attacker_calibration_batches", fake_calibration)
    context = AttackContext(torch.device("cpu"), 5, tmp_path)
    fp16 = apply_attack(AttackConfig("ptq_fp16", 16), state, ARCH, context)
    fused = apply_attack(AttackConfig("ptq_fused_fp32", 32), state, ARCH, context)
    int8 = apply_attack(AttackConfig("ptq_int8_static", 8, {"calibration_images": 300}), state, ARCH, context)
    assert calls == [(300, 5)]
    assert fp16.runtime_model is None
    assert isinstance(fused.runtime_model, quantize.FusedRuntime) and not fused.runtime_model.activation_qparams
    assert isinstance(int8.runtime_model, quantize.FusedRuntime) and int8.runtime_model.activation_qparams
    for out in (fp16, fused, int8):
        json.dumps(out.info, allow_nan=False)
        _assert_short_lists(out.info)
        assert out.info["fine_tuned"] is False


def _assert_short_lists(value, path="info"):
    if isinstance(value, dict):
        for key, item in value.items():
            _assert_short_lists(item, f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        assert len(value) <= 8, path


def test_evaluate_attacked_scores_the_runtime_model_but_extracts_from_the_state_dict(state):
    rng = np.random.default_rng(0)
    images = rng.integers(0, 256, size=(200, 32, 32, 3), dtype=np.uint8)
    labels = rng.integers(0, 10, size=200)
    material = evaluation.derive_owner_material(TEST_KEY, images, labels, list(range(200)), owner_id="zk-crown test owner",
                                                key_kind="test", n=20)

    class AlwaysClass3(nn.Module):
        def forward(self, x):
            return nn.functional.one_hot(torch.full((len(x),), 3), 10).float()

    g = torch.Generator().manual_seed(1)
    labels_t = torch.full((40,), 3)
    labels_t[:10] = 4
    loader = DataLoader(TensorDataset(torch.randn(40, 3, 32, 32, generator=g), labels_t), batch_size=16)
    reference = [True] * 40
    with_runtime = evaluation.evaluate_attacked(state, ARCH, material, loader, reference, torch.device("cpu"),
                                                runtime_model=AlwaysClass3())
    assert with_runtime["clean_accuracy"]["correct"] == 30
    expected_weight = evaluation.evaluate_weight(state, material)
    assert with_runtime["weight"]["correlation"] == expected_weight["correlation"]
    plain = evaluation.evaluate_attacked(state, ARCH, material, loader, reference, torch.device("cpu"))
    assert plain["weight"]["correlation"] == expected_weight["correlation"]


def test_apply_mode_refuses_runtime_model_attacks(state, tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(REPO_ROOT / "experiments"))
    script = importlib.import_module("run_attack_suite")
    monkeypatch.setattr(script, "load_source", lambda name, path: (state, {"name": name, "task": "TEST",
                                                                           "weights_sha256": "0" * 64, "arch": ARCH}))
    with pytest.raises(SystemExit):
        script.main(["apply", "--attack", "ptq_fused_fp32", "--strength", "32", "--task", "P4.4",
                     "--out-dir", str(tmp_path)])
    assert not list(tmp_path.iterdir())


def test_p4_4_configs_are_the_declared_runs():
    expected = {"p4.4_ptq_fp16.json": ("ptq_fp16", [16.0], {}),
                "p4.4_ptq_fused_fp32.json": ("ptq_fused_fp32", [32.0], {}),
                "p4.4_ptq_int8_static.json": ("ptq_int8_static", [8.0], {"calibration_images": 1000})}
    for file, (attack, strengths, params) in expected.items():
        configs = expand_config(json.loads((REPO_ROOT / "experiments" / "configs" / file).read_text(encoding="utf-8")))
        assert [c.strength for c in configs] == strengths
        assert all(c.attack == attack and c.params == params for c in configs)
        assert attack in harness.available_attacks()
