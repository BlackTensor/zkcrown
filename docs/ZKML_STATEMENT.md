# zkML statement: what the Track B (EZKL) proof shows, and what it does not

> **Provenance of this file.** Drafted by Claude (an AI assistant) at the
> owner's request for task P8.7. The owner reviews it. Every number here is
> copied from a committed result file named next to it. Every measured figure
> is from **this machine, not Colab**: a local Windows CPU running x64 Python
> 3.11.9 under emulation on ARM.

This file covers the EZKL proof in `results/zk/p8.4/` and the pipeline
measured in P8.3 to P8.6 (ezkl 23.0.5, Halo2 with KZG commitments over
BN254). Under the Phase 8 decision of 2026-10-03 (option (c3)), Track B
proves **plain inference** of the P0.7 `zk_model` on a **public** input. It
shows that the zkML pipeline works and what it costs. It is not a watermark
test, an ownership claim or a provenance record.

## 1. What the proof shows

**Model.** The P0.7 `zk_model`, a 6,138-parameter MNIST CNN (three 3x3
stride-2 conv + ReLU layers of width 8 / 16 / 16, then a linear classifier),
exported to ONNX in P8.1. The exported file is `results/zk/p8.1/zk_model.onnx`
(SHA-256 `6416735f6d7eef04d30906003a40b85c210b61253a9e53935426f3b56f7f5ca9`).
Its weights are bit-identical to the PyTorch weights, and it carries no
watermark.

**Circuit settings** (`results/zk/p8.3/settings.json`, SHA-256
`f4480ce9bed9864545c0d764879b2c59e17a4d986a40830d4231b09d2633563f`):

- input `public`, output `public`, parameters `fixed`;
- input scale 13 and param scale 13, from P8.3's `accuracy` calibration;
- logrows 18, 145,820 rows;
- `check_mode: UNSAFE` (section 3).

With `fixed` parameters the weights are built into the circuit as fixed
values, not supplied as inputs. A Halo2 verification key commits to a
circuit's fixed values, so a verification key belongs to one set of weights.
The key is committed at `results/zk/p8.3/vk.key` (525,575 bytes, SHA-256
`c582aec46db7e6bd4ea6aae5c7675882e58940b56d776d6a7797d623f7da8023`).

**Statement.** A proof that verifies under that key, with those settings and
the SRS of section 4, shows this: the circuit, evaluated on the 784 public
input values in the proof, produces the 10 public output values in the proof.
That circuit is the quantised (fixed-point) version of the `zk_model` whose
weights are fixed in the key. Inputs and outputs are field elements at scale
13, so values are multiples of 2^-13 (about 0.000122).

For the committed proof (`results/zk/p8.4/proof.json`, SHA-256
`f02b10439632891b85d431ee3cb7ea84bfd728e26614c79b6c89e748f7b8a40d`):

- **Input:** MNIST test image index 0, fixed by position before the run, and
  normalised as in training. The largest difference between the input and
  its circuit encoding is 6.1e-5.
- **Circuit outputs (dequantised):** -0.9819, 0.7084, 2.6079, 4.5282,
  -8.9038, -2.9515, -11.3022, **16.5438**, -1.8096, 1.1270. The class is 7,
  which equals the PyTorch class and the label.
- **PyTorch outputs:** -0.9820, 0.7095, 2.6111, 4.5306, -8.9037, -2.9544,
  -11.3006, **16.5435**, -1.8112, 1.1266. The largest absolute difference is
  0.0031.

The proof verifies with the committed settings and verification key (P8.5).

"Up to quantisation" is measured, not assumed. P8.6 ran the circuit on all
10,000 MNIST test images (section 5): top-1 agreement was 10,000 / 10,000,
and the largest absolute logit difference was 0.0101.

## 2. What the proof does not show

1. **Anything about watermarks, ownership or provenance.** The input is
   public and chosen by whoever makes the proof, and nothing is hidden. A
   proof that a model gives some output on some input is not watermark
   evidence. `zk_model` carries no watermark at all (P0.7 trained it clean).
   The proof makes no ownership claim and has no tie to the provenance record
   or to any timestamp. The watermarked `zk_model` idea is in the Icebox,
   with the per-trigger commitment it would need first.
2. **That the circuit equals PyTorch beyond the 10,000 images measured.**
   P8.6 measured agreement on the MNIST test set only. For any other input,
   in particular inputs far from MNIST or ones built to sit near a decision
   boundary, nothing is measured. A logit difference of up to 0.0101 can
   change the class wherever PyTorch's top two logits are closer than that.
   On the test set that did not happen.
3. **Anything about `main_model`.** `main_model`, the CIFAR-10 model carrying
   both watermarks, is not in any circuit. Section 2.2 of CLAUDE.md
   separates the two models on purpose.
4. **That this model is the one in the Phase 5 commitment.** The commitment
   publication and the provenance record name the dual `W*` fingerprint
   `c0995109…5b064a07`, a `main_model`. The `zk_model` fingerprint is
   `4d6683b7…08241955` (P5.1). Nothing in the commitment, the record or the
   P5.5 timestamp refers to `zk_model`, its ONNX file or this verification
   key.
5. **That the verification key was derived from the committed ONNX file,
   checked by anyone but the owner.** The link is this repo's own pipeline:
   P8.3 checked the ONNX hash, then compiled and set up. A verifier who wants
   the link must redo setup from the ONNX file and compare keys. Whether
   ezkl's setup gives byte-identical keys on a rerun was not tested.
6. **Zero-knowledge of anything.** Input and output are public, and the
   weights are public too (the ONNX file is committed). The proof system
   could hide a witness, but this statement hides nothing, so no privacy
   claim is made.
7. **Soundness in general.** P8.5's 42 tampered variants were all rejected.
   That is evidence about the cases tried, not a soundness proof. Soundness
   rests on Halo2/KZG, on the SRS (section 4) and on ezkl's circuit
   constraining the model correctly. None of these was audited here.

## 3. `check_mode: UNSAFE`

The settings carry `check_mode: UNSAFE`, the value ezkl wrote at
`gen_settings` (P8.3).

**What the documentation says.** The official ezkl 23.0.5 Python bindings
documentation (`pythonbindings.ezkl.xyz/en/stable/`, `PyRunArgs`) says only:
"check mode, accepts `safe`, `unsafe`". **The documentation does not say
what the mode does, so its meaning for the proof cannot be confirmed from
the documentation.**

**What the source says** (my reading of the v23.0.5 source at tag `v23.0.5`,
commit `534ff3e`, not an audit). `CheckMode` is described there as "the
sanity checks we can perform on the accumulated arguments". Its value is
read in three places, all on the prover's side:

- `range_check` in `src/circuit/ops/layouts.rs`: in SAFE mode, witness
  generation also asserts each range-checked value is in range. The source
  comment says this is an extra check "for optimization", and that "range
  violations are detected during both proving and verification" through the
  lookup constraint, which is configured either way.
- `assign_with_duplication_constrained` in `src/tensor/var.rs`: in SAFE
  mode, an `assert_eq!` that duplicated cells hold equal values. The copy
  itself is still constrained either way.
- `create_proof_circuit` in `src/pfsys/mod.rs`: in SAFE mode the prover
  verifies its own proof before returning it.

In this reading, UNSAFE skips prover-side self-checks and does not remove
constraints, so a verifier's check would be the same either way. One
consequence holds regardless of reading: the prover does not re-verify its
own proofs, so a proof is known good only once it has been verified
separately. Here P8.5 and P8.6 did that (11 of 11 honest proofs accepted).
Whatever the setting, the Python `ezkl.prove` in 23.0.5 passes
`CheckMode::UNSAFE` to the prover unconditionally (`src/bindings/python.rs`).

**Not done:** no proof was regenerated with `check_mode: safe` to compare.
The reading above is of source code, not documentation, and it has not been
tested.

## 4. Trust in the SRS

The KZG structured reference string is `kzg18.srs`, 33,554,692 bytes,
SHA-256 `d0148475717a2ba269784a178cb0ab617bc77f16c58d4a3cbdfe785b591c7034`.
It was fetched by `ezkl.get_srs` in P8.3 and is not committed.

**This is a downloaded file whose hash was recorded, not a ceremony verified
here.** What is known:

- ezkl 23.0.5 downloads it from `https://kzg.ezkl.xyz`. The source comment
  describes that site as a mirror of the powers of tau files of
  `github.com/han0110/halo2-kzg-srs`.
- ezkl checks the downloaded file against a SHA-256 table shipped in its own
  source (`src/srs_sha.rs`). The k = 18 entry equals the hash recorded in
  P8.3.

So the file is the one ezkl's maintainers list. Nothing in this repo checks
the ceremony behind it or its contributors. Like Track A's Hermez file
(P7.4), whoever relies on this proof trusts that ceremony and ezkl's hash
table. If the ceremony's secret were known to someone, that person could
forge proofs that verify under this key. Unlike Track A, there is no
circuit-specific phase 2 here, so no owner-only contribution is involved.

## 5. Measured figures (this machine, not Colab)

All figures are on local Windows CPU (x64 Python under emulation on ARM),
**this machine, not Colab**, one run each unless stated. Memory is the peak
working set of the Python process doing the step, measured by that process
itself (P8.3 method).

**Setup (P8.3)**, `results/p8.3_ezkl_setup__seed1337__20261003T185318+0000.json`:

| Item | Value |
|---|---|
| Calibration (`accuracy`, 200 MNIST training images) | 320.8 s, peak 333.3 MiB |
| Circuit | 145,820 rows, logrows 18, input/param scale 13 |
| `setup` | 24.6 s, peak 1,881.7 MiB |
| Proving key | 1,107,822,347 bytes (not committed) |
| Verification key | 525,575 bytes (committed) |
| SRS `kzg18.srs` | 33,554,692 bytes (not committed) |

**Proof (P8.4)**, `results/p8.4_ezkl_prove__seed1337__20261003T185838+0000.json`:

| Item | Value |
|---|---|
| Witness generation | 0.31 s, peak 65.1 MiB |
| `prove` | 21.7 s, peak 2,504.6 MiB |
| Proof | 3,072 bytes; 140,052 bytes as ezkl JSON with 794 public instances (784 input, 10 output) |

**Verification (P8.5)**, `results/p8.5_ezkl_verify__seed1337__20261003T190242+0000.json`:

| Item | Value |
|---|---|
| `verify`, 5 runs | median 0.215 s (0.204–0.313 s); 0.60–0.71 s per process including Python start-up |
| Verify peak | 222.6–223.0 MiB |
| Tampered variants (outputs, inputs, proof bit flips, another image's instances, wrong instance count) | 42 tried, 0 accepted; 3 / 3 honest controls accepted |

**Fidelity (P8.6)**, `results/p8.6_ezkl_fidelity__seed1337__20261005T115835+0000.json`.
One model and one set of scale settings:

| Item | Value |
|---|---|
| Images | all 10,000 MNIST test images, fixed before the run |
| Top-1 agreement, circuit vs PyTorch | 10,000 / 10,000, 0 disagreements, 0 ties |
| Accuracy, circuit / PyTorch | 98.96% / 98.96%, the same images right and wrong |
| Absolute logit difference | max 0.0101, mean 0.0018, median 0.0015, p99 0.0062 |
| Time per image, witness only | median 0.307 s, mean 0.309 s |
| Proofs of 10 seed-fixed images | 10 / 10 verify; outputs equal the witness-only outputs; 19.5–23.7 s and 2.34–2.45 GiB each |

## 6. Callers: an exception is a rejection

In P8.5, **every** rejected verification raised an exception
(`RuntimeError("Failed to run verify: [halo2] …")`). None returned a clean
`False`. A caller of `ezkl.verify`, such as the P9 auditor, must therefore
treat any exception as a rejection, not as a crash to retry or ignore. Only
a return value of exactly `True` counts as accepted. snarkjs behaves the same
way for Track A (P7.8).

## 7. One-sentence summary

The EZKL proof shows that a committed 6,138-parameter MNIST model, its
weights fixed in the circuit, produces the stated scores on the stated public
input, up to quantisation at scale 13, as measured on this machine and not on
Colab. It says nothing about watermarks, ownership, provenance, `main_model`,
or inputs beyond the 10,000 measured.
