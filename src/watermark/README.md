# `src/watermark`

Key-derived triggers now; later the behavioral watermark, weight embedding and
extractor.

- `keygen.py`: deterministic streams derived from the master key `K` (P1.1).
- `triggers.py`: the trigger set `T` (P1.2). The construction is specified in
  the module docstring.

## Trigger design choice (P1.5)

**Decision: additive, key-derived noise, with an independent pattern for each
trigger.** Trigger `i` is `clip(x_i + A * s_i, 0, 255)`, where `x_i` is a
key-selected image from the `main_model` training split and `s_i` is a ±1
pattern drawn from `K`, with A = 16/255.

**Rationale.** Three constraints come from later phases. The triggers must be
a function of `K` alone, so they can be committed to before any model exists
(P5.3) and in principle bound to the commitment in a circuit (P7.9). They must
regenerate byte for byte on any machine. And "k of N triggers fired" has to be
a fair statistical test (P2.8).

- **A learned trigger fails the first two constraints.** Optimising triggers
  against a network makes them depend on the weights as well as `K`, and GPU
  floating point makes exact regeneration hard to guarantee.
- **A patch fails the third.** One visible stamp shared by every trigger makes
  their responses move together, which undermines the independence behind
  the P2.8 test. A small fixed patch is also the backdoor shape that trigger
  reverse-engineering defences such as Neural Cleanse are designed to recover.
- **Per-trigger additive noise meets all three.** It is integer arithmetic on
  uint8 pixels. P1.4 showed byte-identical regeneration, and cross-key
  independence of base indices and sign patterns under all 256 single-bit
  key flips. No component is shared between triggers except the amplitude. At
  A = 16, P1.3 measured a mean PSNR of 24.17 dB: the grain is visible, but the
  image content stays recognisable.

**Costs of this choice, not yet measured:**

- The model has to memorise 100 specific perturbed images rather than learn
  one reusable pattern. That may make the watermark more fragile under
  fine-tuning and distillation than a shared pattern would be (P4.5, P4.7).
- High-frequency noise is the kind of signal that blurring or JPEG
  re-encoding tends to weaken, and no attack in Phase 4 tests that.
- The derivation uses SHA-256, which is expensive to express inside a Circom
  circuit (P7.9).

If Phase 4 shows the memorised triggers do not survive, that is reported as a
finding, not fixed by quietly switching designs.
