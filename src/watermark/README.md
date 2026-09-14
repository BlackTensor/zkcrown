# `src/watermark`

Key-derived triggers now; later the behavioral watermark, weight embedding and
extractor.

- `keygen.py`: deterministic streams derived from the master key `K` (P1.1).
- `triggers.py`: the trigger set `T` (P1.2). The construction is specified in
  the module docstring.
- `signature.py`: the ownership signature `S` (P2.1).
- `responses.py`: the target class each trigger maps to (P2.2).
- `bundle.py`: the secret trigger bundle (triggers plus targets) that carries the
  watermark to a Colab run, so `K` itself never leaves the owner's machine (P2.3).
- `behavioral.py`: joint clean-plus-trigger training, via `TriggerMixLoader` (P2.3).
- `significance.py`: the p-value for "k of N triggers fired" (P2.8).
- `projection.py`: the key-derived projection `P_K` the weight watermark lives
  in (P3.1).
- `carrier.py`: which weights form the carrier vector, and in what order (P3.2).
- `weight_embedding.py`: the spread-spectrum embedding `W* = W + alpha * P_K^T * S` (P3.2).
- `weight_extraction.py`: the blind extractor and its correlation score (P3.3).
- `weight_significance.py`: the weight-watermark p-value bound and threshold (P3.7).

## Weight detection test (P3.7)

**H0:** the suspect was produced independently of `K`. **Statistic:**
`z = correlation * sqrt(128)`. **p-value:** `min(1, exp(-z^2/2))`, one-sided.
**Threshold:** `z* = sqrt(2 ln(1/alpha))`, which gives 5.257 at alpha 1e-6.

Why this is valid for every H0 model. `S` and `P_K` come from different PRF
streams of `K`. A suspect independent of `K` therefore gives a fingerprint
`y = P_K c` that is independent of `S`. Given `y`, z is `sum a_i S_i` with
`sum a_i^2 = 1` and uniform signs `S_i`, so Hoeffding's inequality bounds its
tail by `exp(-t^2/2)`. The bound does not depend on the model's architecture
scale or training, and it is conservative. A Gaussian threshold would be
lower, but it is only an approximation. The exact tests show it is not a
valid bound for small equal-weight sums, so the auditor does not use it.

Because the correlation cannot exceed 1, the smallest achievable p-value is
`exp(-64)`, about 1.6e-28.

1,000 wrong keys on three real models gave a null consistent with N(0, 1),
with exceedance counts well under the bound (numbers in CLAUDE.md section 8).
That check only covers alpha near 0.05 and 0.01; smaller levels rest on the
proof. The assumptions match P2.8's: `K` committed before the suspect was
seen, one pre-declared test, and a correction for multiple suspects.

## Dual watermark: post-hoc, alpha 0.1 (P3.6)

**Decision (owner): the weight watermark is embedded post-hoc into the P2.3
behavioral model, at alpha = 0.1, without BatchNorm recalibration.** The
result is the final dual-watermarked `W*`,
`results/p3.6_dual_wm_W_star.pt` (gitignored, SHA-256 `7a9a9f14…b434c4`),
built by `experiments/p3_6_make_dual_model.py`.

Why post-hoc. P3.4 and P3.5 measured exactly this setup. It keeps the
behavioral model, needs no retraining, and extraction stays blind. Embedding
during training would need a new design and a GPU run, and it would change
what `alpha` means, all for a robustness benefit nobody has measured. It is in
the Icebox, to revisit only if Phase 4 shows the post-hoc watermark does not
survive attacks.

Why 0.1. In P3.5 it is the largest grid value where accuracy showed no
detectable cost on test or holdout and test loss had not yet started to rise.
Detection there is z = 10.29, with 126 of 128 bits right. Alpha was chosen
after seeing P3.5's test-set drops, so the dual model's test accuracy is an
optimistic estimate.

Checked before calling it final, all on the weights reloaded from disk:
- The weight watermark reproduces P3.5.
- The behavioral watermark still passes the P2.8 test at 1e-6. This gate was
  fixed before the run, and it fires on 100 of 100 triggers.
- Accuracy reproduces P3.5.

The measured numbers are in CLAUDE.md, section 8.

## Weight extraction (P3.3)

```
c = suspect carrier with each tensor's mean subtracted
y = P_K c                                  the recovered 128-value fingerprint
correlation = <y, S> / (||y|| sqrt(128))   primary score, in [-1, 1]
```

The extractor also reports `amplitude = <y, S> / 128` (about `alpha` plus host
noise for a post-hoc embedding), `projected_rms`, and `bit_matches`, the number
of bits where `y_i > 0` matches `S`.

- **Blind.** Only the suspect's weights are used, never the clean `W`, so the
  extractor also works if P3.6 embeds during training.
- **Centred per tensor.** This removes the leak of each layer's mean through
  the unbalanced row sums of `P_K` (P3.1), and makes the result unchanged by
  constant shifts. The watermark loses only its component along 7 of 307,040
  directions.
- **Normalised correlation.** It is unchanged by global rescaling of the
  weights.

The host weights contribute a random term to `y`, so a model without the
watermark still gives a non-zero correlation. Its null distribution and the
detection threshold are measured in P3.7, not assumed. The real-model gap
between the correct and a wrong `K` is measured in P3.4. Per-layer rescaling,
which BatchNorm makes function-preserving, is not handled and belongs to
Phase 4. The fingerprint and recovered bits are treated as secret: they are
hidden in `repr` and left out of `to_dict`.

## Weight embedding (P3.2)

**Carrier.** Every `nn.Conv2d` and `nn.Linear` `weight`, in `named_modules()`
order, flattened C-order. For `main_model` that is 7 tensors and
`dim = 307,040`. The 906 BatchNorm and bias parameters and the BN buffers are
left out. `CarrierLayout.digest()` identifies the layout, so a mismatched model
is refused.

**Embedding.** `W* = W + alpha * P_K^T * S` on the carrier. `S` is used as ±1
signs, and everything else in the state_dict is copied bit for bit. With
unit-norm rows, `P_K w* = P_K w + alpha * S + cross-talk`, so `alpha` is the
per-bit amplitude an extractor sees. In weight space each parameter moves by
`alpha / sqrt(dim)` times a sum of 128 fair signs. That is about
`alpha * sqrt(128 / dim)` RMS, the same distribution in every layer, and
exactly zero for about 7% of parameters (`C(128,64) / 2^128`). These follow
from the construction. `EmbeddingSummary` reports the realised change per
tensor, as aggregates only, together with the measured float32 rounding error.

**Open.** `alpha` has no default (P3.5). The function is post-hoc, and BN
running statistics are not recalibrated. Whether to embed post-hoc or during
training is P3.6. The change has the same size in every layer, even though
layer weight scales differ.

## Projection `P_K` (P3.1)

```
P_K[i, j] = sigma[i, j] / sqrt(dim)       sigma in {-1, +1}, shape (rows, dim), rows = 128
sigma row i = bytes [i*B, (i+1)*B) of KeyStream(K, "projection/v1/rademacher/dim=<dim>"),
              B = ceil(dim / 8), bits MSB first, trailing bits dropped, bit 1 -> +1
```

One row per bit of `S`, one column per carrier parameter. Every row has norm
exactly 1 and touches every parameter with the same magnitude, so each bit is
spread over the whole carrier. Two different rows have an inner product with
mean 0 and standard deviation `1/sqrt(dim)`, so `P_K P_K^T` is the identity
plus small cross-talk. That is what lets P3.3 read `S` back by projecting.
The signs are generated as integers, so `P_K` is byte-identical everywhere. It
is deliberately not orthonormalised, because QR would make it a floating point
result. `dim` is part of the label, so different carrier sizes get unrelated
matrices. The first `m` rows do not depend on `rows`.

Left to later tasks: which parameters form the carrier and `alpha` (P3.2),
whether to centre the carrier before projecting (P3.3), since unbalanced row
sums let a non-zero carrier mean leak into `P_K w`, and the detection
threshold (P3.7). `P_K` is as secret as `K`, and its `repr` hides the signs.
The tests check the Gram matrix, row sums and cross-key correlations at
`dim = 307,040` against their null distributions.

## Detection test (P2.8)

**H0:** the suspect model was produced independently of the owner's keyed
targets. Any model built without `K` qualifies. **Statistic:** `k`, the number
of the N triggers that fire (P2.4). **p-value:**

```
p(k) = P(Binomial(N, 1/(C-1)) >= k)       exact, rational arithmetic; 1/9 for CIFAR-10
```

Why this is valid for every H0 model: with the model and images fixed, trigger
`i` fires with probability exactly 1/9 if the model does not predict the base
label, and 0 if it does. The targets are independent draws, so the fired count
is a sum of independent Bernoullis each at most 1/9. That sum is
stochastically dominated by the binomial (P2.2). The test is conservative for
accurate models, which predict the base label on many triggers and fire less.

For N = 100 the thresholds are `k* = 17, 20, 23, 29, 35` at `alpha = 0.05,
0.01, 1e-3, 1e-6, 1e-9` (P2.8 result file). The p-value only means something
if `K` and the trigger set were fixed, and committed, before the suspect was
seen. It also assumes one pre-declared test, one query per trigger, and a
correction when auditing several suspects. The full list of assumptions, and
what the p-value does not say, is in the `significance.py` docstring.

## Joint training (P2.3)

`W*` is trained from scratch with the P0.5 recipe, seed and clean data, plus 4
trigger samples appended to every batch of 128. The clean batches and their
order are exactly P0.5's; only the triggers differ, so P2.5's accuracy drop
isolates the watermark. Triggers are not augmented, and their base images stay
in the clean set with their true labels. Trigger order is balanced per epoch
and survives a resume. The 4 per batch is a starting value; P2.7 sweeps it.
Details and reasons are in the `behavioral.py` docstring.

`K` stays local in `secrets/K.bin`. `experiments/p2_3_make_trigger_bundle.py`
turns it into `secrets/trigger_bundle.npz`, and only that bundle is uploaded.
Result records carry the bundle's SHA-256 digest, never its contents, so the
owner can check a model's triggers against `K` afterwards.

## Trigger responses (P2.2)

**Decision: per-trigger keyed response, never the base image's label.**

```
r_i = i-th KeyStream(K, "responses/v1/target-class").randbelow(C - 1)
t_i = (y_i + 1 + r_i) mod C          C = 10, y_i = dataset label of the base image
```

A trigger fires when the model's top-1 prediction on `T_i` equals `t_i`.
The targets depend on `K` and the dataset labels only, and like the triggers,
the first `m` targets do not depend on `N`.

**Why not a single owner class.** Take any model built without `K`. Its
prediction on `T_i` is fixed, and `t_i` is independent of that prediction and
uniform over the 9 classes other than `y_i`. So each trigger fires with
probability 1/9 if the model does not predict `y_i`, and 0 if it does.
Different triggers fire independently. Under the null, the fired count is
therefore bounded by `Binomial(N, 1/9)` for **every** model, however accurate
or biased it is. That is the bound P2.8 needs. With one owner class `c`, no
such model-independent bound exists: a model biased towards `c` fires on many
triggers it has never seen, an accurate model fires on the roughly N/10
triggers whose base image is already class `c`, and all fire events move
together with the model's bias towards `c`. Excluding `y_i` has a second
benefit: firing always contradicts the image content. So once the watermark is
removed (P4.7), WDR can fall towards 0. Targets drawn uniformly over all 10
classes would leave it near 10%. The tests check the bound with a model
biased to one class, and check the binomial spread of the fired count. They
also plant a shared target class and confirm the spread check catches it.

**Costs, not measured.** The model has to memorise N arbitrary labels (P2.5,
P4.5, P4.7). The bound only holds for models independent of `K`, and an
adversary who knows `K` is out of scope.

## Ownership signature `S` (P2.1)

```
S = HMAC-SHA256(K, "zk-crown/keystream/v1\0" || u16_be(len(L)) || L || u64_be(0))[:16]
L = UTF-8("signature/v1/owner:" + owner_id)
```

This is the first 16 bytes of the P1.1 key stream under the label `L`, so `S`
is a 128-bit PRF output of the owner identity under `K`.

- **Bound to `K` and the owner.** Without `K` you cannot compute `S` for a
  given owner, and a different owner gets an unrelated `S`.
- **Separate from the triggers.** The label namespace keeps `S` apart from the
  trigger streams.
- **Fits one field element.** 128 bits is below the BN254 field modulus, so
  P5.3 can use `S` as a single field element with no reduction.
- **Bits and signs.** Bits are read most significant first. Signs map bit 1 to
  +1, which is the form P3.2 embeds.
- **Owner id rules.** `owner_id` must be NFC-normalised, have no leading or
  trailing whitespace, and be at most 256 UTF-8 bytes.
- **Project owner id.** The real `S` is derived for
  `PROJECT_OWNER_ID = "blacktensor-zkcrown-owner"`, chosen by the owner in P3.4.

`S` is **not** a public-key digital signature. Checking it requires `K`, so it
behaves like a MAC tag. Public verifiability comes from P6.2 and Phase 7. `S`
is treated as secret, like `K`.

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
