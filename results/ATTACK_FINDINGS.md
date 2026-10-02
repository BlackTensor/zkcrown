# Attack findings (Phase 4)

What survived, what did not, and which watermark held up better under which attack.

Every number here is copied from a committed result file. Nothing is measured in this
document. The per-setting table is `results/p4.9_master_robustness_table.md`, the heatmap is
`figures/p4.9_robustness_heatmap.png`, and the source rows are in `results/attacks/`.

## How to read the numbers

- **The model attacked** is the dual-watermarked `W*` (P3.6): 90.85% test accuracy, 100 of 100
  owner triggers firing (p = 3.8e-96), weight z = 10.29 (p ≤ 1.1e-23).
- **Behavioral watermark:** how many of the owner's 100 key-derived triggers get their keyed
  response. A model built without the key fires on at most about 11 by chance.
- **Weight watermark:** z, the scaled correlation between the suspect's weights and the owner's
  signature. A model built without the key gives z near 0.
- **"Detected"** means the watermark's own test rejects at 1e-6: at least 29 triggers fired, or
  z at least 5.257. That level was fixed before any attack ran. The two tests are separate and
  are not combined here.
- **Accuracy** is on the 10,000-image CIFAR-10 test set. Drops are against `W*`.
- **The attacker never has the key.** They know how the scheme works.

## Headline: distillation removes both watermarks, almost for free

**Neither watermark survives distillation into a fresh student when the attacker has enough
data.** This is the most important limitation of the system.

- The thief queries the stolen model on ordinary images and trains a new network on its
  outputs. They use no labels, no key and no triggers.
- With all 50,000 CIFAR-10 training images, a same-size student reaches **90.67%** test
  accuracy. The drop is +0.18 pp (95% CI [-0.27, +0.63], p = 0.46), which is not
  distinguishable from zero.
- In that student, 5 of 100 triggers fire (p = 0.99) and the weight z is -1.11 (p ≤ 1). There
  is no evidence of either watermark at any level.
- All 6 students tested show the same thing: 1 to 7 triggers fired, and weight z between -1.11
  and -0.19 for the three same-size students.
- For the three half-width students the weight test **cannot be run at all**, because their
  weights do not have the owner's layout.

Why it works: the thief never queries a trigger, so the student is never shown the keyed
responses. And the student is a new set of weights that never sees the teacher's. This is a
genuine limitation of trigger-based and weight-based watermarking, not a bug in this
implementation.

Two qualifiers, both of which matter:

1. **The 50,000-image set includes the owner's own 45,000 training images.** A real thief
   would rarely hold those. With only the 5,000-image attacker holdout, the students reach
   69.52% to 78.49%, a cost of 12.4 to 21.3 pp. How distillation does on 50,000 unseen images
   was not measured; CIFAR-10 has no further images to test it with.
2. **Distillation was not the first attack to remove both.** Channel pruning followed by
   fine-tuning had already done it in 3 runs, but at a cost of 9.9 to 12.3 pp.

## What removed what

Counts over the 84 attack settings (P4.9). They depend on which settings were swept, so they
are counts, not rates.

| Outcome | Settings |
|---|---|
| Both watermarks detected | 34 |
| Behavioral lost, weight detected | 39 |
| Weight lost, behavioral detected | 0 |
| Neither detected | 11 |

**In no setting was the weight watermark lost while the behavioral one was still detected.**
Wherever the triggers survived, so did the weight watermark.

The cheapest removal of each watermark, by attack. "Cost" is the test accuracy the thief gives
up. Each is a single run.

| Attack | Cheapest setting that removes the behavioral watermark | Cheapest setting that removes both |
|---|---|---|
| Distillation, 50,000 images | 90.67% (cost 0.18 pp, not distinguishable from zero) | the same run |
| Channel pruning + fine-tuning | 85.94% (cost 4.91 pp) with 7 fired; 88.19% with 19 fired, which still rejects at 0.05 | 80.92% (cost 9.93 pp) |
| Global pruning + fine-tuning | 86.75% (cost 4.10 pp) with 8 fired; 88.99% with 24 fired, which still rejects at 1e-3 | never |
| Fine-tuning alone | 86.06% (cost 4.79 pp) | never |
| Overwrite with the attacker's triggers | 86.03% (cost 4.82 pp), no better than fine-tuning alone | never |
| Channel pruning alone | 46.92% (model badly damaged) | 9.97% (model at chance) |
| Magnitude pruning alone | 50.79% layer-wise, 18.28% global (model badly damaged) | never |
| Overwrite with the attacker's weight watermark | 39.00% (model badly damaged) | never |
| Quantization (FP16, INT8) | never | never |

These costs are not compared statistically. The fine-tuning, pruning-plus-fine-tuning and
trigger-overwrite figures all sit within about one percentage point of each other, on single
runs, so no claim is made about which of them is cheapest.

## The behavioral watermark

**It survives anything that does not retrain the model, for as long as the model is still
useful.**

- **Quantization:** untouched. 100 of 100 triggers at FP16 and at INT8 (90.77% accuracy).
- **Magnitude pruning:** detected up to 60% layer-wise sparsity (71 fired, 74.23% accuracy)
  and 80% global sparsity (42 fired, 49.27%). It is lost only after accuracy has fallen to
  50.8% or lower.
- **Channel pruning:** detected at 5% and 10% of channels (94 and 70 fired, at 86.02% and
  74.74%). Lost from 20%, where accuracy is 46.92%.
- **Weight overwrite by an attacker:** detected up to alpha' = 1.0, ten times the owner's
  strength (52 fired, 80.68%). Lost at 2.0, where accuracy is 39.00%.

**It does not survive retraining at a normal learning rate.**

- **Fine-tuning** on 5,000 images at learning rate 0.05 or 0.1 removes it in every run,
  including 5 epochs. 4 to 11 triggers fire, which is what an unrelated model gives. The
  triggers are not redirected: 39 to 54 of them go back to their base image's label.
- The cheapest removal costs **4.79 pp** (86.06% accuracy, learning rate 0.05, 20 epochs).
- A gentle rate does not do it. At 0.001 all 100 still fire. At 0.01 the count wears down to
  62, 46 and 30 over 5, 20 and 60 epochs. 30 is one above the threshold.
- **Pruning first lets a gentle rate finish the job.** Learning rate 0.01 alone left 46 of
  100. After 90% global pruning the same rate leaves 10 of 100, at 84.61%.
- **Pruning then fine-tuning** left it undetected in 23 of 24 runs.
- **Overwriting with the attacker's own triggers** did not remove more than plain fine-tuning
  at the same rate. At learning rate 0.01 more owner triggers survived with the overwrite (70)
  than without (46). That is one run each, so no claim about direction.
- **Distillation** removes it completely (see the headline).

## The weight watermark

**It survives everything that keeps the stolen weights, except channel pruning combined with
aggressive fine-tuning.** It was detected in 73 of 84 settings.

- **Quantization:** unaffected. z = 10.29 at FP16 and 10.67 at INT8.
- **Magnitude pruning:** detected at every sparsity up to 90%, lowest z 8.17. That includes
  models at 11.63% and 18.28% accuracy.
- **Fine-tuning alone:** detected in all 12 runs. The lowest is z = 6.49 at learning rate 0.1
  for 60 epochs. z falls with rate and length, and whether longer or harder fine-tuning would
  remove it was not measured.
- **Global pruning then fine-tuning:** detected in all 12 runs, lowest z 6.60.
- **Overwrite:** detected in all 11 runs, lowest z 9.49. A second weight watermark under
  another key, even at 20 times the owner's strength, moved z only from 10.29 to 9.72.

**Where it fails:**

- **Channel pruning alone** removes it only at 80% and 90% of channels (z 3.73 and 2.08),
  where the model is at chance accuracy.
- **Channel pruning then fine-tuning at learning rate 0.1** removes it in 3 of 12 runs, with
  a usable model left over:
  - 30% of channels, 60 epochs: z 4.85, 80.92% accuracy.
  - 50% of channels, 20 epochs: z 4.54, 78.51%.
  - 50% of channels, 60 epochs: z 1.93, 78.94%.
  - The first two still reject at 1e-3. Only the third leaves no evidence at any level.
- **Distillation** removes it completely, and for a student of another width the test cannot
  even be run.

**Two conditions the weight results depend on:**

- **The owner needs the suspect's weights.** The behavioral test needs only query access. A
  thief who serves the model behind an API is out of reach of the weight test.
- **The suspect must keep the owner's layer shapes.** Every channel-pruning figure above uses
  weights with the removed channels set to zero, not deleted. A thief who physically deletes
  them ships a narrower model, and the extractor then reports "not applicable". Re-aligning
  such a model was not built and not measured. So against the attack that came closest to
  defeating the weight watermark, the figures here are the owner's best case.

## Which watermark is stronger under which attack

| Attack | Behavioral | Weight | Stronger |
|---|---|---|---|
| Quantization | 100 of 100 | z 10.29 to 10.67 | neither; both untouched |
| Magnitude pruning | lost at 70% (layer-wise) and 90% (global) | detected at all 18 settings | weight |
| Channel pruning | lost from 20% of channels | detected up to 70% (assuming re-alignment) | weight |
| Fine-tuning | lost at learning rate 0.05 and above | detected in all 12 | weight |
| Global pruning + fine-tuning | detected in 1 of 12 | detected in 12 of 12 | weight |
| Channel pruning + fine-tuning | detected in 0 of 12 | detected in 9 of 12 (assuming re-alignment) | weight, but it is lost in 3 |
| Overwrite | lost in 3 of 11 | detected in 11 of 11 | weight |
| Distillation | detected in 0 of 6 | detected in 0 of 6; not applicable in 3 | neither; both removed |
| Model served behind an API only | testable | not testable | behavioral (by design, not measured) |

On the attacks that keep the stolen weights, the weight watermark held up better than the
behavioral one in every family measured. It is also the one with the stricter requirements:
white-box access and an unchanged layout. The behavioral watermark is weaker under retraining
but is the only one that can be checked from outside.

## An overwrite leaves two claims, not one

After an overwrite attack the model carries the attacker's watermark as well as the owner's.
The attacker's own tests pass: their weight z is 10.20 to 11.31, and all 100 of their triggers
fire at learning rate 0.01 and 0.05.

The detection tests cannot say who was first. That question is answered by the commitment
published before the theft (Phase 5), not by anything in Phase 4.

## What these results do not show

- **One run per setting, one key, one source model, one seed.** The intervals in the result
  files cover which test images were sampled. They do not cover variation between training
  runs, keys or models.
- **The attacker in the retraining attacks has 5,000 images.** Only distillation was also run
  with 50,000.
- **No adaptive attacker.** Nobody here designs a loss to target the watermark, queries near
  the triggers, or filters inputs with blur or JPEG compression. Those are not measured.
- **Distillation used soft outputs at one temperature.** A label-only API was not tested.
- **INT8 quantization was simulated**, not run on an integer backend.
- **The accuracy of `W*` itself is optimistic**, because its watermark strength was chosen
  after looking at test-set accuracy (P3.6).
- **Detection at chance accuracy has little practical value.** The weight watermark is still
  found in models pruned down to 11.63% accuracy, but no thief would ship such a model.
- **Counts are not rates.** A family swept with many harsh settings shows more losses.
- This is a technical ownership-verification demonstration, not legal evidence.

## Sources

| Task | Attack | Result files |
|---|---|---|
| P4.1 | control | `results/attacks/p4.1_none_0__*.json` |
| P4.2 | magnitude pruning | `results/attacks/p4.2_magnitude_prune_*.json` |
| P4.3 | channel pruning | `results/attacks/p4.3_channel_prune_l1_*.json` |
| P4.4 | quantization | `results/attacks/p4.4_ptq_*.json` |
| P4.5 | fine-tuning | `results/attacks/p4.5_finetune_holdout_*.json` |
| P4.6 | pruning + fine-tuning | `results/attacks/p4.6_prune_finetune_*.json` |
| P4.7 | distillation | `results/attacks/p4.7_distill_*.json` |
| P4.8 | overwrite | `results/attacks/p4.8_overwrite_*.json` |
| P4.9 | master table | `results/p4.9_master_table__seed1337__20261002T171216+0000.json` |

The detection tests are defined in `src/watermark/significance.py` (P2.8) and
`src/watermark/weight_significance.py` (P3.7).
