# What is in `results/` and `figures/`

An index: which file answers which question, grouped by phase, with the key finding in one line.
Nothing is measured here. Every figure below is copied from the committed file named in its
row, and each record holds its own seed, parameters, git commit and environment.

Headline numbers are not repeated here. They are in the
[README's headline table](../README.md#headline-results), and the attack results are written
up in [`ATTACK_FINDINGS.md`](ATTACK_FINDINGS.md). The full figures with every caveat are in the
Results Ledger, section 8 of [`CLAUDE.md`](../CLAUDE.md).

How to read it:

- File names are `<task>_<name>__seed<seed>__<UTC time>.json`. Below, a file is named by its
  prefix up to `__seed1337`; there is one committed file per prefix unless a row says
  otherwise.
- Every experiment is a single run, with one key and seed 1337. Confidence intervals cover
  test-image sampling only.
- "Detected" means the watermark's own test rejects at 1e-6. The two watermark tests are
  reported separately unless a row says it grades them together.
- Timings and memory figures are from the owner's local Windows machine, not Colab, unless a
  row says Colab.
- Model weights (`*.pt`) are gitignored and not here. The one model file committed is the
  hosted demo bundle (Phase 9).

## Phase 0: baseline models

| File | Question it answers | Key finding |
|---|---|---|
| `p0.4_main_model_summary` | How big is `main_model`? | 307,946 parameters, 307,040 of them conv/linear weights. Recorded from a dirty tree. |
| `p0.5_clean_baseline` | What does the clean model `W` score? (Colab T4) | 91.20% CIFAR-10 test accuracy after 60 of 60 epochs; the baseline every later drop is measured against. |
| `p0.7_zk_model` | How small and how accurate is the MNIST model for the zkML proof? (Colab T4) | 6,138 parameters, 98.96% MNIST test accuracy (final epoch). |

## Phase 1: triggers

| File | Question it answers | Key finding |
|---|---|---|
| `p1.3_trigger_visualization` | How large is the trigger perturbation? (public demo key, not `K`) | At amplitude 16/255, mean PSNR 24.17 dB against the base image; 3.58% of pixel channels clipped. |
| [`../figures/p1.3_trigger_set.png`](../figures/p1.3_trigger_set.png) | What do 100 demo-key triggers look like? | Visible grain, objects still recognisable (judged by eye). |
| [`../figures/p1.3_trigger_detail.png`](../figures/p1.3_trigger_detail.png) | Base, trigger and stretched perturbation side by side. | Same demo key and record. |
| [`../figures/p1.3_amplitude_sweep.png`](../figures/p1.3_amplitude_sweep.png) | How does the trigger look at amplitude 4 to 64? | Mean PSNR 36.14 dB at 4 down to 12.51 dB at 64; 16 was kept. |

## Phase 2: behavioral watermark

| File | Question it answers | Key finding |
|---|---|---|
| `p2.3_behavioral_wm` | What does the trigger-trained model `W*` score? (Colab T4) | 90.73% test accuracy, 91.14% holdout, trigger bundle `fbd65ec7…22baec8`. |
| `p2.4_wdr` | Does `W*` give its keyed response on its triggers? | 100 of 100 (WDR 100%); clean `W` fires on 3 of 100. These are the training triggers. |
| `p2.5_accuracy_drop` | What does the behavioral watermark cost in accuracy? | +0.47 pp on test, 95% CI [-0.04, +0.98], McNemar p = 0.077: not distinguishable from zero. |
| `p2.6_false_positive_rate` | How often do ordinary inputs give an owner target? | `W*`: 12.6% on noise (chance 12.7%), 10.0% on clean test images (chance 10.0%). Input-level rate cannot reach zero; the count test (P2.8) is what separates models. |
| `p2.7_r0001`, `p2.7_r0005`, `p2.7_r0020`, `p2.7_r0078`, `p2.7_r1252` | Training runs at five trigger ratios. (Colab T4) | One run each, 60 of 60 epochs; scored in the record below. |
| `p2.7_ratio_sweep` | How many triggers does training need, and at what accuracy cost? | WDR 3, 9, 19, 64, 100, 100, 100 of 100 from 0% to 12.5% trigger samples per clean sample; the accuracy drop shows no trend with the ratio. |
| [`../figures/p2.7_wdr_vs_accuracy_drop.png`](../figures/p2.7_wdr_vs_accuracy_drop.png) | The sweep as a figure. | From `p2.7_ratio_sweep`. |
| `p2.8_detection_test` | How unlikely is a fired count under "the model is independent of `K`"? | Exact binomial bound: `W*` 100/100 gives p = 3.8e-96, clean `W` 3/100 gives 0.999; at 1e-6 the threshold is 29 fired. 1,000 wrong keys: 0 rejections at 0.05 for either model. |
| [`../figures/p2.8_detection_test.png`](../figures/p2.8_detection_test.png) | The null bound against the wrong-key counts. | From `p2.8_detection_test`. |

## Phase 3: weight watermark

| File | Question it answers | Key finding |
|---|---|---|
| `p3.4_key_specificity` | Does extraction work only with the right key? | Wrong keys give correlation mean +0.001, sd 0.095; the correct key clears all 100 wrong keys from alpha 0.02 on both hosts tested. |
| `p3.5_alpha_sweep` | How does detection trade against accuracy as the strength grows? | z rises to 10.29 at alpha 0.1 then saturates near its 11.31 cap; no measurable accuracy cost up to 0.2, a cost at 0.3 (holdout) and 0.5 (both splits). |
| [`../figures/p3.5_alpha_sweep.png`](../figures/p3.5_alpha_sweep.png) | The sweep as a figure. | From `p3.5_alpha_sweep`. |
| `p3.6_dual_wm` | What is the final dual-watermarked model? | Alpha 0.1, post-hoc: 90.85% test accuracy, 100 of 100 triggers, weight z 10.29. Alpha was chosen after seeing test drops, so 90.85% is optimistic. |
| `p3.7_weight_null` | What false-positive rate does the weight test have? | Proven bound P(z ≥ t) ≤ exp(-t²/2), threshold z 5.257 at 1e-6; 1,000 wrong keys look like N(0, 1). Dual `W*`: p ≤ 1.1e-23. |
| [`../figures/p3.7_weight_null.png`](../figures/p3.7_weight_null.png) | The wrong-key null against the bound. | From `p3.7_weight_null`. |

## Phase 4: attacks

Written up in [`ATTACK_FINDINGS.md`](ATTACK_FINDINGS.md); not repeated here.

| File | Question it answers | Key finding |
|---|---|---|
| `attacks/p4.1_none_0` | Does the attack harness reproduce the unattacked model? | Yes: 90.85%, 100 of 100, z 10.29, equal to P3.6. A check on the harness, not an attack. |
| `attacks/p4.2_*` (18 files) | Magnitude pruning, layer-wise and global, 10% to 90%. | Weight watermark detected at every sparsity; behavioral lost only once accuracy is 50.8% or lower. |
| `attacks/p4.3_*` (10 files) | Structured (channel) pruning, 5% to 90%. | Far more destructive than magnitude pruning; behavioral detected only at 5% and 10%. Weight figures assume a re-alignment step that was never built. |
| `attacks/p4.4_*` (3 files) | FP16, fused FP32 and simulated INT8. | Both watermarks detected in all three; INT8 costs +0.08 pp, not distinguishable from zero. |
| `attacks/p4.5_*` (12 files) | Fine-tuning on the 5,000-image attacker holdout. | Behavioral removed at LR 0.05 and above; weight detected in all 12. |
| `attacks/p4.6_*` (24 files) | Pruning, then fine-tuning with the mask fixed. | Behavioral not detected in 23 of 24; both lost in 3 channel-pruning runs at 78.5–80.9% test accuracy. |
| `attacks/p4.7_*` (6 files) | Distillation into a fresh student. | Neither watermark detected in any student; see the headline in `ATTACK_FINDINGS.md`. |
| `attacks/p4.8_*` (11 files) | Overwriting with an attacker's own watermarks. | Owner's weight watermark detected in all 11; the attacker's watermarks are detected too. |
| `attacks/p4.5_apply/`, `p4.6_apply/`, `p4.7_apply/`, `p4.8_apply/` | Records of the Colab training runs behind those rows. (Colab T4) | Config, hashes and training history per run; the scored rows above are computed from their weights. |
| `p4.9_master_table`, [`p4.9_master_robustness_table.md`](p4.9_master_robustness_table.md) | All 85 attack rows in one place. | Aggregation only, no new measurement. Per-family counts are in `ATTACK_FINDINGS.md`. |
| [`../figures/p4.9_robustness_heatmap.png`](../figures/p4.9_robustness_heatmap.png) | The 85 rows as a heatmap. | From `p4.9_master_table`. |
| [`ATTACK_FINDINGS.md`](ATTACK_FINDINGS.md) | What survived, what did not, which watermark is stronger where. | Written from the rows above. |

## Phase 5: cryptographic identity

| File | Question it answers | Key finding |
|---|---|---|
| `p5.1_model_fingerprint` | Is the model fingerprint stable across save and reload? | Same fingerprint through 8 routes for all 4 models; dual `W*` is `c0995109…5b064a07`. Any changed weight changes it. |
| `p5.2_poseidon_validation` | Does the Python Poseidon match circomlib's? | 128 of 128 circomlibjs vectors, all 10,854 round constants and 16 MDS matrices equal. |
| `p5.4_commitment_publication` | What commitment was published? | `C` for the dual `W*`, artifact `provenance/commitment.json` (SHA-256 `cbdd82d9…fb9f231c`). Its date is the machine's clock. |
| `p5.5_timestamp` (two files) | When did the commitment exist, independently of the owner? | First record (20261002): proof pending, no Bitcoin attestation. Second (20261003): blocks 969627 and 969632 checked on two explorers, so it existed by block 969627. Not a full node. |
| `p5.5_timestamp_status` | Offline status of both proofs at the time. | Commitment proof upgraded with Bitcoin attestations; the record's proof still pending then. |
| `p5.6_opening_verifier` | Does revealing the secret open `C`, and only the right secret? | True opening accepted; 2,635 wrong openings, 0 accepted. A real opening reveals 79 secret bytes. |

## Phase 6: provenance record

| File | Question it answers | Key finding |
|---|---|---|
| `p6.2_signed_provenance_record` | Is the signed record intact and bound to its key? | Signature valid; 11 field changes and 512 signature bit flips, 0 accepted. The signature ties the record to a key, not the key to a person. |
| `p6.2_record_timestamp` | When did the signed record exist? | By Bitcoin block 969708, 81 blocks after the commitment's; two explorers, not a full node. |
| `p6.3_provenance_verifier` | Does the verifier catch tampering and forgeries? | Every tampered case fails; a forger's record under a fresh key fails only the trusted-key check. |
| `p6.4_theft_simulation` | What does an audit find after each simulated theft? | Only the verbatim copy matches the fingerprint; distillation and channel pruning plus fine-tuning leave no evidence. The backdated counter-claim has no Bitcoin time. |

## Phase 7: zero-knowledge, Track A (Circom, Groth16)

| File | Question it answers | Key finding |
|---|---|---|
| `p7.2_toolchain` | Which circom, snarkjs and circomlib? | circom 2.2.3, snarkjs 0.7.6, circomlib 2.0.5, hash-checked. |
| `p7.3_toy_poseidon_proof`, `zk/p7.3_toy/` | Does the full Groth16 loop work on a toy circuit? | Honest proof verifies; a wrong hash is refused. Local single-contributor setup, no soundness. |
| `p7.4_ptau` | Which powers-of-tau file, and was it checked? | Hermez power 15, accepted on its BLAKE2b-512 hash only; `powersoftau verify` stopped after 30 minutes. |
| `p7.5_commitment_circuit` | Does the circuit match the host commitment? | 1,471 constraints; the circomlibjs vector and the real opening are accepted, out-of-range inputs refused. |
| `p7.6_groth16_setup`, `zk/p7.6/verification_key.json` | Proving and verification keys. | Proving key 715,458 bytes, verification key 2,926 bytes. Single phase 2 contributor (the owner), so not sound against its creator. |
| `p7.7_groth16_proof`, `zk/p7.7/` | A proof of knowledge of an opening of the published `C`. | Verifies with the committed key; only public signal is `C`; no secret value in the committed files. |
| `p7.8_negative_tests`, `zk/p7.8/` | Does the verifier reject what it should? | 206 negative cases, 0 accepted; 26 controls verified. One case crashed the verifier rather than returning false. |
| `p7.9_sizing_estimate` | Can the circuit also prove the triggers come from `K`? | No: at least 281,529,423 constraints, far beyond the setup. P7.9 is blocked. |

## Phase 8: zero-knowledge, Track B (EZKL)

| File | Question it answers | Key finding |
|---|---|---|
| `p8.1_onnx_export`, `zk/p8.1/zk_model.onnx` | Does the ONNX export match PyTorch? | 10,000 of 10,000 test images agree on top-1; largest logit difference 1.53e-5. |
| `p8.2_ezkl_example` (three files) | Does EZKL's official example run here? | 20261003T1821 and T1826: failed, each recording one cause. T1831: passed with two documented one-line changes. |
| `p8.3_ezkl_setup`, `zk/p8.3/` | Circuit size and setup cost. | Logrows 18, 145,820 rows; setup peak 1.84 GiB. Settings and verification key committed; proving key not. |
| `p8.4_ezkl_prove`, `zk/p8.4/` | A proof of inference on MNIST test image 0. | 3,072-byte proof; circuit and PyTorch both give class 7. |
| `p8.5_ezkl_verify` | Does the proof verify, and are tampered ones rejected? | Verifies; 42 tampered variants, 0 accepted, every rejection an exception rather than `False`. |
| `p8.6_ezkl_fidelity` | Does the quantised circuit agree with PyTorch? | 10,000 of 10,000 MNIST test images agree on top-1, one model and one set of scales. |

## Phase 9: auditor and dashboard

| File | Question it answers | Key finding |
|---|---|---|
| `p9.1_audit_engine` | Does the audit engine run and keep secrets out of verdicts? | Record precondition passes; no checks wired yet at this stage. |
| `p9.2_audit_suspects` | The five checks on ten suspects. | Thefts reproduce their Phase 4 rows; clean `W` and an untrained model are not detected on either test. |
| `p9.3_graded_verdicts` | How strong is the technical evidence for each suspect? | Graded from the P9.2 p-values; clean `W` and the untrained model grade "none". "None" is not exoneration. |
| `p9.4_auditor_model_classes` | Does the auditor call unrelated models watermarked? | 0 of 41 unrelated models graded above "none"; all 85 attack rows reproduce their grades. |
| `p9.11_demo_trigger_gallery`, [`../figures/p9.11_demo_triggers/`](../figures/p9.11_demo_triggers/) | Demo-key triggers for the dashboard. | 12 triggers with base and perturbation, regenerated byte-identical to P1.3's demo set. Never the real triggers. |
| `p9.15_dual_w_star_bundle`, `p9.15_dual_W_star.npz` | The model bundled for the hosted demo. | Every array bit-identical to the dual `W*`; its fingerprint equals the published one. |
| `p9.15_predeploy_check` (two files) | Is the deploy tree free of secrets? | Both passed all checks. The first (T133254) flags the committed ONNX file for an owner decision; the second (T134647) records it kept by that decision. |
