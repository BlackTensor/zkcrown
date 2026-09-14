# CLAUDE.md

## zk-Crown: Neural Watermarking, Attack Lab & Zero-Knowledge Provenance

This file is the single source of truth for the project. It is both the plan and the progress tracker.

---

# 0. HOW WE WORK (read this every session)

## 0.1 The working protocol

1. I give you **one task at a time**, usually by its task ID (e.g. "do P2.3").
2. You do **only that task**. You do not run ahead into later tasks, and you do not silently redesign earlier ones.
3. When a task is finished and verified, you edit this file and change its checkbox from `[ ]` to `[x]`, and append the measured result to the Results Ledger in Section 8 if the task produces a number.
4. If a task cannot be completed as written, do **not** fake it. Change the checkbox to `[!]`, write a one line note under it explaining the blocker, and tell me.
5. At the start of a session, read Sections 0, 1, 2 and the first unchecked task. Do not re-read everything.

## 0.2 Checkbox legend

- `[ ]` not started
- `[~]` in progress across sessions
- `[x]` done and verified
- `[!]` blocked or failed, note required underneath
- `[-]` deliberately skipped, reason required underneath

Tasks tagged `[GPU]` require Google Colab and follow the handoff protocol in Section 0.5. You never run these yourself.

## 0.3 Rules you must not break

- **No unmeasured claims.** Every number in the repo, README, or dashboard must come from a script in `experiments/` that produced it. If a number is not yet measured, write `TBD`.
- **No scope creep.** New ideas go into Section 9 (Icebox), not into the current task.
- **Reproducibility.** Every experiment takes a seed and writes a JSON result file. No result lives only in a notebook cell output.
- **Commit per task.** One task equals one commit, message format `P<phase>.<task>: <what changed>`.
- **Secrets stay secret.** The real secret key `K` is never committed. `secrets/` is gitignored. Committed keys are demo keys only, clearly labelled.
- **Honest framing.** Never write "unremovable", "court proof", "legally proves ownership", or "immune to pruning". See Section 7.

## 0.4 Definition of done for any task

A task is done when: the code runs end to end from a clean runtime, the result is written to a file, the checkbox is ticked, and the commit is made.

## 0.5 GPU HANDOFF PROTOCOL (mandatory)

**You do not run GPU training yourself. Ever.** All training happens on Google Colab, launched manually by me.

Any task marked `[GPU]` in the phase checklists below follows this protocol exactly:

1. Write all the code the run needs: model, data pipeline, training loop, checkpointing to Drive, result JSON writer, and a runnable entry point.
2. Write a matching `notebooks/<task_id>_colab.ipynb` that is a thin wrapper: Drive mount, `!nvidia-smi`, unzip, dependency install, one call into `src/`, and a save-results cell.
3. Package everything into a single zip at `handoff/<task_id>_colab.zip`.
4. Write `handoff/<task_id>_INSTRUCTIONS.md` containing: what to upload, which notebook to open, which cells to run in order, expected runtime, expected VRAM and RAM, what files come back, and where to put them.
5. **Stop and hand off.** End your turn with exactly this, filled in:

> **COLAB HANDOFF READY.**
> I have created the zip file: `handoff/<task_id>_colab.zip`
> **Upload it to Colab and run it now.**
> Notebook to open: `<name>.ipynb`
> Expected runtime: `<estimate>`
> Bring back: `<list of files>`

6. Then **wait**. Do not proceed to the next task. Do not simulate, estimate, or invent the results. Do not tick the checkbox.
7. When I return with the output files, you verify them, write the numbers into the Results Ledger, tick the checkbox, and commit.

Rules around this:

- Never leave a task half-handed-off. The zip must be complete and self contained, so I never have to patch code inside Colab.
- The zip must run top to bottom without me editing anything. If a path or a secret is needed, the instructions file says exactly what to set and where.
- Because Colab wipes its filesystem and times out at roughly 90 minutes idle, every `[GPU]` run must checkpoint to Drive every epoch and be resumable from the last checkpoint.
- If a run is estimated to exceed 2 hours, split it into multiple `[GPU]` handoffs rather than one long one.
- CPU-only work (crypto, commitments, Circom circuits, auditor logic, analysis of returned results) you do normally in this environment. No handoff needed.

---

# 1. WHAT THIS PROJECT IS

A system that embeds a secret ownership identity into a neural network, attacks that model the way a thief would, measures whether the identity survives, and then proves ownership cryptographically without revealing the secret.

Three pillars:

1. **Watermarking.** A behavioral (black box, trigger based) watermark and a weight (white box, spread spectrum) watermark.
2. **Attack laboratory.** Pruning, quantization, fine-tuning, and distillation applied to our own model, with measured survival rates.
3. **Cryptographic provenance.** A Poseidon commitment published before the simulated theft, a Groth16 proof that we know the committed secret, and an EZKL proof about model behavior.

The output is an **IP Auditor**: give it a suspect model, get a forensic report.

## 1.1 Terminology discipline

- `W` = clean weights. `W*` = watermarked weights.
- `K` = secret master key. `K` is **not** the trigger. Triggers are derived from `K`.
- `T = {T1..TN}` = the trigger set.
- `S` = ownership signature, cryptographically derived, not a plain string.
- `C` = published commitment.
- We only call something "zero-knowledge" once an actual ZK proof relation is implemented and verified.

---

# 2. ENVIRONMENT AND CONSTRAINTS

## 2.1 Hardware reality

Everything runs on **Google Colab free tier**. Verified constraints:

- T4 GPU, 16GB VRAM, roughly 12 to 13GB system RAM.
- Max 12 hour session, roughly 90 minute idle timeout, GPU not guaranteed at peak times.
- **The filesystem is wiped between sessions.** Google Drive is mounted at the top of every notebook and all artifacts are written there.

Implications you must respect:

- No training run may assume more than 2 uninterrupted hours. Checkpoint every epoch to Drive.
- Every notebook begins with a Drive mount cell and a `!nvidia-smi` cell (the accelerator is not guaranteed to be a T4).
- Nothing is CPU-RAM hungry beyond roughly 10GB. If a ZK circuit approaches that, the circuit is too big and gets shrunk.

## 2.2 The two model sizes (important design decision)

There are **two separate models** in this project and they must not be conflated:

| Model | Purpose | Dataset | Size |
|---|---|---|---|
| `main_model` | Watermarking science, attack lab, all robustness experiments | CIFAR-10 | Small CNN, free to be a few hundred thousand parameters |
| `zk_model` | The model we push through EZKL for a zkML proof | MNIST | Deliberately minimal, target under 10K parameters |

Reason: ZK proving cost scales brutally with model size. Published zkML work uses servers with hundreds of GB of RAM even for modest CNNs. EZKL's own maintainers cite MNIST-scale inference as the cheap case. So the attack science gets a real model, and the ZK demo gets a toy model, and the README says so plainly.

## 2.3 Toolchain (all free, all pip or npm installable)

Machine learning: `torch`, `torchvision`, `numpy`, `matplotlib`, `onnx`, `onnxruntime`.

Cryptography: `hashlib` (SHA-256), `cryptography` (signatures), a Poseidon implementation in Python for the host side.

Zero-knowledge, split into two tracks (this is a correction to the original plan):

- **Track A, custom statement proof:** `circom` + `circomlib` + `snarkjs` (Groth16), with a public Hermez powers-of-tau file. This proves "I know `K` such that `Poseidon(K) = C`". EZKL is the wrong tool for this because EZKL proves ONNX graphs, not arbitrary hash preimage relations.
- **Track B, zkML inference proof:** `ezkl` (pip), Halo2 backend, on `zk_model` only.

Interface: `streamlit`, deployed free on Streamlit Community Cloud.

Timestamping: OpenTimestamps or a GPG signed git tag. No blockchain spend, no testnet faucets.

**Explicitly out of scope:** on-chain EVM verifiers, `solc`, smart contracts. They add cost and friction and prove nothing extra for a portfolio.

## 2.4 Version pinning

EZKL is beta software under rapid development and its API changes between releases. The first time it works, pin the exact version in `requirements.txt` and record it in Section 8. Same for `circom`, `snarkjs`, and `circomlib`.

---

# 3. REPOSITORY STRUCTURE

```
zk-crown/
  CLAUDE.md
  README.md
  requirements.txt
  .gitignore
  secrets/               gitignored, holds real K
  src/
    models/              architectures
    watermark/           trigger gen, behavioral, weight embedding, extractor
    attacks/             prune, quantize, finetune, distill
    crypto/              hashing, commitments, poseidon, signing
    zk/
      circuits/          .circom files
      ezkl/              ONNX export and proving pipeline
    auditor/             the forensic verification engine
  experiments/           runnable scripts, one per experiment
  results/               JSON result files, committed
  figures/               generated plots, committed
  notebooks/             Colab entry points, thin wrappers over src/
  app/                   streamlit dashboard
  tests/
```

Rule: notebooks are thin. Logic lives in `src/`. A reviewer must be able to read `src/` without opening a notebook.

---

# 4. PHASE CHECKLIST: FOUNDATIONS

## Phase 0: Repo and baseline model

- [x] **P0.1** Create repo skeleton per Section 3, with `.gitignore` covering `secrets/`, `data/`, `*.pth`, `*.ptau`, `*.zkey`.
- [x] **P0.2** Write `src/utils/seeding.py` with a global seed helper, and `src/utils/results.py` that writes a standard JSON result record (name, timestamp, git hash, seed, params, metrics).
- [x] **P0.3** Write `notebooks/00_setup.ipynb`: Drive mount, `nvidia-smi`, dependency install, repo clone, smoke test.
  - Notebook written and validated; its logic was executed locally (Colab-only paths fall through their guards). The section 8.1 environment rows stay `TBD` until it is actually run on Colab, since those are measured numbers.
- [x] **P0.4** Implement `main_model` (small CNN) in `src/models/` for CIFAR-10.
- [x] **P0.5** `[GPU]` Train clean `W` to a reasonable baseline. Checkpoint per epoch to Drive.
  - Run on Colab T4, 60/60 epochs, not stopped early. Test accuracy 91.20%. `W` SHA-256 `54f112f4…22fdcf` matches the result JSON. The weights file is gitignored (`*.pt`) and lives locally at `results/p0.5_clean_baseline_W.pt` and on Drive.
- [x] **P0.6** Record baseline clean accuracy in the Results Ledger. This is the number every later accuracy drop is measured against.
  - Baseline is 91.20% top-1 on the CIFAR-10 test set, recorded in section 8.2 from the verified P0.5 result JSON.
- [x] **P0.7** `[GPU]` Implement `zk_model` (minimal MNIST CNN, target under 10K params), train it, record its accuracy and exact parameter count.
  - Run on Colab T4, 20/20 epochs, not stopped early. 6,138 params, 98.96% test accuracy. Weights SHA-256 `6bc298d0…9ddec4` matches the JSON. Re-evaluating the saved file on CPU gives exactly the final epoch's numbers (9,896/10,000, loss 0.033630), not the best epoch's 99.02%, so these are final-epoch weights.

## Phase 1: Triggers

- [x] **P1.1** Write `src/watermark/keygen.py`: derive a deterministic PRNG stream from `K`. Same `K` must always give the same triggers. Test this.
  - `K` is 32 bytes. Block `i` of the stream for a given purpose label is `HMAC-SHA256(K, "zk-crown/keystream/v1\0" || u16 len(label) || label || u64 i)`. 44 tests in `tests/test_keygen.py`. They include a known-answer vector checked independently with `openssl`, a check that output is the same in a fresh process with a different `PYTHONHASHSEED`, and tests for label domain separation and unbiased integer draws. Not tackled here: encoding `K` into the BN254 field (P5.3), and the cost of SHA-256 inside a circuit (P7.9).
- [x] **P1.2** Write `src/watermark/triggers.py`: generate a key-derived perturbation trigger set of size N (start N=100).
  - Trigger `i` is `clip(x_i + A * s_i, 0, 255)` on uint8 pixels. `x_i` is a key-selected, distinct image from the 45,000-image training split, never the holdout or test set. `s_i` is an independent key-derived ±1 pattern per trigger, 3,072 entries. `A` defaults to 16 levels; this is a starting value for P1.3 to judge by eye, not a tuned one. The set involves no floats, keeps its first `m` triggers unchanged when N grows, and ignores pool order. 25 tests pass, including a real CIFAR-10 run that stays inside the training split. Left open: target responses (P2.2), the amplitude check (P1.3), regeneration and independence tests (P1.4), and the written rationale (P1.5).
- [x] **P1.3** Visualize the trigger set to `figures/`. Confirm by eye that triggers are not trivially visible garbage and not invisible noise either.
  - `experiments/p1_3_visualize_triggers.py` writes three figures: `figures/p1.3_trigger_set.png` (all 100), `p1.3_trigger_detail.png` (base / trigger / stretched perturbation) and `p1.3_amplitude_sweep.png` (A = 4, 8, 16, 32, 64). It uses a **public demo key**, never `K`, because committed figures of the real triggers would publish them. Verdict by eye at 4x nearest-neighbour upscale: A = 16 shows clear grain, strongest on flat bright regions, and every object stays recognisable. A = 4 is near-invisible, A = 8 faint, A = 32 heavy, A = 64 garbage. Default A = 16 kept, no change to P1.2. This is my judgement from the figures; the owner should look too. At native 32x32 the grain is less visible.
- [x] **P1.4** Write a determinism test: regenerating from `K` reproduces byte-identical triggers; a different `K` gives a statistically independent set.
  - `tests/test_trigger_determinism.py`, 14 tests. Byte-identical regeneration is checked in the same process, in fresh processes with different `PYTHONHASHSEED`, against a pinned digest, with the global RNGs in different states, and against the digest in the committed P1.3 result (real CIFAR-10, demo key). Independence is tested on the two key-derived parts, base indices and sign patterns, which fix the set once the dataset and A are given. Keys compared: all 256 single-bit flips of a reference key, and all 120 pairs of 16 unrelated keys. The tests check base-index overlap against its hypergeometric null, and all 100x100 cross-key sign-pattern correlations against N(0,1). Every observed statistic was within 1.6 sd of its null; the thresholds are at least 6 sd out. Three further tests plant a dependence and confirm the checks catch it.
- [x] **P1.5** Decide and document the trigger design choice (patch vs additive noise vs learned) with a one paragraph rationale in `src/watermark/README.md`.
  - Chosen: additive key-derived noise with an independent pattern for each trigger. Learned triggers were rejected because they depend on `W` as well as `K` and are hard to regenerate exactly. A patch was rejected because one shared stamp makes trigger responses move together, which breaks the P2.8 test, and it is the shape reverse-engineering defences look for. The rationale cites only P1.3 and P1.4 measurements. Its unmeasured costs are listed openly: memorisation may be fragile under P4.5 and P4.7, noise may be weakened by blur or JPEG (not in the attack suite), and SHA-256 is expensive for P7.9.

---

# 5. PHASE CHECKLIST: WATERMARKING

## Phase 2: Behavioral watermark

- [x] **P2.1** Derive the ownership signature `S` cryptographically from `K` (not a hardcoded string). Document the derivation.
  - `src/watermark/signature.py`: `S` is the first 16 bytes of `KeyStream(K, "signature/v1/owner:" + owner_id)`, i.e. a 128-bit HMAC-SHA256 PRF of the owner identity under `K`. The derivation is documented in the module docstring and `src/watermark/README.md`.
  - 128 bits fits in one BN254 field element (P5.3). `S` comes as bytes, as an int, as 128 bits MSB-first, and as ±1 signs for P3.2.
  - `S` is explicitly a MAC-like tag, not a public-key signature, and is treated as secret.
  - 27 tests, including a known-answer vector checked with `openssl`.
- [x] **P2.2** Implement the trigger-to-target-response mapping. Document whether it is a single owner class or a per-trigger keyed response, and why.
  - `src/watermark/responses.py`: **per-trigger keyed response**, `t_i = (y_i + 1 + r_i) mod 10`, with `r_i` the i-th `randbelow(9)` draw from `KeyStream(K, "responses/v1/target-class")`. `t_i` is uniform over the 9 classes other than the base image's label `y_i`, and prefix-stable in N.
  - Why: for any model independent of `K`, each trigger fires with probability at most 1/9, independently across triggers. That bounds the fired count by `Binomial(N, 1/9)` for P2.8 whatever the model's accuracy or class bias. A single owner class has no model-independent bound and its fire events are correlated.
  - Excluding `y_i` means firing always contradicts the image. With uniform targets over all 10 classes, WDR would stay near 10% after watermark removal.
  - Rationale is in the module docstring and `src/watermark/README.md`. Targets depend on `K` only, not `S`.
  - 30 tests. They cover a formula check computed with raw `hmac`, a biased-model null check and the binomial spread of the fired count over 1,000 keys, plus a planted shared class that the spread check catches (sample variance 1,140 vs 9.95). No p-values, WDR or FPR computed (P2.4, P2.6, P2.8).
- [x] **P2.3** `[GPU]` Implement joint training: clean data plus trigger set, producing `W*`.
  - Run on Colab T4, commit `0aaea8e`, 60/60 epochs, not stopped early. Test accuracy 90.73%, holdout 91.14%. `W*` SHA-256 `be00f2b5…197222` matches the JSON. The run used trigger bundle `fbd65ec7…22baec8`, the same digest as the local `secrets/trigger_bundle.npz`. Re-evaluating the saved file on CPU gives exactly the final epoch's numbers (9,073/10,000, loss 0.308929), not the best epoch's (index 58, 90.82%), so these are final-epoch weights. The 100% trigger accuracy is a training diagnostic on the training triggers, not the P2.4 WDR. Trigger accuracy was not stable during training; it stayed at 100% from epoch index 43 onward. Weights file is gitignored and lives at `results/p2.3_behavioral_wm_W_star.pt` and on Drive.
- [x] **P2.4** Measure Watermark Detection Rate (WDR) on the trigger set.
  - `src/watermark/detection.py` defines WDR = k/N, where k counts triggers whose eval-mode top-1 prediction equals the keyed P2.2 target. Triggers use the P2.3 normalisation, with no augmentation. The committable summary holds aggregates only. `experiments/p2_4_measure_wdr.py` regenerates the triggers from `secrets/K.bin` and refuses to report unless their digest equals P2.3's `fbd65ec7…22baec8`. It loads `W*` and `W` only if their SHA-256 matches. CPU run.
  - **WDR of `W*` is 100% (100/100)**, with mean target probability 0.9995 (min 0.9965).
  - Controls, not FPR or p-values. Clean `W` on the same triggers: 3/100 fired, 60 classified as the base label, 37 as some other class. `W*` on the unperturbed base images: 1/100 fired, 98 correct. `W` on the base images: 1/100 fired, 97 correct.
  - Caveat: these are the images `W*` was trained on, so this measures retention of trained responses, not generalisation. 15 new tests, 257 in total, all pass. A repeat run gave an identical record apart from the timestamp.
- [x] **P2.5** Measure clean accuracy of `W*` and compute the accuracy drop against P0.6.
  - `experiments/p2_5_accuracy_drop.py` loads `W` and `W*` by hash and scores both per image on CPU. Test accuracy reproduces the Colab runs exactly: `W` 9,120/10,000 = 91.20%, `W*` 9,073/10,000 = 90.73%. Holdout: `W` 90.88%, `W*` 91.14%.
  - **Drop on the test set: 0.47 percentage points** (0.52% relative).
  - Paired analysis (`src/utils/stats.py`): `W` alone is right on 362 images and `W*` alone on 315. Exact McNemar p = 0.077, 95% CI for the drop [-0.04, +0.98] pp. So the drop cannot be told apart from zero at the 5% level on this test set. On the holdout the sign flips: -0.26 pp, CI [-1.01, +0.49], p = 0.53.
  - Not measured: seed-to-seed variation. Each model is a single training run. 18 new tests, 275 in total, all pass.
- [x] **P2.6** Measure False Positive Rate: run 1000 random and 1000 clean-but-unrelated inputs, count spurious watermark responses. **This is the credibility-critical number.** A high WDR is meaningless without a low FPR.
  - **Definition** (`src/watermark/false_positives.py`): non-trigger input `j` takes trigger slot `j mod 100`. It fires if the eval-mode top-1 prediction equals the owner's real target for that slot, the same rule as P2.4. FPR is fires out of 1,000.
  - The chance level is never zero, since every prediction is some class. Its expectation over slot assignments is `sum_j q(pred_j)`, where `q` is the owner target class frequency.
  - `experiments/p2_6_false_positive_rate.py` regenerates the targets from `K`, with the P2.3 digest checked, and hash-checks the weights. The input sets come from seed 1337: 1,000 uniform-noise images and 1,000 distinct CIFAR-10 test images. A third set, added beyond the task, is those test images plus ±16 sign noise independent of `K`.
  - **`W*` FPR: random 12.6% (126/1000, chance 127.3), clean-unrelated 10.0% (100/1000, chance 100.2).** Noise decoys 11.9% (119, chance 108.7).
  - Clean `W` for reference: 8.8%, 9.1%, 10.3%.
  - Against P2.4's 100% on triggers, these are near the chance level of each model's own predictions. Input-level FPR cannot go to zero; separating "watermarked" from "not" needs the count over N triggers, which is P2.8.
  - Observed, not tested:
    - On uniform noise, `W*`'s predictions fall more often in the owner's target classes than `W`'s do (chance level 12.73% vs 8.90%).
    - On clean images, 18 of `W*`'s fires contradict the label, against 10.1 expected. The slot assignment is unrelated to image content, so that gap is noise from a single fixed pairing, not a leakage signal.
    - `W*` is less accurate than `W` on the noise decoys (41.0% vs 47.5%).
  - 12 new tests, 287 in total, all pass. A repeat run gave an identical record apart from the timestamp.
- [x] **P2.7** `[GPU]` Sweep the trigger-to-clean data ratio, plot the WDR vs accuracy-drop tradeoff curve to `figures/`.
  - Five new Colab runs (commit `4c1e3da`, T4, clean tree), each 60/60 epochs, not stopped early. Each JSON checked for internal consistency: history, best and final epochs, epoch time, trigger sample count, bundle digest, and params identical to P2.3 apart from the trigger fields. All five weights files match their recorded SHA-256. P2.3 and clean `W` are reused as the 3.13% and 0% points.
  - `experiments/p2_7_analyze_sweep.py`, CPU, measures WDR with P2.4's method (triggers regenerated from `K`, digest `fbd65ec7…22baec8`) and the drop with P2.5's paired method. All CPU test accuracies reproduce the recorded Colab accuracies exactly.
  - **WDR by trigger samples per clean sample:** 0% → 3/100, 0.013% → 9, 0.049% → 19, 0.196% → 64, 0.782% → 100, 3.13% → 100, 12.5% → 100.
  - **Test accuracy drops vs `W`, same order:** +0.19, +0.46, +0.25, +0.94, +0.47 and +0.34 pp. Only the 0.782% run's drop is distinguishable from zero (95% CI [+0.42, +1.46], McNemar p = 0.0005). The drop does not rise with the ratio: the 12.5% run, with 16x the triggers, dropped 0.34 pp.
  - One run per ratio, so the shape of the drop curve cannot be separated from seed-to-seed variation. WDR is on the training triggers; no p-values for the low-WDR models (P2.8). No ratio was re-selected; P2.3's `W*` stays the model going forward.
  - Figure `figures/p2.7_wdr_vs_accuracy_drop.png`. A repeat analysis run gave an identical record and a byte-identical figure.
- [x] **P2.8** Write the statistical detection test: given k of N triggers firing, what is the p-value under the null hypothesis of an unwatermarked model? Ownership evidence must be a statistical statement, not a vibe.
  - **Test** (`src/watermark/significance.py`). H0 is that the suspect model is independent of the owner's keyed targets. The p-value is the exact tail `P(Binomial(N, 1/9) >= k)`, in rational arithmetic, with log10 reported. It is valid for every H0 model whatever its accuracy or class bias, because P2.2's construction makes each fire an independent Bernoulli with probability at most 1/9. It is conservative for models that often predict the base label.
  - Assumptions are written down: `K` and the triggers were fixed before the suspect was seen (the P5 commitment), one pre-declared test, one query per trigger, and a correction for multiple suspects. So is what the p-value does not say.
  - **Thresholds for N = 100:** `k* = 17 / 20 / 23 / 29 / 35` at alpha 0.05 / 0.01 / 1e-3 / 1e-6 / 1e-9. The exact false-positive bounds there are 0.049, 0.0066, 5.3e-4, 8.7e-7 and 2.7e-10.
  - **p-values** (`experiments/p2_8_detection_test.py`, CPU):
    - `W*` 100/100 → 3.8e-96, re-measured from `K` with the digest checked.
    - Clean `W` 3/100 → 0.999.
    - P2.7 sweep, from its committed counts: 9/100 → 0.79, 19/100 → 0.014, 64/100 → 2.6e-36, 100/100 → 3.8e-96 (both 0.782% and 12.5%).
  - **Empirical null check, 1,000 public wrong keys**, each building its own triggers and targets. Neither model rejected on any key at 0.05 or 0.01 (bounds 49.3 and 6.6 keys).
    - Mean fired: `W` 5.70, `W*` 6.38, against the bound's 11.11. Largest count: 14 for `W`, 16 for `W*`.
    - Given each model's base-label hits, the expected totals were 5,698.0 and 6,482.2. The observed totals were 5,701 (z = +0.04) and 6,379 (z = -1.36).
    - The check covers alpha around 0.05 and 0.01 only; smaller levels rest on the proof.
  - 33 new tests, 340 in total, all pass. They include exact Poisson-binomial validity checks for null models, a planted too-small bound that the validity check catches, and an end-to-end run over 2,000 test keys.
  - Reproducibility: the committed result was re-run from a clean tree at commit `4c90006`, so it reads `dirty: false`. Its params, metrics and environment are identical to the earlier dirty-tree run, and so is the figure, byte for byte. Only the timestamp, duration and git block differ. The dirty record was removed.

## Phase 3: Weight watermark

- [x] **P3.1** Implement the key-derived pseudo-random projection `P_K`.
  - `src/watermark/projection.py`: `P_K = sigma / sqrt(dim)`, a dense 128 x `dim` matrix of key-derived ±1 signs with unit-norm rows. Row `i` is the next `ceil(dim/8)` bytes of `KeyStream(K, "projection/v1/rademacher/dim=<dim>")`, bits MSB first, bit 1 → +1, the same layout as the trigger signs.
  - Decisions: Rademacher signs rather than Gaussian, and no QR orthonormalisation, so `P_K` is exact integers and byte-identical everywhere. `dim` is bound into the label. Rows are prefix-stable. `project` (`P_K w`) and `back_project` (`P_K^T c`) use float64 in row chunks. `repr` hides the signs.
  - 41 tests, 381 in total, all pass. They include a known-answer vector checked with `openssl`, a raw-`hmac` formula check, a pinned digest and a fresh-process check at `dim` = 307,040, and domain separation from other keys, other `dim` values and the trigger stream.
  - Null checks at `dim` = 307,040 with a test key. The largest off-diagonal Gram z was 3.72 over 8,128 pairs, and the largest cross-key z was 3.40. `P_K P_K^T c` returned `c` with every sign correct and a maximum error of 0.075, about 3.7 cross-talk sd. The test thresholds are 6 to 7 sd out, and a planted dependent row is caught.
  - Not decided here: the carrier parameters and `alpha` (P3.2), and centring the carrier before projection (P3.3). Rows have unbalanced sums, so a non-zero carrier mean leaks into `P_K w`. No ledger numbers.
- [x] **P3.2** Implement spread-spectrum embedding: `W* = W + alpha * P_K^T * S`, spread across many parameters rather than concentrated.
  - `src/watermark/carrier.py`: the carrier is every conv and linear `weight`, in `named_modules()` order, flattened C-order. For `main_model` that is 7 tensors, `dim` = 307,040. It excludes the 896 BN affine parameters, the 10 classifier biases and the BN buffers. `CarrierLayout` supports `flatten`, `split`, `check` and a SHA-256 `digest` of its names and shapes.
  - `src/watermark/weight_embedding.py`: `embed_weight_watermark(state_dict, layout, P_K, S, alpha)` returns a new state_dict. The carrier gets `W + alpha * P_K^T * S`, with `S` as ±1 signs; everything else is bit-identical, and the input is not modified. It also returns an aggregate-only `EmbeddingSummary`: change L2, RMS and max, relative size, zero fraction, per-tensor RMS and energy share, and the measured float32 rounding error.
  - `alpha` is the per-bit amplitude in projected space, since `P_K w* = P_K w + alpha S + cross-talk`. It has no default. Each parameter moves by `alpha/sqrt(dim)` times a sum of 128 fair signs: same distribution in every layer, about `alpha * sqrt(128/dim)` RMS, exactly zero for about 7.0% of parameters.
  - 42 tests, 423 in total, all pass. Checks on a seeded `main_model` with a test key:
    - Exact formula, carrier change equal to the delta up to float32 rounding (1.4e-8 at alpha 0.05), non-carrier entries bit-identical, strict load and a finite forward pass, and `P_K(W* - W)/alpha` recovering every sign of `S`.
    - Spread: every tensor's energy share within 1.05 sd of its size share, the largest single parameter at 24.5/dim of the energy, the top 1% of parameters at 8.4%, and a zero fraction of 7.04%.
  - Not done here: choosing `alpha` and measuring accuracy (P3.5), post-hoc vs during training and BN recalibration (P3.6), extraction (P3.3). No ledger numbers. Icebox line added for per-layer scaled embedding.
- [x] **P3.3** Implement the extractor: recover the fingerprint from weights and compute correlation with the expected signature.
  - `src/watermark/weight_extraction.py`: `extract_weight_watermark(state_dict, layout, P_K, S)` computes `y = P_K c`, where `c` is the suspect carrier with each tensor's mean subtracted. It returns `correlation = <y,S>/(||y|| sqrt(128))` (primary), `amplitude = <y,S>/128`, `projected_rms` and `bit_matches` (bits where `y_i > 0` equals `S`).
  - Decisions:
    - **Blind:** no clean `W` is used, so the extractor still works if P3.6 embeds during training.
    - **Per-tensor centring:** removes each layer's mean leaking in through the unbalanced row sums of `P_K`, at the cost of 7 of 307,040 watermark directions.
    - **Normalised correlation:** unchanged by global rescaling.
    - **All-zero fingerprint:** scores 0, i.e. no evidence.
    - `y` and the recovered bits are hidden in `repr` and left out of `to_dict`.
  - 27 tests, 450 in total, all pass. They use a seeded `main_model` and test keys:
    - Formulas, and all 128 bits recovered at a strong test alpha, with `amplitude` within 6 host sd of alpha.
    - Exact linear host-plus-watermark decomposition, and centring costing only cross-talk.
    - Correlation increasing with alpha.
    - Invariance to global rescaling and per-tensor shifts (without centring, the shifts leak in); non-carrier tensors ignored; FP16 weights giving the same bits.
    - No-watermark, wrong-key and wrong-owner correlations within 6/sqrt(128) at unit level.
  - Not done here: the real-model correct vs wrong `K` gap (P3.4), the alpha sweep (P3.5), the null distribution and threshold (P3.7), and per-layer rescaling attacks (Phase 4). No ledger numbers.
- [x] **P3.4** Verify extraction succeeds with the correct `K` and fails with a wrong `K`. Report the correlation gap between the two cases.
  - `experiments/p3_4_key_specificity.py`, CPU, 274 s, clean tree at `2a2eee9`. The owner id is fixed by the owner as `PROJECT_OWNER_ID = "blacktensor-zkcrown-owner"` in `src/watermark/signature.py`.
  - Method:
    - Hosts, loaded by hash: P2.3 `W*` and P0.5 `W`.
    - Embedding: post-hoc (P3.2) with `K`, at alpha 0 (control), 0.005, 0.01, 0.02, 0.05 and 0.1. The grid was fixed before the run; it is not a selection and has no accuracy check.
    - Extraction (P3.3) three ways: with `K`; with 100 public wrong keys, each deriving its own `P_K'` and `S'`; and with the wrong `P_K'` against the true `S`.
  - **`W*` host, correct-key correlation by alpha:** 0 → +0.123, 0.005 → +0.221, 0.01 → +0.313, 0.02 → +0.473, 0.05 → +0.756, 0.1 → +0.909. Bit matches: 68, 75, 80, 94, 110, 126 of 128.
  - **Wrong keys on `W*`:** mean correlation +0.001, sd 0.095, range [-0.18, +0.24], max |z| 2.71. These barely move with alpha. Wrong `P_K'` against the true `S` gives max |z| 2.6.
  - **Gap on `W*`** (correct minus wrong mean): +0.12, +0.22, +0.31, +0.47, +0.76, +0.91, i.e. 1.3, 2.3, 3.3, 5.0, 8.0, 9.6 wrong-key sd.
    - The correct key is above all 100 wrong keys from alpha 0.01 up.
    - The alpha-0 control is not separated. Its +0.12 (z = 1.39) is the host's chance projection on `K`, and it also lifts the small-alpha `W*` points.
  - **Clean `W` host:** correlations +0.016, +0.138, +0.253, +0.453, +0.779, +0.927. Wrong keys: mean +0.014, sd 0.088, max |z| 3.01. Gap 0.0, 1.4, 2.7, 5.0, 8.8, 10.5 sd. Above all wrong keys from alpha 0.02 up.
  - Reading: extraction is key-specific. A wrong key gives a null-looking correlation whether or not the model carries our watermark. The correct key separates from 100 wrong keys once alpha is about 0.02 or more. Below that, the host term (projected RMS 0.049 on `W*`, 0.041 on `W`) dominates.
  - Finding: normalised correlation is at most 1, so z = corr·sqrt(128) cannot exceed 11.3. At alpha 0.1 it is already at 10.3 to 10.5, so the evidence this statistic can give is capped. This matters for P3.7.
  - No threshold or p-value (P3.7), no accuracy (P3.5). 10 new tests, 460 in total, all pass.
- [x] **P3.5** Sweep embedding strength `alpha`, plot detection confidence vs accuracy drop.
  - `experiments/p3_5_alpha_sweep.py`, CPU, 457 s. Committed result from a clean tree at `e943c19`. Setup:
    - Host `W*` (P2.3), embedded post-hoc (P3.2) with `K` and the project owner id, no BN recalibration.
    - 12 alphas, fixed before the run: 0, 0.005, 0.01, 0.02, 0.03, 0.05, 0.075, 0.1, 0.15, 0.2, 0.3, 0.5.
    - Detection by blind extraction (P3.3), with a reference from 50 wrong keys, the first 50 of P3.4's family.
    - Accuracy scored per image on test (10,000) and holdout (5,000). Drops use P2.5's paired method against `W*` (the weight watermark's own cost) and against `W` (total vs P0.6).
  - Built-in checks passed: alpha 0 reproduces `W*`'s recorded 90.73% / 91.14% exactly, and the correct-key correlations at the six alphas shared with P3.4 equal P3.4's committed values exactly.
  - **Detection z (corr·sqrt(128)) by alpha:** 1.39, 2.50, 3.55, 5.35, 6.75, 8.55, 9.71, 10.29, 10.80, 11.01, 11.17, 11.26. All 128 bits are recovered from alpha 0.15 up. Wrong keys: max |z| 2.03 to 2.12, sd of correlation 0.095 to 0.098 at every alpha.
  - **Test drop vs `W*`, same order:** 0, -0.02, -0.06, -0.08, -0.09, -0.09, -0.11, -0.12, -0.04, 0.00, +0.19, +1.22 pp. Only alpha 0.5 is distinguishable from zero: 95% CI [+0.79, +1.65], McNemar p = 2.6e-8.
  - **Holdout drop vs `W*`:** within ±0.22 pp and not significant up to alpha 0.2. It is +0.54 pp at 0.3 (CI [+0.08, +1.00], p = 0.027) and +1.74 pp at 0.5 (p = 1.6e-8).
  - Reading: in this run, detection rises steeply up to alpha 0.1 and then saturates against the z cap of 11.31. Accuracy shows no measurable cost up to 0.2 on either split. The cost appears at 0.3 on the holdout and at 0.5 on both.
    - At 0.1 the change is 4.6% of the carrier L2 norm, and 7.1% relative RMS in `features.17`.
    - Test loss rises from alpha 0.15 on: 0.3078 at 0.1, 0.3089, 0.3111, 0.3188, then 0.3509 at 0.5.
  - Limits: one host, one key, one run. The intervals cover image sampling only.
  - No alpha selected; that is P3.6. Picking alpha from these test drops would bias that model's test accuracy, and the ledger would have to say so.
  - Figure `figures/p3.5_alpha_sweep.png`. A first run of the same code was flagged dirty only because its new figure was untracked when the record was written. It gave identical metrics and was discarded, not committed. 6 new tests, 466 in total, all pass.
- [x] **P3.6** `[GPU]` Decide whether the weight watermark is embedded post-hoc or during training, document the choice, and produce the final dual-watermarked model `W*`.
  - **Decision (owner): post-hoc at alpha 0.1**, into the P2.3 behavioral model, without BN recalibration. No GPU run was needed: the `[GPU]` tag covered the during-training option, which was not chosen. The rationale is in `src/watermark/README.md`. During-training embedding is in the Icebox, conditional on Phase 4.
  - `experiments/p3_6_make_dual_model.py`, CPU, 78 s, clean tree at `82c0241`.
    - It saves `results/p3.6_dual_wm_W_star.pt` (gitignored) with SHA-256 `7a9a9f141b55c7894b208c1f338f4c7385362f969b82b319ce5893fb4bb434c4`. Saving a second time under the same file name gives the same bytes; `torch.save` embeds the file name in the archive.
    - All checks below are on the weights reloaded strictly from that file.
  - **Weight watermark:** correlation +0.9091, z = 10.29, 126/128 bits. That equals P3.5's alpha 0.1 row exactly. The P2.3 model without it gives z = 1.39. The change is 4.56% of the carrier L2 norm, and the largest float32 rounding error is 2.8e-8.
  - **Behavioral watermark, gate fixed before the run** (P2.8 test must reject at 1e-6, i.e. fired >= 29):
    - The dual model fires on **100/100** triggers, p = 3.8e-96, so it passes. Mean target probability is 0.9994 (min 0.9932), against 0.9995 (min 0.9965) for the P2.3 model.
    - Base-image control: 1/100 fired and 99 correct (P2.3 model: 1 fired, 98 correct).
    - Trigger bundle digest matched P2.3.
  - **Accuracy** (reproduces P3.5's counts exactly):
    - Test 90.85% (9,085/10,000): drop vs the P2.3 model -0.12 pp, CI [-0.32, +0.08], p = 0.27; vs clean `W` +0.35 pp, CI [-0.16, +0.86], p = 0.19.
    - Holdout 91.16% (4,558/5,000): drop vs the P2.3 model -0.02 pp, p = 1.0; vs `W` -0.28 pp, p = 0.50.
    - Alpha was chosen after seeing P3.5's test drops, so the test figure is optimistic.
  - Limits: one host, one key, one embedding. No weight-watermark threshold or false-positive rate yet (P3.7). 8 new tests, 474 in total, all pass.
- [x] **P3.7** Null distribution: extract with 1000 random wrong keys, fit the correlation null distribution, derive a detection threshold with a stated false positive rate.
  - **Test** (`src/watermark/weight_significance.py`). H0: the suspect is independent of `K`.
    - Statistic: z = correlation·sqrt(128).
    - `S` and `P_K` come from separate PRF streams, so under H0 `S` is uniform and independent of `y = P_K c`. Given `y`, z is a Rademacher sum with unit norm, and Hoeffding gives **P(z ≥ t) ≤ exp(-t²/2) for every H0 model**. This is a proof, like P2.8's bound.
    - p-value `min(1, exp(-z²/2))`, one-sided. Threshold **z\* = sqrt(2 ln(1/alpha)), false positive rate ≤ alpha**.
    - Floor: `exp(-64)` = 1.6e-28, reached only at correlation 1. Assumptions are written down as in P2.8.
  - **Thresholds:** z\* = 2.448 / 3.035 / 3.717 / 5.257 / 6.438 at alpha 0.05 / 0.01 / 1e-3 / 1e-6 / 1e-9. In correlation: 0.216, 0.268, 0.329, 0.465, 0.569. The Gaussian approximation would give lower values (1.645 … 5.998), but that is not a proven bound, and the exact tests show it fails for a small equal-weight sum. It is reported only.
  - **Empirical null** (`experiments/p3_7_weight_null.py`, CPU with 8 workers, 368 s, clean tree at `a757281`):
    - Method: 1,000 public wrong keys, each deriving its own `P_K'` and `S'`, on clean `W`, the P2.3 behavioral-only model and the P3.6 dual `W*`.
    - z mean -0.019 / +0.021 / +0.020 and sd 0.967 / 1.009 / 1.008, all 95% sd intervals covering 1. Skew and excess kurtosis are within 2 se of 0. KS against the fixed N(0, 1): p = 0.45 / 0.64 / 0.50.
    - Fitted Gaussian thresholds at 1e-6: 4.58 / 4.82 / 4.81 (descriptive only).
    - Largest z 2.64 / 3.21 / 3.22. Exceedances at z\*(0.05) 4 / 8 / 8 (bound 50); at z\*(0.01) 0 / 3 / 4 (bound 10); none at 1e-3 or below.
    - Bit matches: mean 63.9 to 64.1, variance 30.4 to 32.1, against Binomial(128, 1/2) at 64 and 32.
    - The behavioral-only and dual models share keys and their null z correlate at 0.999, so they are not independent checks. The samples are not pooled.
  - **Applied:**
    - Owner `K` on clean `W`: z = 0.18, p ≤ 0.98. On the behavioral-only model: z = 1.39, p ≤ 0.38. Neither is detected.
    - **Dual `W*`: z = 10.29, p ≤ 1.1e-23**, rejecting at 1e-9. It reproduces P3.6 exactly.
    - P3.5 sweep: alpha 0.005 rejects only at 0.05 (p ≤ 0.044), 0.01 at 1e-3 (p ≤ 0.0019), 0.02 at 1e-6 (p ≤ 6.2e-7), and 0.03 and above at 1e-9.
  - Limits: 1,000 keys check the false positive rate only near 0.05 and 0.01; smaller levels rest on the proof. The test is per suspect, and combining it with the behavioral test is P9. Figure `figures/p3.7_weight_null.png`. 29 new tests, 503 in total, all pass.

---

# 6. PHASE CHECKLIST: ATTACKS, CRYPTO, ZK

## Phase 4: Attack laboratory

Each attack task must report, in one table row: attack strength, resulting clean accuracy, behavioral WDR, weight-watermark correlation, and the p-value from P2.8 / P3.7.

- [x] **P4.1** Build the harness `experiments/run_attack_suite.py`: takes a model plus an attack config, writes a standard JSON row. All following tasks use this harness.
  - **Interface** (`src/attacks/harness.py`): an attack is a function registered by name with `register_attack`. It gets a private copy of the source weights, the arch, one swept `strength` and `params`, and an `AttackContext` (device, seed, data root). It returns `AttackOutput(state_dict, arch, info)`.
    - `AttackContext` has no key field, so **an attack never sees `K`**.
    - `apply_attack` seeds every backend first, never modifies the source, and checks that the output loads strictly into `main_model(**arch)`. A narrower student is allowed.
    - Config JSON: `{"attack", "strengths": [...], "params"}`, one row per strength. Unknown keys are refused.
    - Only `none`, the control, is registered here. P4.2 to P4.8 add their own attacks.
  - **Row** (`src/attacks/evaluation.py`), using methods already fixed and nothing new:
    - Test-set accuracy with P2.5's paired drop against the source. The holdout is not scored, because P4.5 and P4.6 train on it.
    - P2.4 WDR with the P2.8 p-value, on triggers regenerated from `K`. The bundle digest must match P2.3.
    - P3.3 correlation with the P3.7 bound. If the owner's carrier layout is gone (a student, removed channels), the row says `applicable: false` with the reason rather than guessing.
    - `table` holds the Phase 4 columns. The row holds aggregates only, and a test checks that.
    - `detected` flags use **alpha 1e-6, fixed here before any attack ran**. The two tests are reported separately; combining them is P9.3.
  - **Modes:** `run` (apply and score locally); `apply` (saves attacked weights plus a hash record, never reads `K`, for `[GPU]` notebooks); `evaluate` (scores returned weights locally, hash-checked). Sources are only the committed `dual` (default), `behavioral` and `clean` models, each loaded by hash. Rows go to `results/attacks/`.
  - **Control run** (CPU, 46 s, clean tree at `b807c8b`): `none` on the dual `W*` gives test 90.85%, drop 0, WDR 100/100 (p = 3.8e-96) and correlation +0.9091, z = 10.29 (p ≤ 1.1e-23). Both watermarks are detected at 1e-6.
    - Built-in checks passed: zero discordant images, and exact reproduction of P3.6 (fired, correlation to 1e-12, 9,085 correct).
    - An `apply` then `evaluate` run of the same control, written to a scratch directory and not committed, gave an identical table.
  - 43 new tests, 546 in total, all pass.
- [x] **P4.2** `[GPU]` Magnitude pruning sweep (10 to 90 percent sparsity).
  - **Owner decisions:**
    - Run on local CPU with no Colab run: pruning without fine-tuning is not training.
    - Sweep both scopes.
  - **Attacks** (`src/attacks/prune.py`):
    - `magnitude_prune_layerwise` zeroes exactly `round(s·n)` smallest-|w| entries of each conv/linear weight tensor.
    - `magnitude_prune_global` uses one threshold over all 307,040 of those weights.
    - Common to both: stable float64 ranking, so ties go in index order and exact counts are reproducible. Kept weights stay bit-identical. BN parameters, BN buffers and the classifier bias are never touched. No fine-tuning (P4.6), no BN recalibration, no data.
    - The pruned set is also the watermark carrier. That is because it is the conventional pruning target, not because the attacker knows the carrier.
  - Configs `experiments/configs/p4.2_magnitude_prune_{layerwise,global}.json`: s = 0.1 … 0.9, fixed before the run.
  - Harness fix: `write_result` takes a `git` snapshot, and `run_attack_suite.py` takes one per invocation and accepts several configs. Without this, the first row a sweep writes (untracked) would mark every later row in the same sweep dirty.
  - **Run:** 18 rows on the dual `W*`, CPU, about 5 min, clean tree at `506d99c`. Every row's achieved sparsity equals its target, and no tensor was emptied.
  - **Layer-wise, s = 0.1 … 0.9:**
    - Test accuracy: 90.83, 90.83, 90.22, 89.24, 86.92, 74.23, 50.79, 25.08, 11.63%.
    - Fired: 100, 100, 100, 100, 99, 71, 23, 15, 15.
    - Weight z: 10.27, 10.25, 10.24, 10.20, 10.22, 10.15, 9.99, 9.81, 9.36.
  - **Global, s = 0.1 … 0.9:**
    - Test accuracy: 90.86, 90.76, 90.74, 90.14, 89.26, 87.59, 80.33, 49.27, 18.28%.
    - Fired: 100, 100, 100, 100, 100, 97, 78, 42, 12.
    - Weight z: 10.27, 10.26, 10.23, 10.18, 10.13, 10.04, 9.83, 9.29, 8.17.
  - **Reading:**
    - The first accuracy drop distinguishable from zero is at 30% (layer-wise, +0.63 pp) and 40% (global, +0.71 pp).
    - The behavioral watermark is detected at 1e-6 up to layer-wise 60% (74.2% accuracy) and global 80% (49.3% accuracy). It fails only after accuracy has fallen to 50.8% or below.
    - The weight watermark is detected at 1e-6 at every sparsity in both scopes, lowest z 8.17 (p ≤ 3.1e-15), including models near chance accuracy. Why it holds up is not investigated here.
  - Limits: one model, one key, one deterministic pruning per point, no recovery training.
  - 28 new tests, 574 in total, all pass.
- [x] **P4.3** `[GPU]` Structured pruning (whole channels or filters). Expect this to hurt more than magnitude pruning. Report it either way.
  - **Owner decisions:**
    - Run on local CPU with no Colab run, since there is no training (as in P4.2).
    - Removed channels are zero-masked in the same shapes.
  - **Attack** (`src/attacks/structured_prune.py`, `channel_prune_l1`):
    - Each of the 6 conv layers loses `min(round(s·C), C-1)` output channels, the filters with the smallest L1 norm (Li et al. 2017).
    - Filters are ranked on the source weights, each layer independently, with stable ties in index order.
    - A removed channel zeroes its filter, its BN weight and bias, and the next layer's input slice (the classifier's 16 columns after the last conv).
    - BN running statistics, kept weights and the classifier bias are untouched. No fine-tuning, no BN recalibration, no data.
    - A test checks that the zero-masked model matches a hand-sliced, physically narrower network (logits within 1e-5, identical argmax) and that the narrower network's parameter count equals the one `info` records.
    - **Caveat:** zero-masking keeps the owner's carrier layout. A thief who physically deletes the channels ships a narrower model, and the owner would first have to re-align the removed positions as zeros. That alignment is not implemented, so the weight-watermark numbers here assume it has been done.
  - Config `experiments/configs/p4.3_channel_prune_l1.json`: s = 0.05, 0.1, 0.2 … 0.9, fixed before the run. 0.05 was added because structured pruning was expected to fail early.
  - **Run:** 10 rows on the dual `W*`, CPU, about 3 min, clean tree at `ccfc044`.
    - No layer was capped.
    - Weights zeroed exceed the channel fraction, because middle layers lose both outputs and inputs: 9.0% of the carrier at s = 0.05, 18.2% at 0.1, 73.3% at 0.5.
  - **s = 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9:**
    - Test accuracy: 86.02, 74.74, 46.92, 30.70, 16.90, 15.81, 10.01, 11.00, 9.97, 10.00%.
    - Fired: 94, 70, 19, 12, 9, 10, 10, 12, 8, 10.
    - Weight z: 10.09, 9.90, 9.59, 9.37, 9.04, 8.22, 7.10, 5.47, 3.73, 2.08.
  - **Reading:**
    - It hurts far more than magnitude pruning. Removing 5% of channels costs 4.83 pp (CI [+4.28, +5.38]), and the model is at chance from s = 0.6.
    - Compared at a similar fraction of weights zeroed: 9.0% zeroed → 86.02% here, against 90.83% at P4.2 layer-wise 10%. 18.2% zeroed → 74.74%, against 90.83% at layer-wise 20%.
    - Behavioral watermark detected at 1e-6 at s = 0.05 (94/100) and 0.1 (70/100, 74.7% accuracy). Lost from 0.2 (19/100, p = 0.014, accuracy 46.9%).
    - Weight watermark detected at 1e-6 up to s = 0.7, and lost at 0.8 (z 3.73) and 0.9 (z 2.08). From s = 0.4 on the model is at or near chance (≤ 16.9%). Detection there has little practical meaning, and it rests on the re-alignment caveat.
    - Unlike P4.2, the weight z does fall steadily, as whole carrier slices go to zero.
  - Limits: one source model, one key, one criterion (L1, layer-wise), one deterministic pruning per point, no recovery training. Global and BN-scale criteria were not run.
  - 25 new tests, 599 in total, all pass.
- [x] **P4.4** `[GPU]` Post-training quantization, FP32 to INT8. Also try FP16.
  - **Owner decisions:**
    - Local CPU run.
    - INT8 is simulated in plain PyTorch, not run on a quantized backend.
    - Scope is the task only: FP16, INT8, plus a fused-FP32 control.
    - Weight extraction reads the shipped fused weights; fusion is not undone.
  - **Why simulated:** the eager `torch.ao.quantization` API warns it is deprecated and slated for removal. The only local engine is oneDNN on ARM, whose full-range kernels saturated on a seeded untrained model in an exploratory check (not a ledger number).
  - **Attacks** (`src/attacks/quantize.py`); strength is the bit width:
    - `ptq_fp16` (16): every floating tensor cast to float16, no fusion. Inference is float32 on the rounded values, so activations are not float16.
    - `ptq_fused_fp32` (32): conv-BN-ReLU fusion with no rounding, the control separating fusion from quantization.
    - `ptq_int8_static` (8): fusion, then the eager static PTQ recipe:
      - qint8 per-channel symmetric weights;
      - quint8 per-tensor affine activations with min/max calibration and full 8-bit range;
      - int32 biases;
      - calibrated on 1,000 attacker-holdout images chosen by the run seed.
  - **Simulation runs on integer codes:** convolution sums are exact in float64, then requantized, so results do not depend on batch size.
    - My first float32 version did depend on batch size; a test caught that before any run.
    - Tests check it against an independent float64 dequantized reference (same codes).
    - They also check it against `torch.ao`'s converted INT8 model, with 7-bit activations so the kernels cannot saturate: identical qparams, logits within 6 steps, same top-1. That comparison is on a seeded model only.
  - **Harness extension:** `AttackOutput.runtime_model` (optional). Accuracy and the behavioral test are scored on it, while weight extraction still reads the `state_dict`. `apply` mode refuses such attacks. The `none` control re-run after the change (scratch dir, not committed) still reproduced P3.6 exactly.
  - **Run** (CPU, clean tree at `ab853c5`), test accuracy vs the dual `W*` (90.85%):
    - Fused FP32: 90.85%, 0 discordant images. Fired 100/100. Weight correlation +0.943, z 10.67, 128/128 bits.
    - FP16: 90.85% (1 image lost, 1 gained). Fired 100/100. Weight correlation +0.9091, z 10.29, the same to 6 decimals.
    - INT8: 90.77%, drop +0.08 pp, CI [-0.08, +0.24], p = 0.40. Fired 100/100, mean target probability 0.9994. Weight correlation +0.943, z 10.67, 128/128 bits.
  - **Reading:** neither quantization removes either watermark or costs measurable accuracy. Both watermarks are detected at 1e-6 in every row.
    - Fusion moves the carrier by 153% of its L2 norm (BN scales -0.09 to 5.2), yet the blind correlation rose from 0.909 to 0.943 and INT8 rounding then barely changed it. Why it rose is not investigated. It is one model; a model with different BN statistics could go the other way.
  - Limits: one model, one key, one calibration draw; INT8 simulated rather than run on a real backend; FP16 activations not rounded.
  - 27 new tests, 626 in total, all pass.
- [x] **P4.5** `[GPU]` Fine-tuning on a held-out data split, sweeping epochs and learning rate. Include an aggressive high-learning-rate run, since that is the realistic removal attack.
  - **Owner decision:** grid LR 0.001 / 0.01 / 0.05 / 0.1 × epochs 5 / 20 / 60, fixed before the run. 0.1 is the aggressive setting, the peak LR `W*` was trained with.
  - **Attack** (`src/attacks/finetune.py`, `finetune_holdout`):
    - Starts from the dual `W*` and uses only the 5,000-image attacker holdout.
    - P0.5 recipe: SGD Nesterov 0.9, wd 5e-4, batch 128, augmentation, 1-epoch warmup, cosine decay to 0.
    - No key, no triggers, no test set; monitored on the holdout; final-epoch weights.
  - **Run:** 12 Colab `apply` runs (T4, commit `4897934`, clean tree), each in one session with no resume. 3.1–3.3 s per epoch, 1,087 s of fine-tuning in total. The handoff estimated 20–35 min; the actual was 18 min.
  - **Verified before scoring:**
    - every record's config equals the committed config;
    - source hash and fine-tuning start-state digest equal the dual `W*`;
    - not a smoke run, key unused;
    - 5,000 images and 40 steps per epoch, epoch indices consistent;
    - weights hash matches, strict load, finite values, weights changed.
  - The files arrived in `results/attacks/` and were moved to `results/attacks/p4.5_apply/`. The records were committed, then all 12 were scored locally with `evaluate` from a clean tree at `260b64a`.
  - **Test accuracy** (drop vs the dual `W*`, 90.85%; order is epochs 5 / 20 / 60):
    - LR 0.001: 90.48 / 90.49 / 90.09%.
    - LR 0.01: 89.02 / 89.05 / 87.99%.
    - LR 0.05: 85.69 / 86.06 / 85.30%.
    - LR 0.1: 81.86 / 83.74 / 83.36%.
  - **Fired /100** (P2.8 detected at 1e-6 needs ≥ 29):
    - LR 0.001: 100 / 100 / 100.
    - LR 0.01: 62 / 46 / 30.
    - LR 0.05: 11 / 8 / 6.
    - LR 0.1: 9 / 8 / 4.
  - **Weight z** (all detected at 1e-6):
    - LR 0.001: 10.28 / 10.27 / 10.26.
    - LR 0.01: 10.27 / 10.22 / 10.07.
    - LR 0.05: 10.12 / 9.66 / 8.26.
    - LR 0.1: 9.89 / 9.16 / 6.49.
  - **Reading:**
    - **Behavioral watermark removed.** Every LR 0.05 and 0.1 run removes it, from 5 epochs on. Fired counts of 4–11 are what a model independent of `K` gives. The cost is 4.8 to 9.0 pp of test accuracy; the cheapest removal is LR 0.05 at 20 epochs (86.06%, drop +4.79 pp). LR 0.01 wears it down (62 → 46 → 30) but it is still detected; 30 is one above the threshold.
    - **Weight watermark detected in all 12.** z falls with LR and length, lowest 6.49 (p ≤ 7.4e-10) at LR 0.1, 60 epochs, just above z\*(1e-9) = 6.438. Whether longer or harder fine-tuning would remove it is not measured.
    - **Memorisation, not generalisation.** The owner noted holdout accuracy near 99.9%. Holdout accuracy is on the attacker's own training images, so it is a fit figure. The holdout-minus-test gap grows with LR and length, to +16.3 pp (LR 0.1, 60 epochs: 99.64% vs 83.36%). Test loss rises with length at LR ≥ 0.01, and test accuracy does not improve beyond 20 epochs. The longer runs overfit 5,000 images rather than recover accuracy.
  - Limits: one run per cell, one seed, one source model; the attacker has 5,000 images; GPU training is not bit-reproducible; plain fine-tuning only.
  - 27 tests added at handoff, 653 in total, all pass.
- [ ] **P4.6** `[GPU]` Combined attack: prune then fine-tune. This is the strongest realistic threat and the most interesting result.
- [ ] **P4.7** `[GPU]` Knowledge distillation to a student model. Test whether either watermark transfers. **Expect the behavioral watermark to largely NOT survive distillation.** That is a real finding, not a failure. Report it prominently and honestly.
- [ ] **P4.8** `[GPU]` Overwrite attack: an adversary embeds their own watermark with their own key. Does ours still extract?
- [ ] **P4.9** Produce the master robustness table and heatmap figure across all attacks.
- [ ] **P4.10** Write `results/ATTACK_FINDINGS.md`: what survived, what did not, and which watermark is stronger under which attack. Blunt and quantitative.

## Phase 5: Cryptographic identity

- [ ] **P5.1** Implement SHA-256 model fingerprinting over a canonical serialization of the weights. Verify it is stable across save and reload.
- [ ] **P5.2** Integrate a Python Poseidon implementation. Validate it against known test vectors before trusting it.
- [ ] **P5.3** Implement the commitment `C = Poseidon(K, S, nonce)`. Document the exact field layout, because the Circom circuit in P7 must match it bit for bit.
- [ ] **P5.4** Write the commitment publication artifact: `provenance/commitment.json` holding `C`, the model fingerprint, the owner identity, and a timestamp.
- [ ] **P5.5** Timestamp it for real: GPG signed git tag, plus OpenTimestamps if it cooperates. This is what makes "I committed before the dispute" meaningful.
- [ ] **P5.6** Write the verifier that checks a revealed secret against a published commitment (the non-ZK baseline, so the ZK version has something to be compared against).

## Phase 6: Provenance record

- [ ] **P6.1** Define the provenance record schema: owner, model fingerprint, watermark commitment, trigger set commitment, timestamp, signature.
- [ ] **P6.2** Sign the record with a real keypair via the `cryptography` library.
- [ ] **P6.3** Write the provenance verifier: signature valid, fingerprint matches, commitment well formed.
- [ ] **P6.4** Simulate the full theft timeline end to end: publish commitment, hand model to "attacker", attacker modifies it, we audit. Script it as `experiments/theft_simulation.py`.

## Phase 7: Zero-knowledge, Track A (the real ZK statement)

- [ ] **P7.1** Learn and document the four primitives in `docs/ZK_NOTES.md`, in your own words: witness, circuit, constraint system, Groth16 setup. No copy-paste.
- [ ] **P7.2** Install the toolchain in Colab: `circom`, `snarkjs`, `circomlib`. Record exact versions.
- [ ] **P7.3** Toy circuit first: prove knowledge of a Poseidon preimage. Full loop, setup through verification. Do not skip this step.
- [ ] **P7.4** Download an appropriately sized Hermez `.ptau` file. Use the smallest power of tau that fits the circuit.
- [ ] **P7.5** Write the real circuit: private inputs `K`, `S`, nonce; public input `C`; constraint `Poseidon(K, S, nonce) == C`. Must match P5.3 exactly.
- [ ] **P7.6** Generate the proving and verification keys. Record key sizes and peak RAM.
- [ ] **P7.7** Generate a proof and verify it. Record proof size, prove time, verify time. **Measure, do not quote marketing numbers.**
- [ ] **P7.8** Negative tests: a proof with the wrong `K` must fail; a proof against the wrong `C` must fail. A ZK proof you have never seen fail is a ZK proof you have not tested.
- [ ] **P7.9** Extend the circuit so the statement also binds the trigger derivation (prove the triggers used in the audit really come from the committed `K`). This closes the loop between the cryptography and the watermark. If it proves too expensive, mark `[!]` and document the constraint count that killed it.
- [ ] **P7.10** Write `docs/ZK_STATEMENT.md` stating precisely what is proved and, equally important, what is **not** proved.

## Phase 8: Zero-knowledge, Track B (zkML with EZKL)

- [ ] **P8.1** Export `zk_model` to ONNX. Verify ONNX Runtime output matches PyTorch output.
- [ ] **P8.2** Install `ezkl` via pip, pin the version, and run the official example notebook unchanged to confirm the environment works before touching our model.
- [ ] **P8.3** Run the EZKL pipeline on `zk_model`: gen-settings, calibrate-settings, compile, setup. Record peak RAM at each stage.
- [ ] **P8.4** Generate a witness and a proof for a single trigger input. Record prove time, proof size, and key sizes.
- [ ] **P8.5** Verify the proof locally, off chain. Record verify time.
- [ ] **P8.6** Check quantization fidelity: EZKL quantizes when converting ONNX to a circuit, so the circuit output can diverge from PyTorch. Measure how often the trigger still produces the watermark response inside the circuit. **This is a real risk to the demo and must be measured, not assumed.**
- [ ] **P8.7** Frame the statement correctly: "this committed model produces this response on an input I am not revealing". Document what this does and does not demonstrate.
- [ ] **P8.8** If `zk_model` blows the RAM budget, shrink it and re-run rather than escalating hardware. Record what size was actually provable on Colab free. That measured ceiling is itself a good portfolio result.

## Phase 9: IP Auditor and dashboard

- [ ] **P9.1** Build the auditor engine in `src/auditor/`: input a suspect model plus a provenance record, output a structured verdict object.
- [ ] **P9.2** Wire in all five checks: model fingerprint, behavioral WDR with p-value, weight correlation with p-value, commitment validity, ZK proof validity.
- [ ] **P9.3** Implement graded verdicts driven by the measured thresholds from P2.8 and P3.7. No hardcoded verdicts. Evidence strength must be a function of statistics.
- [ ] **P9.4** Test the auditor against three model classes: our `W*`, our attacked variants, and genuinely unrelated third party models. The unrelated-model test proves the auditor is not a rubber stamp, so it is mandatory.
- [ ] **P9.5** Build the Streamlit dashboard: upload a suspect model, watch the checks run, see the forensic report.
- [ ] **P9.6** Show "private data revealed: 0" only where it is literally true, with a tooltip explaining precisely what stayed private.
- [ ] **P9.7** Deploy free on Streamlit Community Cloud with a bundled demo model so a reviewer can click through without setup.
- [ ] **P9.8** Record a short demo GIF for the README.

## Phase 10: Portfolio polish

- [ ] **P10.1** README: problem, architecture diagram, measured headline results, honest limitations, quickstart.
- [ ] **P10.2** Results section with the real numbers only, pulled from `results/`.
- [ ] **P10.3** A `LIMITATIONS.md` that a skeptical reviewer would respect. This section is what separates this project from a buzzword repo.
- [ ] **P10.4** Colab badge notebooks so any part can be reproduced in one click.
- [ ] **P10.5** Write-up explaining the two-track ZK design decision (Circom for the commitment relation, EZKL for inference) and why it was the right call.

---

# 7. CLAIM DISCIPLINE

## 7.1 Never write these

- "Unremovable" or "immune to pruning" watermark.
- "Court proof" or "legally proves ownership".
- Any accuracy, WDR, proof size, or timing figure not produced by a script in this repo.
- "Sub-second verification" copied from a vendor blog post.
- "Zero-knowledge" applied to the watermark itself. The watermark is not zero-knowledge. Only the proof relation is.

## 7.2 Write these instead

- "Behavioral WDR degrades from X percent to Y percent at Z percent sparsity, measured over N triggers."
- "The weight watermark survived quantization with correlation X, p < Y."
- "Distillation removed the behavioral watermark, reducing WDR from X to Y. This is a genuine limitation of trigger based watermarking."
- "This is a technical ownership verification demonstration, not legal evidence."

A negative result, measured and clearly reported, is worth more in a portfolio than a positive result that a reviewer can poke a hole in within thirty seconds.

---

# 8. RESULTS LEDGER

Fill in as tasks complete. `TBD` until measured.

## 8.1 Environment actually used

| Item | Value |
|---|---|
| Accelerator assigned | Tesla T4 (P0.5 run) |
| System RAM available | 12.67 GiB total (P0.7 run) |
| torch version | 2.11.0+cu128 |
| ezkl version (pinned) | TBD |
| circom / snarkjs / circomlib versions | TBD |

The values above come from the `environment` block of
`results/p0.5_clean_baseline__seed1337__20260913T071152+0000.json`. That block
also records Python 3.13.15, CUDA 12.8, numpy 2.1.3, and
`Linux-6.6.122+-x86_64-with-glibc2.39`. It does not record system RAM. The RAM
row is `MemTotal` from the P0.7 result
(`results/p0.7_zk_model__seed1337__20260913T075832+0000.json`,
`environment.system_ram_gb`). That run was on the same T4, torch, CUDA and
Python versions. The RAM figure is total memory, not free memory. The
accelerator is what those sessions got; section 2.1 says a T4 is not
guaranteed.

## 8.2 Models

| Model | Dataset | Params | Clean accuracy |
|---|---|---|---|
| `main_model` clean `W` | CIFAR-10 | 307,946 | 91.20% |
| `main_model` watermarked `W*` (behavioral only) | CIFAR-10 | 307,946 | 90.73% |
| `main_model` dual-watermarked `W*` (P3.6, final) | CIFAR-10 | 307,946 | 90.85% (alpha chosen after seeing test drops; optimistic) |
| `zk_model` | MNIST | 6,138 | 98.96% |

`main_model` parameter count from `experiments/p0_4_model_summary.py` (P0.4).
307,040 of the 307,946 (99.7%) are conv/linear weights, which is the pool the
weight watermark spreads into (P3.2). The `width=16` distillation student for
P4.7 is 82,554.

Clean `W` accuracy from `experiments/p0_5_train_clean.py` (P0.5), result file
`results/p0.5_clean_baseline__seed1337__20260913T071152+0000.json`: top-1
91.20% on the official 10,000-image CIFAR-10 test set (test loss 0.3087), after
the 60th of 60 epochs, seed 1337, Tesla T4, commit `55f2607`. These are the
final-epoch weights, not a test-selected checkpoint; final and best epoch happen
to coincide (epoch index 59). Accuracy on the 5,000-image attacker holdout is
90.88%. Trained on 45,000 images. Weights SHA-256
`54f112f4d7edc2ddacc181f8f4db25da27874e7e65aaf9407778a3ce5122fdcf`, checked
against the local file, which loads strictly into `MainModel(width=32)`.

`zk_model` numbers from `experiments/p0_7_train_zk_model.py` (P0.7), result
file `results/p0.7_zk_model__seed1337__20260913T075832+0000.json`: 6,138
parameters (budget 10,000), 2,608 ReLU output elements per input, widths
8/16/16. Top-1 accuracy is 98.96% on the official 10,000-image MNIST test set
(test loss 0.0336) after the 20th of 20 epochs, seed 1337, Tesla T4, commit
`ca412fb`. The best epoch was index 17 at 99.02%. The reported figure and the
saved weights are from the final epoch, confirmed by re-evaluating the local
weights file on CPU (9,896/10,000). Trained on all 60,000 training images in
276.65 s of epoch time. Weights SHA-256
`6bc298d04edfe5af136349904c53ccc9a82e80dfb03729f616e552a3df9ddec4`.

Behavioral-watermark `W*` numbers from `experiments/p2_3_train_watermarked.py`
(P2.3), result file
`results/p2.3_behavioral_wm__seed1337__20260913T112054+0000.json`. Top-1
accuracy is 90.73% on the official 10,000-image CIFAR-10 test set (test loss
0.3089) after the 60th of 60 epochs, seed 1337, Tesla T4, commit `0aaea8e`.
Recipe, split and seed are the same as P0.5, plus 4 un-augmented triggers
appended to every batch of 128 (100 triggers, 1,408 trigger samples per epoch).
The best epoch was index 58 at 90.82%. The reported figure and the saved
weights are from the final epoch, confirmed by re-evaluating the local weights
file on CPU (9,073/10,000, loss 0.308929). Accuracy on the 5,000-image attacker
holdout is 91.14% (4,557/5,000, reproduced on CPU). 1,177.13 s of epoch time.
Weights SHA-256
`be00f2b556979457b045b9a4925dd3e547ee586f984b6969b9266e61a7197222`. This `W*`
has the behavioral watermark only; P3.6 produces the dual-watermarked model.
The accuracy drop against P0.6 is left to P2.5.

Dual-watermarked `W*` numbers from `experiments/p3_6_make_dual_model.py`
(P3.6), result file `results/p3.6_dual_wm__seed1337__20260914T085541+0000.json`,
CPU, seed 1337, commit `82c0241`, clean tree. This is the P2.3 model plus the
post-hoc weight watermark `alpha * P_K^T * S` at alpha 0.1, with owner id
`blacktensor-zkcrown-owner` and no BN recalibration. Weights SHA-256
`7a9a9f141b55c7894b208c1f338f4c7385362f969b82b319ce5893fb4bb434c4`, gitignored.
All figures are measured on the weights reloaded from that file.

- **Test:** 90.85% (9,085/10,000, loss 0.3078).
- **Holdout:** 91.16% (4,558/5,000).
- **Paired test drop against the P2.3 model:** -0.12 pp, 95% CI [-0.32,
  +0.08], McNemar p = 0.27 (44 images only the P2.3 model got right, 56 only
  the dual model).
- **Paired test drop against clean `W`:** +0.35 pp, CI [-0.16, +0.86],
  p = 0.19.
- **Holdout drops:** -0.02 pp (p = 1.0) against the P2.3 model and -0.28 pp
  (p = 0.50) against `W`.

The counts equal P3.5's alpha 0.1 row. Alpha was chosen from P3.5's grid after
seeing its test-set drops, so 90.85% is an optimistic estimate of this model's
accuracy. This is the model Phase 4 attacks.

## 8.3 Watermark baseline

| Metric | Value |
|---|---|
| Trigger set size N | 100 (P2.3 bundle `fbd65ec7…22baec8`) |
| Behavioral WDR | 100% (100/100), `W*` from P2.3; 100% (100/100) on the P3.6 dual `W*`, p = 3.8e-96 |
| Weight watermark on the P3.6 dual `W*` (alpha 0.1) | correlation +0.909, z = 10.29, 126/128 bits (threshold TBD, P3.7) |
| Behavioral FPR (input level, 1,000 each) | random 12.6% (chance 12.7%), clean-unrelated 10.0% (chance 10.0%) |
| Accuracy drop from watermarking | 0.47 pp on test (91.20% → 90.73%), 95% CI [-0.04, +0.98], McNemar p = 0.077; single run each |
| Behavioral detection p-value (P2.8) | `W*` 100/100: p = 3.8e-96; clean `W` 3/100: p = 0.999 |
| Behavioral detection thresholds, N = 100 (P2.8) | k* = 17 / 20 / 23 / 29 / 35 at alpha 0.05 / 0.01 / 1e-3 / 1e-6 / 1e-9 |
| Weight extraction correlation, correct key (P3.4, post-hoc on `W*`, alpha not chosen) | +0.123 / +0.313 / +0.473 / +0.756 / +0.909 at alpha 0 / 0.01 / 0.02 / 0.05 / 0.1 |
| Weight extraction correlation, wrong key (P3.4, 100 keys, `W*` host) | mean +0.001, sd 0.095, max \|z\| 2.71 |
| Weight detection threshold and its FPR (P3.7) | z\* = sqrt(2 ln(1/alpha)), FPR ≤ alpha for any model independent of `K` (proof): z\* = 2.448 / 3.035 / 3.717 / 5.257 / 6.438 at alpha 0.05 / 0.01 / 1e-3 / 1e-6 / 1e-9; p-value floor 1.6e-28 |
| Weight detection p-value, dual `W*` (P3.7) | z = 10.29, p ≤ 1.1e-23; owner `K` on unwatermarked models: z = 0.18 (p ≤ 0.98) and 1.39 (p ≤ 0.38) |

Trigger perturbation size, from `experiments/p1_3_visualize_triggers.py`
(P1.3), result file
`results/p1.3_trigger_visualization__seed1337__20260913T091124+0000.json`. The
set is 100 triggers built with the public demo key on the CIFAR-10 training
split. At the default amplitude A = 16/255, mean PSNR against the base image
is 24.17 dB (range 24.05 to 25.13), the L-infinity norm is 16 levels, the mean
L2 norm is 3.430 on the [0, 1] scale, and clipping shortened 3.58% of pixel
channels. At the other sweep amplitudes, mean PSNR is 36.14 dB at A = 4,
30.15 dB at 8, 18.25 dB at 32 and 12.51 dB at 64. These are image statistics
only. No model was run. A real-`K` set has different bases and signs, but the
same A gives nearly the same figures, because PSNR for unclipped ±A noise is
fixed at 20·log10(255/A).

Behavioral WDR from `experiments/p2_4_measure_wdr.py` (P2.4), result file
`results/p2.4_wdr__seed1337__20260913T113841+0000.json`, CPU, seed 1337. The
N = 100 triggers and targets were regenerated from `K` at run time, and their
bundle digest matched the P2.3 training bundle `fbd65ec7…22baec8`. `W*`
(SHA-256 `be00f2b5…197222`) gave its keyed target on 100 of 100 triggers, so
WDR is 100%. Mean softmax probability of the target was 0.9995, median 0.9997,
minimum 0.9965. These are the training triggers, so the
figure measures whether `W*` kept the responses it was trained on. It is not a
false-positive rate (P2.6) or a p-value (P2.8).

Controls from the same run, counts out of 100, all eval-mode top-1:

| Model | Inputs | Fired (target) | Base label | Other class |
|---|---|---|---|---|
| `W*` | triggers | 100 | 0 | 0 |
| `W*` | unperturbed base images | 1 | 98 | 1 |
| clean `W` (P0.5) | triggers | 3 | 60 | 37 |
| clean `W` (P0.5) | unperturbed base images | 1 | 97 | 2 |

`W*` gives the target on the perturbed images but not on their bases, so it is
responding to the key-derived perturbation rather than the image. Clean `W`
was trained without `K` and fired on 3 triggers. The ±16 perturbation alone
cut its accuracy on these 100 images from 97 to 60.

Accuracy drop from `experiments/p2_5_accuracy_drop.py` (P2.5), result file
`results/p2.5_accuracy_drop__seed1337__20260913T114811+0000.json`, CPU, seed
1337. Both weights files were hash-checked, and both reproduced the accuracies
their Colab runs recorded, on test and holdout. The drop is
acc(`W`) - acc(`W*`), so positive means `W*` is worse.

| Split | `W` | `W*` | Drop | 95% CI | `W` only right | `W*` only right | McNemar p (exact) |
|---|---|---|---|---|---|---|---|
| test, 10,000 (primary, P0.6) | 91.20% | 90.73% | +0.47 pp | [-0.04, +0.98] pp | 362 | 315 | 0.077 |
| attacker holdout, 5,000 | 90.88% | 91.14% | -0.26 pp | [-1.01, +0.49] pp | 178 | 191 | 0.53 |

The CI is a normal approximation on per-image paired differences. Both it and
the McNemar test treat the evaluation images as the random sample. They do not
cover variation between training seeds, which is unmeasured because each model
is one run. Read this as: the watermark's accuracy cost, measured once, is 0.47
pp on the test set. That is not distinguishable from zero at the 5% level on
these images, and the holdout gives the opposite sign.

False positive rate from `experiments/p2_6_false_positive_rate.py` (P2.6),
result file
`results/p2.6_false_positive_rate__seed1337__20260913T120034+0000.json`, CPU,
seed 1337. The owner targets were regenerated from `K` (bundle digest matched
P2.3), and both weights files were hash-checked. Each set has 1,000 inputs.
Non-trigger input `j` is scored against the owner's target for trigger slot
`j mod 100`; it fires if top-1 equals that target. "Chance" is the expected
fire count over random slot assignments given the model's own predictions,
`sum_j q(pred_j)`.

| Model | Set | Fired | FPR | Chance fired | Accuracy | Fires contradicting label (chance) |
|---|---|---|---|---|---|---|
| `W*` | uniform random noise | 126 | 12.6% | 127.3 | n/a | n/a |
| `W*` | clean unrelated (CIFAR-10 test) | 100 | 10.0% | 100.2 | 90.5% | 18 (10.1) |
| `W*` | noise decoys (test + ±16, not `K`) | 119 | 11.9% | 108.7 | 41.0% | 71 (65.7) |
| clean `W` | uniform random noise | 88 | 8.8% | 89.0 | n/a | n/a |
| clean `W` | clean unrelated | 91 | 9.1% | 100.3 | 92.0% | 7 (8.4) |
| clean `W` | noise decoys | 103 | 10.3% | 104.8 | 47.5% | 55 (56.2) |

How to read it. With per-trigger targets spread over 10 classes, any model
fires on roughly 1 in 10 non-trigger inputs by chance. That is what both models
show, against 100% for `W*` on its own triggers (P2.4). Input-level FPR
therefore has a floor around 10% by construction. What separates a watermarked
model from an unwatermarked one is how many of the N owner triggers fire,
which P2.8 turns into a p-value.

The chance column depends on the model: it measures how much a model's
predictions land on the owner's target classes. On uniform noise it is higher
for `W*` (12.73%) than for `W` (8.90%). That is one pair of numbers, with no
test and no mechanism established. Fired minus chance is noise from the single
fixed pairing, not a leakage signal, because slot `j mod 100` is unrelated to
image content; that includes `W*`'s 18 vs 10.1 on clean images. The noise
decoys go beyond the two sets the task names: `K`-independent ±16 noise lowers
accuracy on these test images to 41.0% for `W*` and 47.5% for `W`.
Input-set digests are in the result file. Owner targets and per-input
predictions are not.

Trigger-ratio sweep from `experiments/p2_7_analyze_sweep.py` (P2.7), result
file `results/p2.7_ratio_sweep__seed1337__20260914T070218+0000.json`, CPU,
seed 1337. It covers five new training runs from
`experiments/p2_7_ratio_sweep.py` (Colab T4, commit `4c1e3da`, clean tree, 60/60
epochs each). Result files: `results/p2.7_r0001__…T124406`, `r0005__…T130556`,
`r0020__…T132720`, `r0078__…T134858` and `r1252__…T141053`, all
`+0000.json`. Also included: the reused P2.3 `W*` and P0.5 `W`. Each model uses
the P0.5 recipe, seed, split and clean batch order. Only the trigger schedule
changes: `tpb` triggers added to every `k`-th batch. Ratio is trigger samples
per epoch divided by the 45,000 clean training images.

WDR uses P2.4's method, on the 100 triggers regenerated from `K` (digest
matched P2.3). The drop uses P2.5's paired comparison against clean `W` on the
10,000-image test set. Every weights file matched its recorded SHA-256, and
every CPU test accuracy equals the accuracy its run recorded.

| Run | tpb / every k | Trigger samples per epoch | Ratio | WDR | Mean target prob | Test acc | Drop | 95% CI | McNemar p | Trigger acc stable at 100% from epoch |
|---|---|---|---|---|---|---|---|---|---|---|
| P0.5 clean `W` | 0 | 0 | 0 | 3/100 | 0.039 | 91.20% | 0 | n/a | n/a | n/a |
| `p2.7_r0001` | 1 / 64 | 6 | 0.013% | 9/100 | 0.091 | 91.01% | +0.19 pp | [-0.31, +0.69] | 0.48 | never |
| `p2.7_r0005` | 1 / 16 | 22 | 0.049% | 19/100 | 0.187 | 90.74% | +0.46 pp | [-0.04, +0.96] | 0.077 | never |
| `p2.7_r0020` | 1 / 4 | 88 | 0.196% | 64/100 | 0.377 | 90.95% | +0.25 pp | [-0.26, +0.76] | 0.35 | never |
| `p2.7_r0078` | 1 / 1 | 352 | 0.782% | 100/100 | 0.991 | 90.26% | +0.94 pp | [+0.42, +1.46] | 0.0005 | 50 |
| P2.3 `W*` | 4 / 1 | 1,408 | 3.13% | 100/100 | 0.9995 | 90.73% | +0.47 pp | [-0.04, +0.98] | 0.077 | 43 |
| `p2.7_r1252` | 16 / 1 | 5,632 | 12.5% | 100/100 | 0.9998 | 90.86% | +0.34 pp | [-0.18, +0.86] | 0.22 | 40 |

Minimum target probability at the three 100% points: 0.960 (0.782%), 0.996
(3.13%), 0.999 (12.5%). Holdout accuracies, as recorded by the Colab runs and
not re-scored on CPU, are 91.36%, 91.22%, 90.58%, 91.14% and 90.80% for the
five sweep runs in ratio order. Epoch times ranged from 1,270 s to 1,300 s.

How to read it. WDR climbs steeply between 0.05% and 0.8% trigger samples per
clean sample, and it is saturated at 100/100 from 0.782% up. Beyond that point
more triggers raised the target probability, from 0.991 to 0.9998 mean, and
the 100% training trigger accuracy stabilised earlier. The accuracy drop shows
no trend with the ratio. The largest drop, and the only one distinguishable
from zero on these test images, is at 0.782%. The 12.5% run, with 16 times as
many trigger samples, lost only 0.34 pp. Each point is a single training run,
and the intervals cover test-set sampling only, so the drop differences between
ratios cannot be separated from seed-to-seed variation. That variation is not
measured, and no claim is made about it either way. WDR here is on the trained triggers;
whether 9/100 or 19/100 is evidence of a watermark is a P2.8 question. No
ratio was re-selected: P2.3's `W*` remains the behavioral model.

Detection test from `experiments/p2_8_detection_test.py` (P2.8), result file
`results/p2.8_detection_test__seed1337__20260914T073942+0000.json`, CPU, seed
1337, commit `4c90006`, clean tree. The test is defined in `src/watermark/significance.py`. H0: the suspect
model is independent of the owner's keyed targets. The p-value is the exact
`P(Binomial(N, 1/9) >= k)`. Under H0, each trigger fires independently with
probability at most 1/9 (P2.2), so this p-value is valid for any model
independent of the targets. That validity is a proof, not a measurement. It is
conservative for models that often predict the base label.

Thresholds for N = 100. `k*` is the smallest fired count whose p-value is at
most alpha. "Exact bound" is the largest false-positive probability any H0
model can have when the test rejects at `k >= k*`.

| alpha | k* | Exact bound |
|---|---|---|
| 0.05 | 17 | 0.0493 |
| 0.01 | 20 | 0.00655 |
| 1e-3 | 23 | 5.29e-4 |
| 1e-6 | 29 | 8.67e-7 |
| 1e-9 | 35 | 2.66e-10 |

p-values for the models measured so far, all on the N = 100 owner triggers.
`W*` and `W` were re-scored here from `K` (bundle digest matched P2.3, weights
hash-checked), and the counts reproduced P2.4. The P2.7 counts come from its
committed result file.

| Model | Fired | p-value | Rejects at 1e-6 |
|---|---|---|---|
| clean `W` (P0.5) | 3/100 | 0.999 | no |
| P2.7 0.013% | 9/100 | 0.793 | no |
| P2.7 0.049% | 19/100 | 0.0136 | no (rejects at 0.05 only) |
| P2.7 0.196% | 64/100 | 2.59e-36 | yes |
| P2.7 0.782% | 100/100 | 3.76e-96 | yes |
| `W*` (P2.3, 3.13%) | 100/100 | 3.76e-96 | yes |
| P2.7 12.5% | 100/100 | 3.76e-96 | yes |

Empirical null check, same run. 1,000 public wrong keys
`SHA-256("zk-crown/p2.8/null-key/v1" || u64 1337 || u64 j)` each build their
own 100 triggers and targets on the same split. `W` and `W*` are independent
of every wrong key, so each count is an H0 draw.

| Model | Mean fired (bound 11.11) | Var (bound 9.88) | Max | Mean base-label hits | Rejected at 0.05 (bound 49.3) | Rejected at 0.01 (bound 6.6) | Observed vs expected total fired | z |
|---|---|---|---|---|---|---|---|---|
| clean `W` | 5.70 | 5.42 | 14 | 48.7 | 0 / 1,000 | 0 / 1,000 | 5,701 vs 5,698.0 | +0.04 |
| `W*` | 6.38 | 6.22 | 16 | 41.7 | 0 / 1,000 | 0 / 1,000 | 6,379 vs 6,482.2 | -1.36 |

"Expected total" is `sum_j (100 - base_label_hits_j) / 9`, the exact
expectation given each model's own predictions. Agreement to within 1.4 sd
says the wrong-key targets behaved as independent of the predictions, as P2.2
requires. Both models sit well below the bound, because they predict the base
label on 42 to 49 of 100 triggers, so on these models the test has a lower
real false-positive rate than its stated bound. With 1,000 keys the
rejection rate is only checked near alpha 0.05 and 0.01. Smaller alphas rest
on the proof and on the exact tests in `tests/test_significance.py`.

Reading the p-values. The P2.7 0.049% run fired on 19 of 100 triggers
(p = 0.014). That rejects H0 at 0.05 but not at 0.01 or any stricter level.
The p-values assume `K` and the trigger set were fixed before the suspect was
seen, one pre-declared test, one query per trigger, and a correction when
several suspects are audited. The 100/100 figures are on the training triggers
of unattacked models; survival under attack is Phase 4.

Weight watermark key specificity from `experiments/p3_4_key_specificity.py`
(P3.4), result file
`results/p3.4_key_specificity__seed1337__20260914T082116+0000.json`, CPU, seed
1337, commit `2a2eee9`, clean tree. `S` was derived from `K` for owner id
`blacktensor-zkcrown-owner`. Each host was watermarked post-hoc (P3.2) with
`K` at each alpha, then extracted blind (P3.3). The 100 wrong keys are
`SHA-256("zk-crown/p3.4/wrong-key/v1" || u64 1337 || u64 j)`, each deriving its
own `P_K'` and `S'`. z = correlation × sqrt(128), used as a scale only, not a
p-value. The alpha grid was fixed in advance and is not a selection; the
accuracy of these models is not measured (P3.5).

| Host | alpha | Correct corr (z) | Bits /128 | Amplitude | Wrong mean | Wrong sd | Wrong range | Wrong max \|z\| | Gap | Gap / wrong sd | Above all 100 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `W*` | 0 | +0.123 (1.39) | 68 | 0.0060 | +0.001 | 0.095 | [-0.180, +0.240] | 2.71 | +0.122 | 1.28 | no |
| `W*` | 0.005 | +0.221 (2.50) | 75 | 0.0110 | +0.001 | 0.095 | [-0.180, +0.239] | 2.71 | +0.220 | 2.32 | no |
| `W*` | 0.01 | +0.313 (3.55) | 80 | 0.0160 | +0.001 | 0.095 | [-0.180, +0.239] | 2.71 | +0.313 | 3.29 | yes |
| `W*` | 0.02 | +0.473 (5.35) | 94 | 0.0260 | +0.001 | 0.095 | [-0.180, +0.239] | 2.71 | +0.472 | 4.97 | yes |
| `W*` | 0.05 | +0.756 (8.55) | 110 | 0.0560 | +0.001 | 0.095 | [-0.181, +0.239] | 2.71 | +0.755 | 7.98 | yes |
| `W*` | 0.1 | +0.909 (10.29) | 126 | 0.1061 | +0.001 | 0.094 | [-0.182, +0.239] | 2.70 | +0.908 | 9.64 | yes |
| clean `W` | 0 | +0.016 (0.18) | 62 | 0.0006 | +0.014 | 0.088 | [-0.219, +0.266] | 3.01 | +0.002 | 0.02 | no |
| clean `W` | 0.005 | +0.138 (1.56) | 68 | 0.0056 | +0.014 | 0.088 | [-0.219, +0.266] | 3.01 | +0.124 | 1.41 | no |
| clean `W` | 0.01 | +0.253 (2.87) | 70 | 0.0107 | +0.014 | 0.088 | [-0.219, +0.266] | 3.01 | +0.240 | 2.73 | no |
| clean `W` | 0.02 | +0.453 (5.12) | 88 | 0.0207 | +0.014 | 0.088 | [-0.220, +0.265] | 3.00 | +0.439 | 5.00 | yes |
| clean `W` | 0.05 | +0.779 (8.82) | 119 | 0.0507 | +0.014 | 0.088 | [-0.221, +0.264] | 2.99 | +0.766 | 8.75 | yes |
| clean `W` | 0.1 | +0.927 (10.48) | 127 | 0.1008 | +0.014 | 0.087 | [-0.222, +0.262] | 2.96 | +0.913 | 10.48 | yes |

With the wrong projection scored against the true `S`, max |z| was 2.55 to
2.60 on `W*` and 2.28 to 2.30 on `W`. The size of the embedded change, from
the same run: RMS per carrier parameter is 0.00010, 0.00020, 0.00041, 0.00102
and 0.00204 at the five non-zero alphas, which is 0.23%, 0.46%, 0.91%, 2.28%
and 4.56% of the carrier L2 norm of `W*`. The largest float32 rounding error
was 2.8e-8.

How to read it. A wrong key gives a correlation that looks like noise: its
spread does not change with alpha, and it is about the 1/sqrt(128) = 0.088 that
independent projections would give. That holds whether the wrong key derives
its own `S'` or is scored against the true `S`. The correct key's correlation
grows with alpha. It clears all 100 wrong keys from alpha 0.02 on both hosts,
and from 0.01 on `W*`. The 0.01 case on `W*` is helped by that host's chance
correlation of +0.12 on `K` at alpha 0, which is noise, not a watermark. The
host term's projected RMS, 0.049 for `W*` and 0.041 for `W`, sets the scale
alpha must beat. The normalised correlation cannot exceed 1, so z is capped at
sqrt(128) = 11.3, and at alpha 0.1 it is already about 10.4. Any p-value built
on this statistic will have a floor, and P3.7 has to state it. Separation
from 100 keys is not a false positive rate; that is P3.7's 1,000-key null.

Weight watermark strength sweep from `experiments/p3_5_alpha_sweep.py` (P3.5),
result file `results/p3.5_alpha_sweep__seed1337__20260914T084524+0000.json`,
CPU, seed 1337, commit `e943c19`, clean tree.
- **Host:** `W*` (P2.3), embedded post-hoc with `K` and owner id
  `blacktensor-zkcrown-owner`, BN statistics not recalibrated.
- **Detection:** blind extraction. The wrong-key reference is the first 50
  keys of P3.4's family, recomputed at every alpha.
- **Accuracy:** eval-mode top-1, per image. Drop = acc(reference) -
  acc(embedded), with P2.5's paired 95% CI and exact McNemar test.
- **Checks:** alpha 0 reproduces `W*`'s recorded accuracy exactly (9,073 /
  10,000 test, 4,557 / 5,000 holdout), and the six alphas shared with P3.4
  give P3.4's correlations exactly.
- **Alpha grid:** fixed before the run; no alpha is selected here.

| alpha | z | Bits /128 | Change RMS | Change / carrier L2 | Test acc | Test drop vs `W*` (95% CI) | p | Holdout drop vs `W*` (95% CI) | p | Test drop vs `W` | Test loss |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 1.39 | 68 | 0 | 0 | 90.73% | 0 | 1 | 0 | 1 | +0.47 | 0.3089 |
| 0.005 | 2.50 | 75 | 0.00010 | 0.23% | 90.75% | -0.02 [-0.05, +0.01] | 0.50 | 0.00 [-0.06, +0.06] | 1 | +0.45 | 0.3088 |
| 0.01 | 3.55 | 80 | 0.00020 | 0.46% | 90.79% | -0.06 [-0.12, +0.00] | 0.11 | +0.04 [-0.06, +0.14] | 0.69 | +0.41 | 0.3086 |
| 0.02 | 5.35 | 94 | 0.00041 | 0.91% | 90.81% | -0.08 [-0.18, +0.02] | 0.15 | +0.04 [-0.08, +0.16] | 0.75 | +0.39 | 0.3084 |
| 0.03 | 6.75 | 100 | 0.00061 | 1.37% | 90.82% | -0.09 [-0.20, +0.02] | 0.15 | +0.04 [-0.11, +0.19] | 0.79 | +0.38 | 0.3082 |
| 0.05 | 8.55 | 110 | 0.00102 | 2.28% | 90.82% | -0.09 [-0.23, +0.05] | 0.25 | +0.10 [-0.10, +0.30] | 0.42 | +0.38 | 0.3079 |
| 0.075 | 9.71 | 122 | 0.00153 | 3.42% | 90.84% | -0.11 [-0.28, +0.06] | 0.25 | +0.06 [-0.18, +0.30] | 0.74 | +0.36 | 0.3077 |
| 0.1 | 10.29 | 126 | 0.00204 | 4.56% | 90.85% | -0.12 [-0.32, +0.08] | 0.27 | -0.02 [-0.29, +0.25] | 1 | +0.35 | 0.3078 |
| 0.15 | 10.80 | 128 | 0.00306 | 6.84% | 90.77% | -0.04 [-0.28, +0.20] | 0.80 | +0.10 [-0.23, +0.43] | 0.64 | +0.43 | 0.3089 |
| 0.2 | 11.01 | 128 | 0.00409 | 9.12% | 90.73% | 0.00 [-0.27, +0.27] | 1 | +0.22 [-0.17, +0.61] | 0.31 | +0.47 | 0.3111 |
| 0.3 | 11.17 | 128 | 0.00613 | 13.69% | 90.54% | +0.19 [-0.14, +0.52] | 0.28 | +0.54 [+0.08, +1.00] | 0.027 | +0.66 [+0.13, +1.19] | 0.3188 |
| 0.5 | 11.26 | 128 | 0.01021 | 22.81% | 89.51% | +1.22 [+0.79, +1.65] | 2.6e-8 | +1.74 [+1.14, +2.34] | 1.6e-8 | +1.69 [+1.12, +2.26] | 0.3509 |

Wrong keys (50) at every alpha: correlation mean within ±0.0011 of zero, sd
0.095 to 0.098, max |z| 2.03 at alpha 0 rising to 2.12 at 0.5. The correct key
is above all 50 from alpha 0.005 up. With P3.4's 100 keys that happened only
from 0.01, so "above every wrong key" depends on how many keys are tried and
is not a threshold. The largest float32 rounding error was 3.0e-8.

How to read it. Detection rises steeply up to alpha 0.1, where z = 10.29 and
126 of 128 bits are right, then saturates towards the z cap of 11.31. On both
splits accuracy shows no cost that the paired test can detect up to alpha 0.2.
The small negative test drops are inside their intervals. The first
significant cost is on the holdout at 0.3 (+0.54 pp), and at 0.5 it is
significant on both splits (+1.22 pp test, +1.74 pp holdout). Test loss is
flat to 0.1 and then climbs, so the change starts to matter to the model
somewhere between 0.1 and 0.3. Every row is one host, one key and one
embedding. The intervals cover which images were sampled, not other keys or
other training seeds. Choosing alpha from this table is P3.6. A choice made
by looking at the test drops makes the chosen model's test accuracy an
optimistic estimate, and the ledger will have to say so.

Weight-watermark detection test from `experiments/p3_7_weight_null.py`
(P3.7), result file
`results/p3.7_weight_null__seed1337__20260914T091625+0000.json`, CPU (8 worker
processes), seed 1337, commit `a757281`, clean tree. The test is defined in
`src/watermark/weight_significance.py`.
- **H0:** the suspect model is independent of `K`.
- **Statistic:** z = correlation × sqrt(128) from blind, per-tensor-centred
  extraction (P3.3).
- **Validity:** under H0, `S` is uniform and independent of the fingerprint
  `y`, so z given `y` is a unit-norm Rademacher sum and
  P(z ≥ t) ≤ exp(-t²/2) (Hoeffding). That holds for any model independent of
  `K`. It is a proof, not a measurement.
- **p-value:** `min(1, exp(-z²/2))`, one-sided.
- **Floor:** exp(-64) = 1.6e-28, because the correlation cannot exceed 1.

| alpha | Proven z\* | Correlation | Gaussian z (approximation only) | Fitted Gaussian z, dual `W*` null (descriptive) |
|---|---|---|---|---|
| 0.05 | 2.448 | 0.216 | 1.645 | 1.677 |
| 0.01 | 3.035 | 0.268 | 2.326 | 2.364 |
| 1e-3 | 3.717 | 0.329 | 3.090 | 3.134 |
| 1e-6 | 5.257 | 0.465 | 4.753 | 4.810 |
| 1e-9 | 6.438 | 0.569 | 5.998 | 6.063 |

Empirical null: 1,000 public wrong keys
`SHA-256("zk-crown/p3.7/null-key/v1" || u64 1337 || u64 j)`, each deriving its
own `P_K'` and `S'`, on each of three models.

| Model | z mean | z sd (95% CI) | Skew | Excess kurtosis | KS vs N(0,1), D (p) | min / max z | ≥ z\*(0.05), bound 50 | ≥ z\*(0.01), bound 10 | ≥ z\*(1e-3) | Bit matches mean / var (Binomial 64 / 32) |
|---|---|---|---|---|---|---|---|---|---|---|
| clean `W` (P0.5) | -0.019 | 0.967 [0.925, 1.010] | -0.04 | +0.13 | 0.027 (0.45) | -4.18 / 2.64 | 4 | 0 | 0 | 63.85 / 30.4 |
| behavioral-only (P2.3) | +0.021 | 1.009 [0.964, 1.053] | +0.09 | -0.07 | 0.023 (0.64) | -3.05 / 3.21 | 8 | 3 | 0 | 64.08 / 32.1 |
| dual `W*` (P3.6) | +0.020 | 1.008 [0.963, 1.052] | +0.09 | -0.07 | 0.026 (0.50) | -3.05 / 3.22 | 8 | 4 | 0 | 64.14 / 32.1 |

Standard errors: skew 0.077, excess kurtosis 0.155. For the Gaussian
reference, a standard normal would exceed z\*(0.05) and z\*(0.01) about 7.2
and 1.2 times in 1,000. The behavioral-only and dual models are
evaluated with the same keys and their null z values correlate at 0.999, since
the weight watermark is almost orthogonal to every wrong projection. They are
therefore one check, not two, and are not pooled. Clean `W` correlates with
them at 0.11.

Applying the test:

| Suspect | z | p-value bound | Rejects at 1e-6 |
|---|---|---|---|
| owner `K` on clean `W` | 0.18 | 0.98 | no |
| owner `K` on behavioral-only (P2.3) | 1.39 | 0.38 | no |
| owner `K` on dual `W*` (P3.6) | 10.29 | 1.1e-23 | yes (also 1e-9) |
| P3.5 alpha 0.005 | 2.50 | 0.044 | no (0.05 only) |
| P3.5 alpha 0.01 | 3.55 | 0.0019 | no (0.01 only) |
| P3.5 alpha 0.02 | 5.35 | 6.2e-7 | yes, not 1e-9 |
| P3.5 alpha 0.03 | 6.75 | 1.3e-10 | yes (also 1e-9) |
| P3.5 alpha 0.05 to 0.5 | 8.55 to 11.26 | 1.3e-16 to 3.0e-28 | yes (also 1e-9) |

How to read it. The threshold is a proven bound, so the stated false positive
rate holds for every model built without `K`, whatever its architecture
scale, training or accuracy. The 1,000-key null agrees with the theory: it is
centred, has unit spread, and is consistent with N(0, 1). Every exceedance
count at the proven thresholds is well under its bound. At 0.01 the
behavioral-only and dual models show 3 and 4 exceedances, against a Gaussian
expectation of 1.2 and a bound of 10. That is one shared sample, and it is
within the bound. With 1,000 keys the rate is only checked near 0.05 and 0.01;
the 1e-6 and 1e-9 thresholds rest on the proof. The bound is conservative. A
Gaussian threshold would be about 0.5 z lower at 1e-6, but it is not a valid
bound for every fingerprint. The final dual `W*` gives p ≤ 1.1e-23 on
unattacked weights. The floor means no single weight extraction can report
below 1.6e-28. Survival under attack is Phase 4. How this test is combined with
the behavioral one is P9.

## 8.4 Attack survival

| Attack | Strength | Clean acc | Behavioral WDR | Weight corr | Verdict |
|---|---|---|---|---|---|
| None (P4.1 control, dual `W*`) | 0 | 90.85% | 100% (100/100), p = 3.8e-96 | +0.909 (z 10.29), p ≤ 1.1e-23 | both detected at 1e-6 |
| Magnitude prune, layer-wise (P4.2) | 10% → 60% → 90% | 90.83% → 74.23% → 11.63% | 100% → 71% → 15% | +0.908 → +0.897 → +0.828 | behavioral detected up to 60%; weight detected at all 9 |
| Magnitude prune, global (P4.2) | 10% → 80% → 90% | 90.86% → 49.27% → 18.28% | 100% → 42% → 12% | +0.908 → +0.821 → +0.723 | behavioral detected up to 80%; weight detected at all 9 |
| Structured prune, L1 filters, zero-masked (P4.3) | 5% → 10% → 20% → 70% → 90% of channels | 86.02% → 74.74% → 46.92% → 11.00% → 10.00% | 94% → 70% → 19% → 12% → 10% | +0.892 → +0.875 → +0.848 → +0.484 → +0.184 | behavioral detected up to 10%; weight detected up to 70% (assumes channel re-alignment; chance accuracy from 60%) |
| Quantize FP16 (P4.4) | 16 bits | 90.85% | 100% (100/100), p = 3.8e-96 | +0.909 (z 10.29), p ≤ 1.1e-23 | both detected |
| Conv-BN fusion, FP32 control (P4.4) | 32 bits, fused | 90.85% | 100% (100/100), p = 3.8e-96 | +0.943 (z 10.67), p ≤ 1.9e-25 | both detected |
| Quantize INT8, static, simulated (P4.4) | 8 bits, fused | 90.77% | 100% (100/100), p = 3.8e-96 | +0.943 (z 10.67), p ≤ 1.9e-25 | both detected |
| Fine-tune on 5,000-image attacker holdout (P4.5) | LR 0.001–0.1 × 5/20/60 epochs (12 runs) | 90.48% (LR 0.001, 5 ep) → 86.06% (0.05, 20) → 83.36% (0.1, 60) | 100% at LR 0.001; 62% → 30% at 0.01; 4–11% at 0.05 and 0.1 | +0.909 → +0.890 (0.01, 60) → +0.573 (0.1, 60), z ≥ 6.49 | behavioral removed at LR ≥ 0.05 (every length, acc ≤ 86.1%); weight detected in all 12 |
| Prune + fine-tune | TBD | TBD | TBD | TBD | TBD |
| Distillation | TBD | TBD | TBD | TBD | TBD |
| Overwrite | TBD | TBD | TBD | TBD | TBD |

Control row from `experiments/run_attack_suite.py run --attack none --strength 0`
(P4.1), result file
`results/attacks/p4.1_none_0__seed1337__20260914T093619+0000.json`, CPU, seed
1337, commit `b807c8b`, clean tree. It is the P3.6 dual `W*`, loaded by hash
and scored by the harness with no attack. Every figure equals P3.6's, and the
harness refuses to write a `none` row on the dual model if any differs. This
row is a check on the harness, not an attack result.

- **Accuracy:** 90.85% on test (9,085 / 10,000), with 0 discordant images
  against the source.
- **Behavioral:** fired on 100 of 100 triggers, p = 3.8e-96.
- **Weight:** correlation +0.9091, z = 10.29, p ≤ 1.1e-23.

"Detected" means rejected at alpha 1e-6, a level fixed in P4.1 before any
attack was run. Each watermark is tested separately; a combined verdict is
P9.3.

Magnitude pruning sweep from `experiments/run_attack_suite.py run --config
experiments/configs/p4.2_magnitude_prune_layerwise.json
experiments/configs/p4.2_magnitude_prune_global.json --task P4.2` (P4.2), 18
result files `results/attacks/p4.2_magnitude_prune_{layerwise,global}_<s>__seed1337__20260914T0955…–1000…+0000.json`,
CPU, seed 1337, commit `506d99c`, clean tree.
- **Source:** the dual `W*` (P3.6), loaded by hash, with the trigger bundle
  digest matched to P2.3.
- **Pruning:** unstructured magnitude pruning of the 7 conv/linear weight
  tensors (307,040 weights). No fine-tuning, no BN recalibration.
- **Accuracy:** on the 10,000-image test set. The drop is paired against the
  dual `W*` (90.85%).
- **Behavioral:** fired out of N = 100 owner triggers, P2.8 p-value.
- **Weight:** P3.3 extraction, P3.7 bound.
- **Detected:** p ≤ 1e-6, per watermark.

| Scope | s | Test acc | Drop pp (95% CI) | Fired /100 | Base label | Behav. p | Behav. detected | Weight corr | z | Bits /128 | Weight p ≤ | Weight detected |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| layer-wise | 0.1 | 90.83% | +0.02 [-0.13, +0.17] | 100 | 0 | 3.8e-96 | yes | +0.9081 | 10.27 | 126 | 1.2e-23 | yes |
| layer-wise | 0.2 | 90.83% | +0.02 [-0.22, +0.26] | 100 | 0 | 3.8e-96 | yes | +0.9060 | 10.25 | 126 | 1.5e-23 | yes |
| layer-wise | 0.3 | 90.22% | +0.63 [+0.32, +0.94] | 100 | 0 | 3.8e-96 | yes | +0.9048 | 10.24 | 126 | 1.8e-23 | yes |
| layer-wise | 0.4 | 89.24% | +1.61 [+1.19, +2.03] | 100 | 0 | 3.8e-96 | yes | +0.9018 | 10.20 | 126 | 2.5e-23 | yes |
| layer-wise | 0.5 | 86.92% | +3.93 [+3.39, +4.47] | 99 | 1 | 3.0e-93 | yes | +0.9037 | 10.22 | 126 | 2.0e-23 | yes |
| layer-wise | 0.6 | 74.23% | +16.62 [+15.79, +17.45] | 71 | 8 | 7.6e-45 | yes | +0.8970 | 10.15 | 125 | 4.3e-23 | yes |
| layer-wise | 0.7 | 50.79% | +40.06 [+39.03, +41.09] | 23 | 12 | 5.3e-4 | no (1e-3 only) | +0.8827 | 9.99 | 124 | 2.2e-22 | yes |
| layer-wise | 0.8 | 25.08% | +65.77 [+64.77, +66.77] | 15 | 10 | 0.14 | no | +0.8674 | 9.81 | 124 | 1.2e-21 | yes |
| layer-wise | 0.9 | 11.63% | +79.22 [+78.35, +80.09] | 15 | 8 | 0.14 | no | +0.8275 | 9.36 | 115 | 9.2e-20 | yes |
| global | 0.1 | 90.86% | -0.01 [-0.12, +0.10] | 100 | 0 | 3.8e-96 | yes | +0.9078 | 10.27 | 126 | 1.3e-23 | yes |
| global | 0.2 | 90.76% | +0.09 [-0.11, +0.29] | 100 | 0 | 3.8e-96 | yes | +0.9066 | 10.26 | 126 | 1.4e-23 | yes |
| global | 0.3 | 90.74% | +0.11 [-0.14, +0.36] | 100 | 0 | 3.8e-96 | yes | +0.9041 | 10.23 | 126 | 1.9e-23 | yes |
| global | 0.4 | 90.14% | +0.71 [+0.39, +1.03] | 100 | 0 | 3.8e-96 | yes | +0.8995 | 10.18 | 126 | 3.2e-23 | yes |
| global | 0.5 | 89.26% | +1.59 [+1.19, +1.99] | 100 | 0 | 3.8e-96 | yes | +0.8958 | 10.13 | 125 | 5.0e-23 | yes |
| global | 0.6 | 87.59% | +3.26 [+2.76, +3.76] | 97 | 0 | 3.1e-88 | yes | +0.8876 | 10.04 | 127 | 1.3e-22 | yes |
| global | 0.7 | 80.33% | +10.52 [+9.81, +11.23] | 78 | 8 | 2.1e-54 | yes | +0.8686 | 9.83 | 123 | 1.1e-21 | yes |
| global | 0.8 | 49.27% | +41.58 [+40.54, +42.62] | 42 | 7 | 3.1e-15 | yes | +0.8214 | 9.29 | 118 | 1.8e-19 | yes |
| global | 0.9 | 18.28% | +72.57 [+71.64, +73.50] | 12 | 3 | 0.44 | no | +0.7225 | 8.17 | 108 | 3.1e-15 | yes |

Drops with McNemar p: layer-wise 0.1 p = 0.90 and 0.2 p = 0.94; global 0.1
p = 1.0, 0.2 p = 0.43 and 0.3 p = 0.43. Every other drop has p ≤ 6.4e-5.
Where the global scope put its sparsity, per tensor, at s = 0.5 and 0.9
(`features.0, 3, 7, 10, 14, 17, classifier.2`):
- **s = 0.5:** 0.16, 0.37, 0.31, 0.36, 0.42, 0.63, 0.35.
- **s = 0.9:** 0.39, 0.76, 0.73, 0.82, 0.89, 0.98, 0.77.

It prunes `features.17` hardest and the first conv least. No tensor was
emptied at any point.

How to read it:
- **Accuracy.** From 30% sparsity up, global pruning costs less accuracy than
  layer-wise pruning at the same overall sparsity: 87.59% against 74.23% at
  60%, and 80.33% against 50.79% at 70%. At 10% and 20% the two are within
  0.07 pp of each other.
- **Behavioral watermark.** It survives pruning while the model is still
  useful. At 1e-6 it is lost only at layer-wise 70% and global 90%, where test
  accuracy is 50.8% and 18.3%. Fired counts fall together with accuracy. At
  layer-wise 70% the count of 23 gives p = 5.3e-4, which rejects at 1e-3 but
  not at 1e-6. This borderline case is the one the Icebox conditional test
  names. That test was not computed.
- **Weight watermark.** It is detected at every point, including models near
  chance accuracy, and z falls only from 10.29 to 8.17 at global 90%. That is a
  measurement of this statistic on these pruned weights. Why it holds up is not
  investigated, and detecting a watermark in a destroyed model has little
  practical value to a thief.

Limits: one source model, one key, a single deterministic pruning per point, no
fine-tuning after pruning (P4.6), and no BN recalibration. Intervals cover
test-image sampling only.

Structured pruning sweep from `experiments/run_attack_suite.py run --config
experiments/configs/p4.3_channel_prune_l1.json --task P4.3` (P4.3), 10 result
files `results/attacks/p4.3_channel_prune_l1_<s>__seed1337__20260914T1545…–1548…+0000.json`,
CPU, seed 1337, commit `ccfc044`, clean tree.
- **Source:** the dual `W*` (P3.6), loaded by hash, with the trigger bundle
  digest matched to P2.3.
- **Pruning:** each of the 6 conv layers (32, 32, 64, 64, 128, 128 channels)
  loses `min(round(s·C), C-1)` output channels with the smallest filter L1
  norm, ranked on the source weights layer by layer. A removed channel's
  filter, BN weight and bias, and the next layer's input slice are set to zero.
  That is output-equivalent to deleting the channel (tested). No fine-tuning,
  no BN recalibration.
- **Scoring:** as P4.2. Accuracy on the 10,000-image test set, paired against
  the dual `W*` (90.85%). P2.8 and P3.7 tests, detected means p ≤ 1e-6.
- **Weight-watermark caveat:** the zero-masked weights keep the owner's carrier
  layout. A physically narrower model would not, and the owner would first
  have to re-insert zeros at the removed positions. That is not implemented, so
  these weight figures assume the alignment has been done.

| s | Channels removed /448 | Carrier zeroed | Narrow params | Test acc | Drop pp (95% CI) | Fired /100 | Base label | Behav. p | Behav. detected | Weight corr | z | Bits /128 | Weight p ≤ | Weight detected |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.05 | 22 | 9.03% | 280,185 | 86.02% | +4.83 [+4.28, +5.38] | 94 | 3 | 1.2e-81 | yes | +0.8922 | 10.09 | 126 | 7.5e-23 | yes |
| 0.1 | 44 | 18.18% | 252,039 | 74.74% | +16.11 [+15.29, +16.93] | 70 | 8 | 1.4e-43 | yes | +0.8752 | 9.90 | 122 | 5.1e-22 | yes |
| 0.2 | 90 | 35.22% | 199,629 | 46.92% | +43.93 [+42.89, +44.97] | 19 | 19 | 0.014 | no (0.05 only) | +0.8478 | 9.59 | 120 | 1.1e-20 | yes |
| 0.3 | 134 | 49.25% | 156,473 | 30.70% | +60.15 [+59.14, +61.16] | 12 | 14 | 0.43 | no | +0.8286 | 9.37 | 115 | 8.3e-20 | yes |
| 0.4 | 180 | 62.46% | 115,817 | 16.90% | +73.95 [+73.04, +74.86] | 9 | 6 | 0.79 | no | +0.7991 | 9.04 | 117 | 1.8e-18 | yes |
| 0.5 | 224 | 73.26% | 82,554 | 15.81% | +75.04 [+74.14, +75.94] | 10 | 6 | 0.68 | no | +0.7269 | 8.22 | 112 | 2.1e-15 | yes |
| 0.6 | 268 | 82.25% | 54,871 | 10.01% | +80.84 [+80.03, +81.65] | 10 | 6 | 0.68 | no | +0.6279 | 7.10 | 110 | 1.1e-11 | yes |
| 0.7 | 314 | 89.67% | 31,981 | 11.00% | +79.85 [+79.00, +80.70] | 12 | 9 | 0.43 | no | +0.4839 | 5.47 | 95 | 3.1e-7 | yes |
| 0.8 | 358 | 94.79% | 16,185 | 9.97% | +80.88 [+80.04, +81.72] | 8 | 5 | 0.88 | no | +0.3301 | 3.73 | 80 | 9.4e-4 | no (1e-3 only) |
| 0.9 | 404 | 98.39% | 5,049 | 10.00% | +80.85 [+80.04, +81.66] | 10 | 7 | 0.68 | no | +0.1841 | 2.08 | 67 | 0.11 | no |

"Carrier zeroed" is the fraction of the 307,040 conv/linear weights that are
zero afterwards. "Narrow params" is the parameter count of the equivalent
physically pruned network (source 307,946). At s = 0.5 it is 82,554, the same
as the width-16 student, as expected for halving every layer. Removed
channels per layer at s = 0.05: 2, 2, 3, 3, 6, 6; at s = 0.1: 3, 3, 6, 6, 13,
13. Every drop has McNemar p ≤ 1.4e-68 (the smallest underflow to 0 in float64).

How to read it:
- **Accuracy.** Structured pruning is much more destructive than P4.2's
  magnitude pruning. Removing 5% of channels costs 4.83 pp, 10% costs 16.1 pp,
  and the model is at chance from s = 0.6. The fairer comparison is at a
  similar fraction of weights zeroed, not at equal s. 9.0% zeroed here gives
  86.02%, against 90.83% at P4.2 layer-wise 10%. 18.2% zeroed gives 74.74%,
  against 90.83% at layer-wise 20%.
- **Behavioral watermark.** It survives only as long as the model is useful:
  94/100 at 86.0% accuracy and 70/100 at 74.7%. At s = 0.2 (46.9% accuracy) it
  fires on 19, p = 0.014, not detected at 1e-6. From s = 0.3 on, fired counts
  of 8 to 12 are what a model independent of `K` gives (bound 11.1).
- **Weight watermark.** Detected at 1e-6 up to s = 0.7. Lost at 0.8 (z 3.73,
  rejects at 1e-3 only) and 0.9 (z 2.08). Unlike magnitude pruning, z falls
  steadily, because whole slices of the carrier are zeroed rather than only
  its smallest entries. From s = 0.4 on the model is at or below 16.9%
  accuracy. Detecting a watermark in a model that useless has little practical
  value to a thief, and every weight figure here assumes the owner can re-align
  physically deleted channels.

Limits: one source model, one key, one criterion (L1 norm, same fraction per
layer), a single deterministic pruning per point, no fine-tuning after pruning
(P4.6), no BN recalibration. Global or BN-scale channel criteria were not run.
Intervals cover test-image sampling only.

Post-training quantization from `experiments/run_attack_suite.py run --config
experiments/configs/p4.4_ptq_fused_fp32.json experiments/configs/p4.4_ptq_fp16.json
experiments/configs/p4.4_ptq_int8_static.json --task P4.4` (P4.4), result files
`results/attacks/p4.4_ptq_fused_fp32_32__…T160757`, `p4.4_ptq_fp16_16__…T160812`
and `p4.4_ptq_int8_static_8__…T160859` (all `seed1337`, `+0000.json`), CPU,
seed 1337, commit `ab853c5`, clean tree.
- **Source:** the dual `W*` (P3.6), loaded by hash, with the trigger bundle
  digest matched to P2.3. No retraining in any row.
- **FP16:** every floating tensor, including BN statistics, cast to float16 and
  scored in `main_model` in float32. Activations are not rounded to float16.
- **Fused FP32:** each conv-BN-ReLU fused (`W_f = W·gamma/sqrt(var+eps)`,
  `b_f = beta - mean·gamma/sqrt(var+eps)`), no rounding.
- **INT8:** the fused model, simulated static PTQ. Weights are qint8,
  per-channel symmetric (`scale = max|W_c|/127.5`). Activations (input, every
  conv-ReLU output, the logits) are quint8, per-tensor affine, with min/max
  from 1,000 attacker-holdout images (index digest `2f95da01…9223fb`) and the
  full [0, 255] range. Biases are int32. Inference runs on integer codes with
  exact sums. It is simulated, not run on an integer backend.
- **Weight extraction** reads the weights as shipped: fused, and for INT8
  dequantized. Fusion is not undone with the owner's BN statistics.
- **Scoring:** accuracy and the behavioral test run on the fused runtime for
  the fused and INT8 rows. Otherwise as P4.2, detected means p ≤ 1e-6.

| Row | Test acc | Correct | Drop pp (95% CI) | Only source right / only attacked right | McNemar p | Test loss | Fired /100 | Mean / min target prob | Behav. p | Weight corr | z | Bits /128 | Amplitude | Weight p ≤ |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| dual `W*` (P4.1 control) | 90.85% | 9,085 | 0 | 0 / 0 | 1 | 0.3078 | 100 | 0.9994 / 0.9932 | 3.8e-96 | +0.9091 | 10.29 | 126 | 0.1061 | 1.1e-23 |
| FP16 | 90.85% | 9,085 | +0.00 [-0.03, +0.03] | 1 / 1 | 1 | 0.3078 | 100 | 0.9994 / 0.9932 | 3.8e-96 | +0.9091 | 10.29 | 126 | 0.1061 | 1.1e-23 |
| fused FP32 | 90.85% | 9,085 | 0 | 0 / 0 | 1 | 0.3078 | 100 | 0.9994 / 0.9932 | 3.8e-96 | +0.9431 | 10.67 | 128 | 0.2786 | 1.9e-25 |
| INT8 static | 90.77% | 9,077 | +0.08 [-0.08, +0.24] | 39 / 31 | 0.40 | 0.3090 | 100 | 0.9994 / 0.9937 | 3.8e-96 | +0.9431 | 10.67 | 128 | 0.2786 | 1.9e-25 |

What each conversion does to the carrier weights:
- **FP16:** largest rounding change 2.4e-4, change 0.021% of the carrier L2
  norm, no weight underflowed to zero. The weight correlation is unchanged to
  6 decimals (0.909120).
- **Fusion:** the carrier changes by 153% of its L2 norm. The per-channel BN
  scale ranges, by conv layer, are 0.21–0.88, 0.54–1.40, 0.52–0.89,
  0.85–1.41, 0.64–2.20 and -0.09–5.21. The last conv has one channel with a
  negative scale.
- **INT8 on top of fusion:**
  - Weight rounding error is 0.56% to 1.03% of each tensor's norm, and 1.37%
    of carrier weights round to 0.
  - The correlation moves from 0.943058 to 0.943050.
  - Input qparams: scale 0.0161, zero point 123. Logit range [-17.8, 46.9],
    scale 0.254, zero point 70.
  - On the attacker's own 1,000 calibration images, INT8 top-1 agrees with the
    fused FP32 model on 99.4%.

How to read it:
- **Accuracy.** Neither conversion has a measurable cost. Fusion is exact (no
  discordant images). FP16 swaps one image each way. INT8 loses 0.08 pp, not
  distinguishable from zero (p = 0.40).
- **Behavioral watermark.** Unaffected: 100/100 in every row, and the target
  probabilities barely move.
- **Weight watermark.** Unaffected by FP16 and by INT8 rounding. Conv-BN
  fusion rescales every channel and changes the carrier by more than its own
  norm, yet the blind correlation went up (0.909 → 0.943, z 10.29 → 10.67)
  rather than down, with all 128 bits recovered. Why is not investigated. It is
  one model and one set of BN statistics, so it is not a claim that fusion
  helps the watermark in general. The P3.7 bound applies unchanged, because it
  holds for any model independent of `K`.

Limits: one source model, one key, one calibration draw of 1,000 images;
min/max observers only (no histogram/percentile calibration); INT8 simulated,
not run on a real integer backend (the tests compare it with `torch.ao` on a
seeded model only); FP16 activations computed in float32; lower bit widths not
run. Intervals cover test-image sampling only.

Fine-tuning sweep (P4.5).
- **Training:** 12 `run_attack_suite.py apply` runs on Colab (Tesla T4,
  commit `4897934`, clean tree), configs
  `experiments/configs/p4.5_finetune_lr{0.001,0.01,0.05,0.1}.json`. Apply
  records are in `results/attacks/p4.5_apply/`; the weights files are
  gitignored.
- **Scoring:** locally on CPU with `run_attack_suite.py evaluate --task P4.5`,
  commit `260b64a`, clean tree, seed 1337. Result files are
  `results/attacks/p4.5_finetune_holdout_lr<LR>_<epochs>__seed1337__20260914T1803…–1807…+0000.json`.
- **Attack:** start from the dual `W*` (P3.6, hash-checked on Colab and in
  every record). Train on the 5,000-image attacker holdout only, with the P0.5
  recipe (SGD Nesterov 0.9, wd 5e-4, batch 128, crop and flip augmentation,
  1-epoch linear warmup, cosine to 0 over the run). Final-epoch weights; no
  key, triggers or test data on Colab.
- **Scoring detail:** test accuracy with the paired drop against the dual
  `W*` (90.85%); P2.4 WDR with the P2.8 p-value; P3.3 weight correlation with
  the P3.7 bound; detected means p ≤ 1e-6.
- **Holdout column:** accuracy on the un-augmented holdout at the end of
  training, measured on Colab. These are the attacker's training images, so it
  is a fit figure, not a generalisation estimate.

| LR | Epochs | Test acc | Drop pp (95% CI) | Test loss | Holdout (train data) | Holdout − test | Fired /100 | Base label | Mean target prob | Behav. p | Behav. detected | Weight corr | z | Bits /128 | Weight p ≤ | Weight detected |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.001 | 5 | 90.48% | +0.37 [+0.07, +0.67] | 0.3077 | 92.16% | +1.68 | 100 | 0 | 0.999 | 3.8e-96 | yes | +0.9087 | 10.28 | 126 | 1.1e-23 | yes |
| 0.001 | 20 | 90.49% | +0.36 [-0.01, +0.73] | 0.3100 | 94.00% | +3.51 | 100 | 0 | 0.996 | 3.8e-96 | yes | +0.9079 | 10.27 | 126 | 1.2e-23 | yes |
| 0.001 | 60 | 90.09% | +0.76 [+0.35, +1.17] | 0.3177 | 96.84% | +6.75 | 100 | 0 | 0.981 | 3.8e-96 | yes | +0.9068 | 10.26 | 126 | 1.4e-23 | yes |
| 0.01 | 5 | 89.02% | +1.83 [+1.35, +2.31] | 0.3348 | 94.62% | +5.60 | 62 | 11 | 0.536 | 4.8e-34 | yes | +0.9073 | 10.27 | 126 | 1.3e-23 | yes |
| 0.01 | 20 | 89.05% | +1.80 [+1.30, +2.30] | 0.3607 | 98.30% | +9.25 | 46 | 23 | 0.409 | 1.9e-18 | yes | +0.9036 | 10.22 | 126 | 2.0e-23 | yes |
| 0.01 | 60 | 87.99% | +2.86 [+2.32, +3.40] | 0.4551 | 99.90% | +11.91 | 30 | 22 | 0.274 | 2.5e-7 | yes (k\* = 29) | +0.8898 | 10.07 | 123 | 9.9e-23 | yes |
| 0.05 | 5 | 85.69% | +5.16 [+4.52, +5.80] | 0.4281 | 92.60% | +6.91 | 11 | 39 | 0.101 | 0.56 | no | +0.8946 | 10.12 | 127 | 5.7e-23 | yes |
| 0.05 | 20 | 86.06% | +4.79 [+4.17, +5.41] | 0.4870 | 98.20% | +12.14 | 8 | 40 | 0.087 | 0.88 | no | +0.8541 | 9.66 | 121 | 5.3e-21 | yes |
| 0.05 | 60 | 85.30% | +5.55 [+4.89, +6.21] | 0.6100 | 99.88% | +14.58 | 6 | 40 | 0.069 | 0.97 | no | +0.7297 | 8.26 | 106 | 1.6e-15 | yes |
| 0.1 | 5 | 81.86% | +8.99 [+8.27, +9.71] | 0.5523 | 87.92% | +6.06 | 9 | 51 | 0.082 | 0.79 | no | +0.8737 | 9.89 | 122 | 6.0e-22 | yes |
| 0.1 | 20 | 83.74% | +7.11 [+6.42, +7.80] | 0.5609 | 96.72% | +12.98 | 8 | 54 | 0.072 | 0.88 | no | +0.8097 | 9.16 | 117 | 6.0e-19 | yes |
| 0.1 | 60 | 83.36% | +7.49 [+6.80, +8.18] | 0.7143 | 99.64% | +16.28 | 4 | 47 | 0.046 | 0.997 | no | +0.5732 | 6.49 | 94 | 7.4e-10 | yes |

McNemar p for the drops: LR 0.001: 0.020, 0.061, 3.4e-4. Every other row
≤ 1.4e-12. The dual `W*` scores 91.16% on this holdout before fine-tuning.
After the first epoch (training diagnostics from the apply records), holdout
accuracy was 91.22%, 88.76%, 27.12% and 34.18% for LR 0.001, 0.01, 0.05 and
0.1. The two high rates knock the model far off first, then retrain it on the
5,000 images. Fine-tuning time on the T4 was 3.1–3.3 s per epoch, 1,087 s in
total.

How to read it:
- **Behavioral watermark.** Fine-tuning at a high learning rate removes it.
  - At LR 0.05 and 0.1, every run, including 5 epochs, leaves 4 to 11 of 100
    triggers firing. That is the range a model independent of `K` gives
    (bound 11.1), so detection fails at every level, not only 1e-6.
  - The triggers are not moved to some other response. 39 to 54 of them are
    classified as their base image's label: the model has stopped reacting
    to the key perturbation.
  - The cheapest removal in this grid is LR 0.05 for 20 epochs, at 86.06% test
    accuracy (−4.79 pp).
  - LR 0.01 erodes it steadily (62 → 46 → 30 fired) without removing it. The
    60-epoch run's 30 is one above k\*(1e-6) = 29, a borderline case. The
    Icebox conditional test would have more power there, given 22 base-label
    hits; it was not computed.
  - LR 0.001 leaves it untouched.
- **Weight watermark.** Detected at 1e-6 in all 12 rows. Correlation falls with
  both learning rate and length: at LR 0.1 from 0.874 (5 epochs) to 0.573
  (60 epochs, z 6.49, 94 bits), still above z\*(1e-9) = 6.438. Every run that
  removed the behavioral watermark kept the weight watermark at z ≥ 6.49. The
  trend within the grid is downward, and whether more epochs, a higher rate or
  more data would push it under the threshold is not measured.
- **Accuracy and memorisation.** As the owner noted, holdout accuracy reaches
  99.6% to 99.9% in the 60-epoch runs at LR ≥ 0.01. These are the images being
  trained on, and test accuracy does not follow: the holdout-minus-test gap
  grows with LR and length, to +16.3 pp. Test loss rises with length at every
  LR ≥ 0.01 (0.335 → 0.361 → 0.455 at LR 0.01; 0.428 → 0.487 → 0.610 at
  0.05; 0.552 → 0.561 → 0.714 at 0.1). Test accuracy is flat or worse after
  20 epochs. So the long runs memorise 5,000 images; they do not recover
  generalisation. A thief with this little data pays for removal in test
  accuracy (4.8 to 9.0 pp here). One with more data might pay less. That is
  not tested.

Limits: one fine-tuning run per cell and one seed. GPU training is not
bit-reproducible. One source model and one key. The attacker has 5,000 images.
Plain SGD fine-tuning only, with no removal-specific loss. Intervals cover
test-image sampling only.

## 8.5 ZK measurements

| Metric | Track A (Circom) | Track B (EZKL) |
|---|---|---|
| Constraint count / circuit rows | TBD | TBD |
| Setup peak RAM | TBD | TBD |
| Proving key size | TBD | TBD |
| Verification key size | TBD | TBD |
| Prove time | TBD | TBD |
| Verify time | TBD | TBD |
| Proof size | TBD | TBD |
| Ran on Colab free without OOM | TBD | TBD |

---

# 9. ICEBOX

Ideas that are explicitly not in scope right now. Add here instead of expanding the current phase.

- On-chain EVM verifier.
- Watermarking a transformer or LLM.
- Adaptive adversary who knows the watermarking algorithm but not `K`.
- Multi-owner or threshold commitments.
- Recursive proof composition.
- Blur/JPEG input-preprocessing attack: may weaken the high-frequency per-trigger noise (see `src/watermark/README.md`), untested; candidate addition to the Phase 4 attack suite.
- Target-class bias on noise (P2.6): on uniform noise, `W*`'s predictions land in the owner's target classes more often than clean `W`'s (chance level 12.7% vs 8.9%). This is a single untested pair of numbers; revisit only if it shows up again elsewhere.
- Conditional detection test (P2.8): the P2.8 p-value uses the worst-case bound `Binomial(N, 1/9)`. Given the count `m` of triggers on which the suspect does *not* predict the base label, the fired count is exactly `Binomial(m, 1/9)` under H0. That is still valid and has more power against accurate models (the P2.8 null models had m of about 51 to 58). Worth revisiting if attacked models in Phase 4 end up with borderline p-values.
- Per-layer scaled weight embedding (P3.2): the P3.2 formula moves every carrier parameter by the same amount, even though `main_model` layers differ in weight scale. Scaling the change per layer (for example by layer weight RMS) would change the formula and the extractor. Revisit only if P3.5 shows the uniform version costs too much accuracy for its detection strength.
- During-training weight embedding (P3.6): the final dual `W*` uses post-hoc embedding at alpha 0.1. Embedding during training might let the network adapt around the watermark and survive fine-tuning better, but that is unmeasured. It would need a new design (for example re-adding the watermark after each step, or a loss term), a GPU run, and a redo of P3.5, because alpha would mean something different. Revisit only if Phase 4 shows the post-hoc weight watermark does not survive attacks well.
- Channel re-alignment for physically pruned suspects (P4.3): the P4.3 weight-watermark figures use zero-masked weights, which keep the carrier layout. A thief who deletes channels ships a narrower model, and the extractor then reports "not applicable". The owner holds `W*` and the kept weights are unchanged, so matching surviving filters back to their original positions and re-inserting zeros should be possible. It is not built or measured. Candidate for the P9 auditor.
- Fusion raised the weight correlation (P4.4): conv-BN fusion moved the dual `W*` blind weight correlation from 0.909 to 0.943 (z 10.29 → 10.67), even though it changed the carrier by 153% of its norm. Unexplained, one model only; revisit if it recurs in later attacks.

---

# 10. SESSION LOG

Append one line per session: date, tasks touched, key outcome.

- 2026-09-13: P0.1. Repo skeleton created per Section 3, `.gitignore` written, git repo initialised, `CLAUDE.md` moved out of `.venv/` to the repo root so it is actually tracked. No experiments run, no numbers produced.
- 2026-09-13: P0.2. `src/utils/seeding.py` and `src/utils/results.py` written; 16 tests in `tests/test_utils.py` pass. Result records carry a dirty-tree flag and warn when set. No numbers produced.
- 2026-09-13: P0.3. `notebooks/00_setup.ipynb` written (20 cells), nbformat-valid, logic executed locally. Supports both source routes (git clone and 0.5 handoff zip). Not yet run on Colab, so section 8.1 is still `TBD`.
- 2026-09-13: P0.4. `main_model` implemented, 307,946 params, 34 tests pass. CPU-only torch installed locally, so watermark/crypto/auditor logic can now be tested off-Colab. Section 8.2 params filled; accuracy still `TBD`.
- 2026-09-13: P0.5 HANDED OFF, not done. Data pipeline, resumable training loop, double-buffered per-epoch checkpointing, entry point, Colab notebook and `handoff/P0.5_colab.zip` all written; 52 tests pass and the zip was verified by extracting it outside the repo and running its tests plus a synthetic smoke run. Awaiting the Colab run. Checkbox stays `[ ]` per 0.5 step 6. Decision made here that affects later phases: CIFAR-10 train is split 45,000 / 5,000, the 5,000 reserved as the P4.5/P4.6 attacker holdout, fixed by `SPLIT_SEED = 20260913`.
- 2026-09-13: P0.5 pre-upload self-review, still NOT RUN. Found and fixed a real resume bug: every resumed epoch replayed epoch 0's shuffle order, now caught by a test that a run interrupted after every epoch is bit-identical to an uninterrupted one. Also: resume refuses changed hyperparameters or seed, `--no-resume` clears stale slots, notebook commands now fail loudly, Drive-mount and GPU guards, zip carries its commit in `BUILD_INFO.json`, and `W` is exported as final-epoch weights rather than test-selected `best.pt`. Checkpoint format and split unchanged. 60 tests pass.
- 2026-09-13: P0.5 first Colab run FAILED: loss flat at ln(10), eval_acc exactly 10.00% for 12 epochs. Reproduced on CPU. At lr 0.1 with no warmup, the first SGD steps on the 2,048-input classifier blew logits to std ~50, and ~98% of the last conv block's channels died within 12 steps. Fix: per-step linear LR warmup, `TrainConfig.warmup_epochs=1.0` (352 steps). The scheduled LR is restored before `scheduler.step()`, so checkpoint and scheduler state format are unchanged. Split untouched. The one config key added makes the failed run's Drive checkpoints refuse to resume, on purpose. CPU check of the fixed entry point on real CIFAR-10: 57% / 66% test accuracy after epochs 1 / 2, verification only, not a ledger number. 65 tests pass. Zip rebuilt; checkbox still `[ ]`.
- 2026-09-13: P0.5 DONE. Second Colab run (commit `55f2607`, T4) finished 60/60 epochs in 1,235 s. Test accuracy 91.20%, holdout 90.88%. I checked the returned result JSON: it is internally consistent (history, best epoch, timing, split, seed) and the local weights file's SHA-256 matches the recorded hash. The weights load strictly with 307,946 params. Nothing was retrained. Section 8.2 filled. P0.6 left unticked.
- 2026-09-13: P0.6 ticked; the baseline was already in 8.2. Section 8.1 filled from the P0.5 JSON's `environment` block (T4, torch 2.11.0+cu128). System RAM stays `TBD` because the JSON does not record it.
- 2026-09-13: P0.7 HANDED OFF, not done. `zk_model` implemented (`src/models/zk_model.py`): three 3x3 stride-2 conv+ReLU layers (8/16/16) and a linear classifier, 6,138 params (budget 10K), 2,608 ReLU elements per input. No BatchNorm, dropout or pooling, to keep the Phase 8 circuit simple. Also added: MNIST pipeline, entry point `experiments/p0_7_train_zk_model.py` using the P0.5 loop, notebook, instructions and `handoff/P0.7_colab.zip`. Decisions: `zk_model` trains on all 60,000 MNIST images, with no holdout since it is never attacked; lr 0.05, 20 epochs, 1-epoch warmup. Before choosing the LR I ran one CPU epoch per candidate on the training set only, to rule out a P0.5-style collapse; no test evaluation, nothing recorded. Result records now include `environment.system_ram_gb`, for section 8.1. 80 tests pass. Checkbox stays `[ ]`.
- 2026-09-13: P0.7 DONE. Colab run (commit `ca412fb`, T4) finished 20/20 epochs with 276.65 s of epoch time. Test accuracy 98.96%; the best epoch (index 17) reached 99.02% but was not selected. Checks: the JSON is internally consistent, the local weights file's SHA-256 matches, and the weights load strictly with 6,138 params. CPU inference on the MNIST test set reproduces the final epoch's accuracy and loss exactly, which proves these are final-epoch weights. Peak allocated VRAM was 25 MB. Section 8.2 `zk_model` row filled; section 8.1 system RAM filled with 12.67 GiB. Nothing was retrained.
- 2026-09-13: P1.1. `src/watermark/keygen.py`: a stdlib-only HMAC-SHA256 counter-mode stream keyed by the 32-byte `K`, with a length-prefixed purpose label for domain separation, plus exact `uniforms` and unbiased `randbelow`. 44 new tests, 124 in total, all pass. The known-answer vector was cross-checked with `openssl`. No triggers generated, no numbers produced.
- 2026-09-13: P1.2. `src/watermark/triggers.py`: each trigger is a key-selected image from the training split plus an independent key-derived ±16-level sign pattern, in integer arithmetic. Two decisions made here: bases come only from the 45,000-image training split, and every trigger gets its own perturbation rather than one shared pattern. The per-trigger choice keeps triggers closer to independent for the P2.8 test. 25 new tests, 149 in total, all pass. No figure, no numbers produced.
- 2026-09-13: P1.3. Trigger figures and an amplitude sweep written to `figures/` with a public demo key; perturbation statistics recorded in 8.3 (A = 16: mean PSNR 24.17 dB). By-eye verdict: A = 16 is visible but not garbage, so the default is kept. matplotlib installed into the local venv; it was already in `requirements.txt`. No model run.
- 2026-09-13: P1.4. Determinism and cross-key independence tests for the trigger set: 14 new, 163 in total, all pass. The demo-key set regenerates to the SHA-256 recorded by P1.3. Independence is tested on base indices and sign patterns for related (single-bit flip) and unrelated keys, against exact null distributions, and each check is confirmed able to fail. No library code changed, no ledger numbers.
- 2026-09-13: P1.5. `src/watermark/README.md` written: the trigger design is additive per-trigger key noise, with a one-paragraph rationale against learned and patch triggers and the unmeasured costs stated. Documentation only; Phase 1 is complete.
- 2026-09-13: Icebox line added for the untested blur/JPEG preprocessing attack. P2.1: `S` = 128-bit HMAC-SHA256 PRF of `owner_id` under `K`, via the P1.1 stream with label `signature/v1/owner:<id>`. Documented, with the "not a public-key signature, treated as secret" caveat. 27 new tests, 190 in total, all pass. No numbers produced.
- 2026-09-13: P2.2. `src/watermark/responses.py`: each trigger maps to its own key-derived target class, uniform over the 9 classes other than its base image's label. Chosen over a single owner class so that a model independent of `K` has a fire rate of at most 1/9 per trigger, independently across triggers, which gives P2.8 a model-independent null. 30 new tests, 220 in total, all pass. No ledger numbers.
- 2026-09-13: P2.3 HANDED OFF, not done. The real master key `K` was created locally at `secrets/K.bin` (gitignored, never printed) by `experiments/make_master_key.py`, which refuses to overwrite. `K` never goes to Colab. Instead, `experiments/p2_3_make_trigger_bundle.py` builds `secrets/trigger_bundle.npz`, 100 triggers plus P2.2 targets, SHA-256 `fbd65ec7…22baec8`. That digest is baked into the notebook, and the run records it. Code added: `src/watermark/bundle.py`, `src/watermark/behavioral.py` (`TriggerMixLoader`), the entry point `experiments/p2_3_train_watermarked.py`, and an optional `epoch_metrics` hook in `fit` (P0.5 and P0.7 behaviour unchanged). Decisions: train from scratch with the exact P0.5 recipe, seed and clean batch order; append 4 un-augmented trigger samples per batch of 128 (a starting value, not tuned); keep base images in the clean set. 22 new tests, 242 in total, all pass. They include a synthetic check that the mixer really embeds triggers and a resume-equivalence test. Checkbox stays `[ ]`.
- 2026-09-13: P2.3 DONE. Colab run (commit `0aaea8e`, T4) finished 60/60 epochs with 1,177.13 s of epoch time. Test accuracy 90.73%, holdout 91.14%; the best epoch (index 58) reached 90.82% but was not selected. Checks: the JSON is internally consistent (60-entry history, best and final epochs, timing, trigger bundle digest, params identical to P0.5 apart from the trigger fields). The local weights file's SHA-256 matches, and the weights load strictly with 307,946 params. CPU inference reproduces the final epoch's test accuracy and loss exactly, plus the holdout accuracy, which proves these are final-epoch weights. Section 8.2 `W*` row and 8.3 N filled. Training trigger accuracy (100%) is not recorded as WDR, and the accuracy drop is not computed; those are P2.4 and P2.5. Nothing was retrained.
- 2026-09-13: P2.4. `src/watermark/detection.py` (WDR = k/N, aggregate-only summary) and `experiments/p2_4_measure_wdr.py`, which regenerates the triggers from `K`, checks them against P2.3's bundle digest, and loads models by hash. CPU run: `W*` WDR 100% (100/100). Controls: clean `W` fires on 3/100 triggers; `W*` and `W` each fire on 1/100 unperturbed base images. 15 new tests, 257 in total, all pass. FPR (P2.6), p-value (P2.8) and accuracy drop (P2.5) not computed.
- 2026-09-13: P2.5. `experiments/p2_5_accuracy_drop.py`, `src/utils/stats.py` (paired difference plus exact McNemar) and `per_sample_correct` in `src/training/loop.py`. CPU re-scoring of hash-checked `W` and `W*` reproduced both Colab accuracies exactly. Test drop is 0.47 pp (91.20% → 90.73%), 95% CI [-0.04, +0.98], McNemar p = 0.077. Holdout drop is -0.26 pp, p = 0.53. Seed variance not measured. 18 new tests, 275 in total, all pass.
- 2026-09-13: P2.6. `src/watermark/false_positives.py` and `experiments/p2_6_false_positive_rate.py`: non-trigger inputs are scored against the owner's real targets by slot, with the chance level computed from each model's predictions. `W*` input-level FPR is 12.6% on random noise (chance 12.7%) and 10.0% on clean unrelated test images (chance 10.0%). Clean `W` gives 8.8% and 9.1%. Extra noise-decoy set: `W*` 11.9%, `W` 10.3%. Mid-task I corrected my own reading: fired minus chance is fixed-pairing noise, not a leakage test, and the docstrings now say so. 12 new tests, 287 in total, all pass.
- 2026-09-13: Icebox line added for the P2.6 target-class bias on noise. P2.7 HANDED OFF, not done.
  - Sweep of five new runs, each the P2.3 recipe with a different trigger ratio: 0.013%, 0.049%, 0.196%, 0.782% and 12.5% trigger samples per clean sample. P2.3 (3.13%) and P0.5 (0%) are reused.
  - Getting below one trigger per batch needed `TriggerMixLoader(trigger_every=k)`, which adds triggers to every k-th batch only. With k = 1 the trigger order is byte-identical to P2.3's (tested).
  - `experiments/p2_3_train_watermarked.py` now exposes `train(args, run_name, task)`. P2.3's own record shape and resume config are unchanged (tested).
  - Added `experiments/p2_7_ratio_sweep.py` (one run or all, skips runs already complete on Drive after checking the weights hash), the local CPU analysis `experiments/p2_7_analyze_sweep.py` (P2.4 WDR plus P2.5 paired drop per model, 3-panel figure), notebook, instructions and `handoff/P2.7_colab.zip`.
  - Estimated ~1 h 50 min, under the 2 h split threshold but close, so every run has its own resumable cell. 20 new tests, 307 in total, all pass. Checkbox stays `[ ]`.
- 2026-09-14: P2.7 DONE. Five Colab runs (commit `4c1e3da`, T4) returned, 60/60 epochs each; nothing was retrained.
  - Checks: every JSON is consistent with its history and with P2.3's params apart from the trigger fields, and every weights SHA-256 matches. CPU re-scoring reproduced all five test accuracies exactly.
  - WDR from 0% to 12.5% trigger samples per clean sample: 3, 9, 19, 64, 100, 100, 100 (out of 100). It saturates at 0.782%.
  - Test drops: +0.19, +0.46, +0.25, +0.94, +0.47, +0.34 pp, with no trend in the ratio. Only the 0.782% drop has a CI excluding zero. One run per ratio.
  - Figure written; a repeat analysis gave an identical record and figure. The analysis record reads `dirty: true` only because the returned JSONs were untracked when it ran; no code changed.
- 2026-09-14: P2.8. `src/watermark/significance.py` gives the exact binomial p-value under H0 "independent of the owner's targets", with bound 1/9 per trigger, plus thresholds. Assumptions are written down, including that `K` must be committed before any dispute.
  - `experiments/p2_8_detection_test.py`: `W*` 100/100 → p = 3.8e-96, clean `W` 3/100 → 0.999. P2.7 sweep: 9 → 0.79, 19 → 0.014, 64 → 2.6e-36.
  - Thresholds for N = 100: k* = 17 / 20 / 23 / 29 / 35 at alpha 0.05 down to 1e-9.
  - Null check over 1,000 public wrong keys: 0 rejections for either model at 0.05 or 0.01. Mean fired 5.70 and 6.38 against the 11.11 bound; calibration z +0.04 and -1.36.
  - 33 new tests, 340 in total, all pass. Icebox line added for the conditional, more powerful test.
  - Follow-up: re-ran P2.8 from the clean committed tree (`4c90006`). The result now reads `dirty: false` and every number reproduced exactly. It replaces the dirty-tree result file.
- 2026-09-14: P3.1. `src/watermark/projection.py`: `P_K` is a 128 x `dim` key-derived ±1/sqrt(dim) matrix from the `projection/v1/rademacher/dim=<dim>` stream, with integer generation, no orthonormalisation and unit rows, plus `project` and `back_project`. Generation takes about 1 s on the local CPU at `dim` = 307,040. 41 new tests, 381 in total, all pass. Carrier choice, `alpha` and centring are left to P3.2 and P3.3.
- 2026-09-14: P3.2. `src/watermark/carrier.py` makes every conv and linear weight the carrier (`dim` = 307,040 for `main_model`). `src/watermark/weight_embedding.py` does a post-hoc `W* = W + alpha * P_K^T * S` on the carrier only, with an aggregate-only summary. `alpha` has no default and is left to P3.5. On clean `W` with a test key, a one-off smoke run at alpha 1.0 took about 1.2 s; it is not recorded, because alpha 1.0 is not a chosen value. 42 new tests, 423 in total, all pass. Icebox line added for per-layer scaling.
- 2026-09-14: P3.3. `src/watermark/weight_extraction.py` is a blind extractor: it centres each carrier tensor, projects with `P_K`, and scores against `S` with normalised correlation, amplitude, RMS and bit matches. It reports aggregates only. 27 new tests, 450 in total, all pass. They cover recovery, invariances and unit-level wrong-key checks. No real-model numbers; those are P3.4 and P3.7.
- 2026-09-14: P3.4. The owner fixed the owner id as `blacktensor-zkcrown-owner` (`PROJECT_OWNER_ID`).
  - `experiments/p3_4_key_specificity.py` watermarks `W*` and `W` post-hoc at alpha 0 to 0.1 and extracts with `K`, with 100 wrong keys, and with a wrong `P_K'` against the true `S`. The code was committed first (`2a2eee9`), so the run is on a clean tree.
  - `W*`: correct-key correlation +0.12 (control) to +0.91, against wrong keys at mean +0.001, sd 0.095, max |z| 2.71. The gap is 5.0 wrong-key sd at alpha 0.02 and 9.6 at 0.1.
  - Finding: z is capped at sqrt(128) = 11.3 by construction.
  - 10 new tests, 460 in total, all pass.
- 2026-09-14: P3.5. `experiments/p3_5_alpha_sweep.py`: 12 alphas post-hoc on `W*`, with detection (correct key plus 50 wrong keys) and paired accuracy drops on test and holdout against `W*` and `W`.
  - z runs from 1.39 (alpha 0) to 10.29 (0.1) and saturates towards 11.31. No significant accuracy cost up to 0.2 on either split. Holdout +0.54 pp at 0.3; +1.22 pp test and +1.74 pp holdout at 0.5.
  - Checks: alpha 0 reproduced `W*` exactly, and the shared alphas reproduced P3.4 exactly.
  - The first run was flagged dirty only because its fresh figure was untracked. I fixed overlapping figure labels (`e943c19`) and re-ran from the clean tree, writing the figure outside the repo. Metrics were identical, and the figure was byte-identical to one rendered from the first run.
  - No alpha selected. 6 new tests, 466 in total, all pass.
- 2026-09-14: P3.6. Owner decision: post-hoc at alpha 0.1, so no Colab run was needed.
  - `experiments/p3_6_make_dual_model.py` built the final dual `W*` (SHA-256 `7a9a9f14…b434c4`, gitignored) and verified it on the reloaded file.
    - Weight watermark z = 10.29, 126/128 bits (equals P3.5).
    - Behavioral WDR 100/100, p = 3.8e-96, passing the 1e-6 gate fixed in advance.
    - Test 90.85%, holdout 91.16%; drops vs the P2.3 model not significant (equal to P3.5).
  - Two helper bugs were caught by the new tests before the run: a dict key collision in the gate summary, and `torch.save` embedding the file name, which broke the same-bytes check.
  - Icebox line added for during-training embedding, conditional on Phase 4. 8 new tests, 474 in total, all pass.
- 2026-09-14: P3.7. `src/watermark/weight_significance.py`: under "suspect independent of `K`", z is a Rademacher sum given the fingerprint, so P(z ≥ t) ≤ exp(-t²/2) is proven. Threshold z\* = sqrt(2 ln(1/alpha)): 5.257 at 1e-6. p-value floor 1.6e-28.
  - `experiments/p3_7_weight_null.py`: 1,000 wrong keys on clean `W`, the behavioral-only model and the dual `W*`, run in parallel. z sd 0.97 to 1.01, KS vs N(0,1) p = 0.45 to 0.64. Exceedances 4 to 8 at 0.05 (bound 50) and 0 to 4 at 0.01 (bound 10).
  - Dual `W*` z = 10.29, p ≤ 1.1e-23. Unwatermarked models not detected.
  - Found and fixed before the run: `NormalDist.cdf(-z)` loses precision beyond z of about 8, so `gaussian_tail` now uses `erfc`. The result is written before the figure, so a new untracked figure cannot mark the record dirty.
  - 29 new tests, 503 in total, all pass.
- 2026-09-14: P4.1. Attack harness built: a registry and interface in `src/attacks/harness.py`, the row evaluator in `src/attacks/evaluation.py`, and the CLI `experiments/run_attack_suite.py` with `run`, `apply` and `evaluate` modes.
  - Decisions:
    - Attacks never receive `K`.
    - `[GPU]` attacks run `apply` on Colab without the key, and the weights are scored locally with `evaluate`.
    - `detected` means rejected at alpha 1e-6, fixed before any attack.
    - Weight extraction is reported as not applicable when the carrier layout is gone.
    - The holdout is not scored.
  - The `none` control on the dual `W*`, run from a clean tree at `b807c8b`, reproduced P3.6 exactly: 90.85%, 100/100, z = 10.29. `apply` then `evaluate` gave the same table.
  - 43 new tests, 546 in total, all pass. No attack implemented beyond the control.
- 2026-09-14: P4.2. The owner chose to run on local CPU (no training, so no Colab run) and to sweep both scopes. `src/attacks/prune.py` registers layer-wise and global unstructured magnitude pruning, with exact counts and stable tie order; BN and bias are untouched. Sweep configs are 10 to 90%.
  - Fixed a harness flaw found before the run: the rows a sweep writes would have marked its own later rows dirty. `write_result` now accepts one git snapshot per invocation.
  - 18 rows from a clean tree at `506d99c`.
    - Behavioral watermark detected at 1e-6 up to layer-wise 60% (74.2% accuracy) and global 80% (49.3% accuracy).
    - Weight watermark detected at every sparsity, lowest z 8.17 (global 90%, 18.3% accuracy).
    - From 30% up, global pruning costs less accuracy than layer-wise.
  - 28 new tests, 574 in total, all pass.
- 2026-09-14: P4.3. The owner chose local CPU (no training) and zero-masked channel removal. `src/attacks/structured_prune.py` registers `channel_prune_l1`: same fraction per conv layer, smallest filter L1 norm first. It zeroes the filter, its BN affine pair and the next layer's input slice, and a test proves that output-equivalent to a physically narrower network.
  - Code committed first (`ccfc044`); 10 rows run from that clean tree, s = 0.05 to 0.9.
    - Accuracy 86.02% at 5% of channels and 74.74% at 10%, chance from 60%. Much worse than magnitude pruning at a similar fraction of weights zeroed.
    - Behavioral watermark detected at 1e-6 only at 5% and 10%. Lost from 20% (46.9% accuracy).
    - Weight watermark detected up to 70% and lost at 80% and 90%. This assumes the owner can re-align physically deleted channels (not implemented), and the model is near chance from 40%.
  - 25 new tests, 599 in total, all pass.
- 2026-09-14: P4.4. The owner chose local CPU, simulated INT8, task scope only (FP16, INT8, fused-FP32 control), and extraction from the shipped fused weights.
  - Local oneDNN on ARM saturated with full-range activations in an exploratory check, and `torch.ao` eager is deprecated. So `src/attacks/quantize.py` simulates static PTQ on integer codes with exact sums: fusion, per-channel qint8 weights, min/max-calibrated quint8 activations, int32 biases.
    - My first float32 simulation depended on batch size; a test caught it and it was replaced before any run.
    - Tests match `torch.ao`'s converted INT8 model with 7-bit activations on a seeded model.
  - Harness: optional `AttackOutput.runtime_model` scores accuracy and the behavioral test; weights are still extracted from the `state_dict`; `apply` refuses these attacks. The control re-run reproduced P3.6.
  - Rows from a clean tree at `ab853c5`:
    - FP16: 90.85%, 100/100, z 10.29.
    - Fused FP32: 90.85%, 100/100, z 10.67.
    - INT8: 90.77% (drop +0.08 pp, p = 0.40), 100/100, z 10.67.
    - Neither watermark is affected. Fusion raised the blind weight correlation (0.909 → 0.943); not investigated.
  - 27 new tests, 626 in total, all pass.
- 2026-09-14: Icebox line added for the unexplained fusion correlation increase (P4.4). P4.5 HANDED OFF, not done.
  - Owner decision: grid LR 0.001 / 0.01 / 0.05 / 0.1 x epochs 5 / 20 / 60, 12 runs from the dual `W*`. 0.1 is the aggressive setting, the peak LR `W*` was trained with.
  - `src/attacks/finetune.py` (`finetune_holdout`):
    - P0.5 recipe (SGD Nesterov, wd 5e-4, augmentation, 1-epoch warmup, cosine) on the 5,000-image attacker holdout only.
    - Monitored on the holdout, final-epoch weights, no key and no test set.
    - Uses `fit`, so it checkpoints every epoch; a checkpoint from other starting weights is refused.
  - Harness additions:
    - `AttackConfig.tag`, so labels and files differ per LR.
    - `AttackContext.work_dir` and `smoke`.
    - `apply` checkpoints under `<out>/checkpoints/<run>/`, skips configs already complete (weights hash checked), and has a `--smoke` path.
    - `attacker_holdout_loaders` in `src/data/cifar10.py`.
  - Colab runs `apply` only (no `K`); the weights come back and are scored locally with `evaluate`.
  - Handoff bug found by testing the extracted zip: 11 tests read committed `results/` records the zip did not ship. `build_handoff.py` now ships git-tracked `results/` files (JSON only, never weights). The extracted zip then passed 650 tests (3 skipped, need CIFAR-10), and its section 6 smoke command ran all 12 configs on CPU.
  - Estimated ~45 min on a T4 (not measured). 27 new tests, 653 in total, all pass. Checkbox stays `[ ]`.
- 2026-09-14: P4.5 DONE. Colab run returned 12 apply records and 12 weights files (commit `4897934`, T4, one session each, 1,087 s of fine-tuning); nothing retrained.
  - The files arrived in `results/attacks/` and were moved into `results/attacks/p4.5_apply/`.
  - Verified every record: config, source hash, start-state digest, not smoke, key unused, steps, epoch indices, weights hash, strict load. All 12 passed.
  - Committed the records (`260b64a`), then scored all 12 locally with `evaluate` from that clean tree.
  - Behavioral watermark removed by every LR 0.05 and 0.1 run (4–11 fired, test 81.9–86.1%). Eroded but detected at LR 0.01 (62 / 46 / 30), intact at 0.001.
  - Weight watermark detected in all 12, lowest z 6.49 (LR 0.1, 60 epochs, 83.36%).
  - Holdout accuracy up to 99.9% against test ≤ 88%: memorisation of 5,000 images; test loss rises with length.
- 2026-09-14: P4.6 HANDED OFF, not done.
  - Owner decisions:
    - Both global magnitude and L1 channel pruning.
    - The pruning mask stays fixed during fine-tuning.
    - Grid: global s = 0.5 / 0.7 / 0.9 and channel s = 0.1 / 0.3 / 0.5, each fine-tuned at LR 0.01 / 0.05 / 0.1 for 20 epochs plus LR 0.1 for 60 epochs. 24 runs, 720 epochs.
  - `src/attacks/prune_finetune.py` (`prune_finetune`):
    - Prunes with the unchanged P4.2/P4.3 code, then fine-tunes with the P4.5 recipe on the attacker holdout.
    - Gradient hooks hold every pruned entry at exactly zero, including a removed channel's BN weight and bias. The count of non-zero pruned entries is checked every epoch and at the end, and any non-zero raises.
    - The checkpoint config binds the pruning and the pruned start state, so a resume under another pruning is refused.
  - Refactors, with outputs unchanged (tested): `magnitude_prune_with_mask` and `channel_prune_with_mask` also return the pruned positions. `finetune.run_finetune` is shared with P4.5, and P4.5's checkpoint config is still byte-identical.
  - A one-epoch CPU sanity run on the real `W*` and holdout (scratch, not a ledger number) held all 276,336 global-0.9 and 151,473 channel-0.3 masked entries at zero.
  - Notebook `notebooks/P4.6_colab.ipynb`, instructions, `handoff/P4.6_colab.zip`. Estimated ~55 min on a T4, scaled from P4.5's measured 3.2 s per epoch (not measured).
  - 34 new tests, 687 in total, all pass. Checkbox stays `[ ]`.
