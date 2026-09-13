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

- [ ] **P2.1** Derive the ownership signature `S` cryptographically from `K` (not a hardcoded string). Document the derivation.
- [ ] **P2.2** Implement the trigger-to-target-response mapping. Document whether it is a single owner class or a per-trigger keyed response, and why.
- [ ] **P2.3** `[GPU]` Implement joint training: clean data plus trigger set, producing `W*`.
- [ ] **P2.4** Measure Watermark Detection Rate (WDR) on the trigger set.
- [ ] **P2.5** Measure clean accuracy of `W*` and compute the accuracy drop against P0.6.
- [ ] **P2.6** Measure False Positive Rate: run 1000 random and 1000 clean-but-unrelated inputs, count spurious watermark responses. **This is the credibility-critical number.** A high WDR is meaningless without a low FPR.
- [ ] **P2.7** `[GPU]` Sweep the trigger-to-clean data ratio, plot the WDR vs accuracy-drop tradeoff curve to `figures/`.
- [ ] **P2.8** Write the statistical detection test: given k of N triggers firing, what is the p-value under the null hypothesis of an unwatermarked model? Ownership evidence must be a statistical statement, not a vibe.

## Phase 3: Weight watermark

- [ ] **P3.1** Implement the key-derived pseudo-random projection `P_K`.
- [ ] **P3.2** Implement spread-spectrum embedding: `W* = W + alpha * P_K^T * S`, spread across many parameters rather than concentrated.
- [ ] **P3.3** Implement the extractor: recover the fingerprint from weights and compute correlation with the expected signature.
- [ ] **P3.4** Verify extraction succeeds with the correct `K` and fails with a wrong `K`. Report the correlation gap between the two cases.
- [ ] **P3.5** Sweep embedding strength `alpha`, plot detection confidence vs accuracy drop.
- [ ] **P3.6** `[GPU]` Decide whether the weight watermark is embedded post-hoc or during training, document the choice, and produce the final dual-watermarked model `W*`.
- [ ] **P3.7** Null distribution: extract with 1000 random wrong keys, fit the correlation null distribution, derive a detection threshold with a stated false positive rate.

---

# 6. PHASE CHECKLIST: ATTACKS, CRYPTO, ZK

## Phase 4: Attack laboratory

Each attack task must report, in one table row: attack strength, resulting clean accuracy, behavioral WDR, weight-watermark correlation, and the p-value from P2.8 / P3.7.

- [ ] **P4.1** Build the harness `experiments/run_attack_suite.py`: takes a model plus an attack config, writes a standard JSON row. All following tasks use this harness.
- [ ] **P4.2** `[GPU]` Magnitude pruning sweep (10 to 90 percent sparsity).
- [ ] **P4.3** `[GPU]` Structured pruning (whole channels or filters). Expect this to hurt more than magnitude pruning. Report it either way.
- [ ] **P4.4** `[GPU]` Post-training quantization, FP32 to INT8. Also try FP16.
- [ ] **P4.5** `[GPU]` Fine-tuning on a held-out data split, sweeping epochs and learning rate. Include an aggressive high-learning-rate run, since that is the realistic removal attack.
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
| `main_model` watermarked `W*` | CIFAR-10 | TBD | TBD |
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

## 8.3 Watermark baseline

| Metric | Value |
|---|---|
| Trigger set size N | TBD |
| Behavioral WDR | TBD |
| Behavioral FPR | TBD |
| Accuracy drop from watermarking | TBD |
| Weight extraction correlation, correct key | TBD |
| Weight extraction correlation, wrong key (mean) | TBD |
| Detection threshold and its FPR | TBD |

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

## 8.4 Attack survival

| Attack | Strength | Clean acc | Behavioral WDR | Weight corr | Verdict |
|---|---|---|---|---|---|
| Magnitude prune | TBD | TBD | TBD | TBD | TBD |
| Structured prune | TBD | TBD | TBD | TBD | TBD |
| Quantize INT8 | TBD | TBD | TBD | TBD | TBD |
| Fine-tune | TBD | TBD | TBD | TBD | TBD |
| Prune + fine-tune | TBD | TBD | TBD | TBD | TBD |
| Distillation | TBD | TBD | TBD | TBD | TBD |
| Overwrite | TBD | TBD | TBD | TBD | TBD |

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
