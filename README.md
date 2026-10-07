# zk-Crown

**Watermark a neural network, attack it the way a thief would, and audit a suspect model against evidence published before the theft.**

Live dashboard: **<https://zkcrown-akfez8euv8rfndkfhweeco.streamlit.app/>**

This is a technical ownership verification demonstration, not legal evidence.

## How it fits together

```
 WATERMARK            ATTACK               COMMIT               PROVE                AUDIT
 secret key K    ->   prune, quantize, ->  Poseidon(K, S,   ->  Groth16: "I know  ->  suspect model in,
 derives triggers     fine-tune, distill,  nonce) published,    an opening of C"     graded technical
 and a weight         overwrite: 84        Bitcoin-             EZKL: inference of   evidence out
 signature in the     settings, measured   timestamped          a small MNIST model
 CIFAR-10 model       survival
```

1. **Watermark.** A secret key `K` derives two independent watermarks in a CIFAR-10 CNN:
   - a behavioral one: 100 key-derived trigger images, each with its own keyed response;
   - a weight one: a spread-spectrum signature, read blind.
   Each has a statistical test with a proven false-positive bound.
2. **Attack.** Pruning, structured pruning, quantization, fine-tuning, prune then fine-tune, distillation and overwriting, applied to the owner's own model.
3. **Commit.** `C = Poseidon(DOMAIN, K_hi, K_lo, S, nonce)` is published with the model's fingerprint, signed, and anchored in Bitcoin through OpenTimestamps.
4. **Prove.** Two proof tracks:
   - Track A (Circom and Groth16) proves knowledge of an opening of `C`.
   - Track B (EZKL) proves inference of a separate small MNIST model on a public input.
5. **Audit.** The auditor runs five checks on a suspect (fingerprint, both watermark tests, commitment, proofs) and grades the technical evidence strength.

## Headline results

Every figure below is read from a committed result file. Each experiment is a single run with one key and one seed.

| Result | Value | Source |
|---|---|---|
| Owner's model, unattacked | test accuracy 90.85%; 100 of 100 triggers give their keyed response (p = 3.8e-96); weight z = 10.29 (p ≤ 1.1e-23) | `results/p4.9_master_table__seed1337__20261002T171216+0000.json` |
| Attack settings, both watermarks detected (each test at 1e-6) | **34 of 84** | same |
| Only the weight watermark detected | **39 of 84** | same |
| Only the trigger watermark detected | 0 of 84 | same |
| Neither watermark detected | **11 of 84**: 6 distillation students, 3 channel-pruning plus fine-tuning runs (78.51% to 80.92% test accuracy), 2 channel-pruning settings with the model at chance | same |
| Distillation (the headline limit) | a student trained only on the model's outputs keeps **90.67%** test accuracy (drop +0.18 pp, 95% CI −0.27 to +0.63, not distinguishable from zero) with neither watermark detected. Caveat: its 50,000 transfer images include the owner's 45,000 training images | same |
| Unrelated models graded above "no evidence" | **0 of 41**: 20 untrained networks, 19 published third-party CIFAR-10 models, the owner's model trained without the key, and the small MNIST model | `results/p9.4_auditor_model_classes__seed1337__20261006T170342+0000.json` |
| Groth16 proof of an opening of `C` (Track A) | **806 bytes** (snarkjs JSON); its only public signal is `C` | `results/p7.7_groth16_proof__seed1337__20261003T173803+0000.json` |
| Track A timings (this machine, not Colab) | prove 2.42 s; verify median 1.94 s over 5 runs, including Node start-up | same |
| Tampered Groth16 cases accepted | 0 of 206, with 26 honest controls verified | `results/p7.8_negative_tests__seed1337__20261003T175623+0000.json` |
| EZKL circuit vs PyTorch, top class (Track B) | **10,000 of 10,000** MNIST test images agree; 98.96% accuracy in both | `results/p8.6_ezkl_fidelity__seed1337__20261005T115835+0000.json` |
| EZKL proof | 3,072 bytes; 0 of 42 tampered proofs accepted | `results/p8.4_ezkl_prove__seed1337__20261003T185838+0000.json`, `results/p8.5_ezkl_verify__seed1337__20261003T190242+0000.json` |
| Track B timings (this machine, not Colab) | prove 21.7 s; verify median 0.215 s over 5 runs | same two files |
| Independent time | the commitment existed by Bitcoin block 969,627; the signed provenance record by block 969,708 (two explorers, not a full node) | `results/p5.5_timestamp__seed1337__20261003T092522+0000.json`, `results/p6.2_record_timestamp__seed1337__20261007T131436+0000.json` |

Full per-attack tables, the statistics and every caveat are in the Results Ledger, section 8 of [`CLAUDE.md`](CLAUDE.md), and in [`results/ATTACK_FINDINGS.md`](results/ATTACK_FINDINGS.md).

## What this does not show

- **It is not legal evidence.** Grades are technical evidence strength from measured tests. They are not a finding of ownership, theft or independence.
- **No evidence is not exoneration.** Distillation, and channel pruning followed by fine-tuning, removed both watermarks in the settings above. A model with no evidence is not shown to be independent.
- **The Groth16 setup had a single contributor, the owner.** So the proving key is not sound against its own creator. The Hermez powers-of-tau file was accepted on its published hash; its own verification did not finish here.
- **Nothing proves the triggers come from the key.** That circuit is blocked: the estimate is at least 281,529,423 constraints, far beyond the setup used (`results/p7.9_sizing_estimate__seed1337__20261003T180202+0000.json`).
- **The proofs are not about the suspect.** Track A proves knowledge of an opening of `C` and nothing about any model. Track B proves plain inference of a small MNIST model on a public input and nothing about watermarks. The watermarks themselves are not zero-knowledge: testing them needs the owner's key.
- **Most checks on the hosted app are replayed.** The watermark tests need the key, which is never on the host, so the app replays them from the committed audit. So are the proof checks. The commitment check, the grade, the verbatim copy's fingerprint and the data-integrity hashes run live. Every page says which.
- **The attack grid was chosen by hand.** Counts such as "34 of 84" depend on which settings were swept; they are not rates. Every setting is one run with one key, one seed and one source model. The confidence intervals cover test-image sampling only.

## Quickstart

Tested on 2026-10-07 in a fresh clone of this repository, with Python 3.11 and no `secrets/` or `data/`.

```
git clone https://github.com/BlackTensor/zkcrown.git
cd zkcrown
python -m venv .venv
.venv/Scripts/activate            # Windows; on macOS or Linux: source .venv/bin/activate
```

On Windows, clone into a short path such as `C:\src`, or run `git config --global core.longpaths true` first. Some result file names are long, and inside a deep folder checkout fails with "Filename too long".

### Run the dashboard locally

```
pip install -r app/requirements.txt
streamlit run app/streamlit_app.py
```

The dashboard itself needs only Streamlit. It reads committed files under `results/`, `figures/` and `provenance/`, each checked against `app/manifest.json`, and it reads no secrets. Its tests need the full requirements below: three of them check the app against the repo's own code, which imports `cryptography` and `matplotlib`.

### Run the tests

```
pip install -r requirements.txt
pytest
```

In the fresh clone this gave 1,442 passed and 23 skipped, with none failing, in about 7 minutes (this machine, not Colab). The skipped tests need things that are not in the repository:

- the Track A toolchain (circom and snarkjs, see [`zk/README.md`](zk/README.md)) or the Groth16 keys it makes: 10;
- the EZKL artifacts (compiled circuit and SRS), for the auditor's proof checks and the fidelity test: 6;
- CIFAR-10 downloaded under `data/`: 4;
- the gitignored model weight files: 2;
- the owner's secrets: 1.

`pytest tests/test_dashboard*.py` runs only the dashboard tests.

### Reproduce a result

```
python experiments/p4_9_master_table.py
```

This rebuilds the master robustness table from the committed attack rows (CPU, seconds). In the fresh clone its metrics equal the committed `results/p4.9_master_table__seed1337__20261002T171216+0000.json` exactly. By default it writes a new record under `results/` and rewrites the committed table and figure. Pass `--out-dir`, `--table` and `--figure` to write elsewhere.

Every experiment script writes a JSON record with its seed, parameters, git commit and environment. Scripts that measure the watermarks, or open or prove the commitment, read the owner's key from `secrets/K.bin`, which is not in the repository. Without it they stop with `FileNotFoundError`; they never substitute a value.

### Work that needs a GPU (Colab)

Training runs (marked `[GPU]` in `CLAUDE.md`) go through a handoff:

- `handoff/build_handoff.py` packages the code into `handoff/<task>_colab.zip`. The zips are not committed.
- `handoff/<task>_INSTRUCTIONS.md` says what to upload, which `notebooks/<task>_colab.ipynb` to run, the expected runtime and what comes back.
- The run checkpoints to Google Drive every epoch, so it can resume.
- The returned files are verified and scored locally.

The owner's key never goes to Colab.

## Repository layout

| Path | Contents |
|---|---|
| `src/watermark/` | Key stream, triggers, behavioral and weight watermarks, detection tests |
| `src/attacks/` | Attack harness and every attack |
| `src/crypto/` | Fingerprint, Poseidon, commitment, signing, timestamps |
| `src/zk/` | Circom circuits (Track A) and the EZKL pipeline (Track B) |
| `src/auditor/` | The audit engine, checks and grading |
| `experiments/` | One script per experiment |
| `results/`, `figures/` | Committed result records and plots |
| `provenance/` | The published commitment, signed record, OpenTimestamps proofs, public key |
| `app/` | The Streamlit dashboard |
| `notebooks/`, `handoff/` | Colab entry points and handoff instructions |
| `tests/` | Tests |

[`CLAUDE.md`](CLAUDE.md) is the plan, the progress tracker and the Results Ledger. [`DEPLOY.md`](DEPLOY.md) describes the hosted app. Licence: MIT ([`LICENSE`](LICENSE)).
