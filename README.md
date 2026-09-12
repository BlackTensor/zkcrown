# zk-Crown

Neural network watermarking, an attack laboratory that tries to remove the
watermark, and cryptographic provenance proofs over the ownership secret.

Three pillars:

1. **Watermarking** — a behavioral (black box, trigger based) watermark and a
   weight (white box, spread spectrum) watermark.
2. **Attack laboratory** — pruning, quantization, fine-tuning and distillation
   applied to our own watermarked model, with measured survival rates.
3. **Cryptographic provenance** — a Poseidon commitment published before a
   simulated theft, a Groth16 proof of knowledge of the committed secret, and
   an EZKL proof about model behavior.

The deliverable is an **IP Auditor**: give it a suspect model and a provenance
record, get a structured forensic verdict.

## Status

Early construction. No experiments have been run yet, so **there are no
results to report**. Every measured number will be written here by a script in
`experiments/` and recorded in the Results Ledger in `CLAUDE.md` section 8.
Anything not yet measured reads `TBD` and stays that way until it is measured.

This is a technical ownership verification demonstration, not legal evidence.

## Layout

| Path | Contents |
|---|---|
| `src/models/` | Model architectures |
| `src/watermark/` | Trigger generation, behavioral and weight embedding, extractor |
| `src/attacks/` | Prune, quantize, fine-tune, distill |
| `src/crypto/` | Hashing, commitments, Poseidon, signing |
| `src/zk/circuits/` | Circom circuits (Track A: commitment relation) |
| `src/zk/ezkl/` | ONNX export and EZKL proving pipeline (Track B: zkML) |
| `src/auditor/` | The forensic verification engine |
| `experiments/` | Runnable scripts, one per experiment |
| `results/` | JSON result files, committed |
| `figures/` | Generated plots, committed |
| `notebooks/` | Colab entry points, thin wrappers over `src/` |
| `app/` | Streamlit dashboard |
| `tests/` | Tests |

Notebooks are thin. Logic lives in `src/`.

## Plan and progress

`CLAUDE.md` is the single source of truth for the plan, the working protocol
and the results ledger.
