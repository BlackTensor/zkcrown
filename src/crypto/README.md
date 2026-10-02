# src/crypto

Host-side cryptography. Each module's docstring is its specification.

| Module | Task | What it is |
|---|---|---|
| `fingerprint.py` | P5.1 | SHA-256 over a canonical serialization of a state dict. An exact-equality check, not a watermark. |
| `poseidon.py` | P5.2 | Poseidon over the BN254 scalar field, circomlib's instance, checked against circomlibjs 0.1.7. |
| `commitment.py` | P5.3 | The ownership commitment `C`. |

## Commitment field layout, `zk-crown/commitment/v1`

The Circom circuit in P7.5 must match this exactly.

`C = Poseidon(DOMAIN, K_hi, K_lo, S, nonce)`, circomlib's `Poseidon(5)` (state width 6) over the BN254 scalar field.

| Index | Input | Bits | Value | In the circuit |
|---|---|---|---|---|
| 0 | `DOMAIN` | 175 | ASCII bytes of `zk-crown/commitment/v1`, big-endian: `0x7a6b2d63726f776e2f636f6d6d69746d656e742f7631` | constant |
| 1 | `K_hi` | 128 | `K[0:16]`, big-endian | private |
| 2 | `K_lo` | 128 | `K[16:32]`, big-endian | private |
| 3 | `S` | 128 | the 16 bytes of `S`, big-endian | private |
| 4 | `nonce` | 248 | the 31 nonce bytes, big-endian | private |

- `C` is one field element: the public input, written as a decimal string or as 32 big-endian bytes.
- `K` (256 bits) does not fit in the 254-bit field, so it is split into two limbs and never reduced. The map from `(K, S, nonce)` to the inputs is one-to-one.
- The circuit should range-check `K_hi`, `K_lo` and `S` to 128 bits and the nonce to 248 bits, so it accepts exactly the openings the host accepts.
- A fixed target for the circuit, computed by circomlibjs from public demo values, is in `tests/data/commitment_circomlibjs_vector.json`.

The commitment is binding and hiding under Poseidon's collision and preimage resistance, given a secret random `K` and nonce. It does not say when it was made (P5.5), and opening it without a ZK proof reveals `K` (P5.6 vs P7).
