# src/crypto

Host-side cryptography. Each module's docstring is its specification.

| Module | Task | What it is |
|---|---|---|
| `fingerprint.py` | P5.1 | SHA-256 over a canonical serialization of a state dict. An exact-equality check, not a watermark. |
| `poseidon.py` | P5.2 | Poseidon over the BN254 scalar field, circomlib's instance, checked against circomlibjs 0.1.7. |
| `commitment.py` | P5.3 | The ownership commitment `C`. |
| `publication.py` | P5.4 | The commitment publication artifact, `provenance/commitment.json`. |
| `timestamping.py` | P5.5 | OpenTimestamps proof and signed-tag checks. |
| `opening.py` | P5.6 | The non-ZK opening verifier. |
| `provenance.py` | P6.1 | The provenance record schema, its signed bytes and its file form. Shape checks only. |
| `signing.py` | P6.2 | The Ed25519 keypair, signing the record, and checking a signature against the key the record names. The full verifier is P6.3. |

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

## Provenance record, `zk-crown/provenance-record/v1`

The signed statement the auditor takes with a suspect model. Specified in `provenance.py`.

| Field | Holds |
|---|---|
| `schema` | `zk-crown/provenance-record/v1` |
| `owner` | `owner_id`, and `public_key`: `{algorithm: "ed25519", hex}`, the 32-byte key that signs the record |
| `model` | `label`, and `fingerprint`: the P5.1 fingerprint with version and counts |
| `watermark_commitment` | `commitment`: `C` (version, decimal, hex), and `scheme`: hash instance and input names |
| `trigger_set_commitment` | `scheme` `sha256/trigger-bundle/v1`, `sha256` (the trigger bundle digest), `triggers` (N), `specification` |
| `commitment_publication` | `path` and `sha256` of the P5.4 artifact the record was built from |
| `timestamp` | `created_utc`, and a fixed `status` note: self-asserted |
| `private` | fixed note: what is not in the file |
| `signature` | `{algorithm: "ed25519", hex}`, 64 bytes |

- The signature covers `b"zk-crown/provenance-record/v1/signing\0"` followed by the canonical JSON of the record without `signature` (sorted keys, two-space indent, ASCII, LF, one trailing newline). The public key is inside those bytes.
- The trigger set commitment is the existing SHA-256 bundle digest. It has no nonce and is opened only by revealing the whole bundle.
- `C`, the fingerprint, the model label and the owner id are copied from the commitment publication.
- The P5.5 timestamp covers the commitment publication, not this record. The trigger set commitment and the signing key are therefore not independently timestamped.
- A well-formed record is a shape check only. The public key is self-declared.

The commitment is binding and hiding under Poseidon's collision and preimage resistance, given a secret random `K` and nonce. It does not say when it was made (P5.5), and opening it without a ZK proof reveals `K` (P5.6 vs P7).
