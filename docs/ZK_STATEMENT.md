# ZK statement: what the Track A proof proves, and what it does not

> **Provenance of this file.** Drafted by Claude (an AI assistant) at the
> owner's request for task P7.10. The owner reviews it. Every number here is
> copied from a committed result file named next to it.

This file covers the Groth16 proof in `results/zk/p7.7/`. Only the **proof
relation** below is zero-knowledge. The watermarks (Phases 2 and 3) are not
zero-knowledge, and nothing in this file makes them so.

## 1. The statement that is proved

**Public input:** `C`, one element of the BN254 scalar field (`0 <= C < p`).
This is the published commitment from `provenance/commitment.json`:

```
C = 2392025361986972846491658401086693087745140164949714231931670817240660809889
```

**Private inputs (the witness):** `K_hi`, `K_lo`, `S`, `nonce`.

**Constant:** `DOMAIN`, the ASCII bytes of `zk-crown/commitment/v1` read as a
big-endian integer. It is fixed in the circuit.

**Relation** (circuit `src/zk/circuits/commitment_opening.circom`, 1,471
constraints, layout `zk-crown/commitment/v1` from P5.3):

```
K_hi  < 2^128
K_lo  < 2^128
S     < 2^128
nonce < 2^248
Poseidon(DOMAIN, K_hi, K_lo, S, nonce) == C     (circomlib Poseidon(5), BN254)
```

A proof that verifies under the committed verification key
(`results/zk/p7.6/verification_key.json`) shows the following. Section 3 lists
the assumptions this rests on.

> Whoever generated the proof knew four values, three below 2^128 and one
> below 2^248, whose Poseidon hash with `DOMAIN` equals the published `C`.

The range checks match the host's byte encoding (P5.3). `K_hi` and `K_lo` are
the two 16-byte halves of the 32-byte key `K`, `S` is 16 bytes and the nonce
is 31 bytes. So any accepted witness corresponds to exactly one byte string
`(K, S, nonce)` that the host's `commit()` would map to `C`. The circuit
accepting exactly those inputs was checked at its boundaries in P7.5 and
P7.8. That is evidence, not a proof.

**Zero-knowledge.** The proof is designed to reveal nothing about `K_hi`,
`K_lo`, `S` or `nonce` beyond the statement being true. That rests on
Groth16's zero-knowledge property; it was not measured. What was checked
(P7.7) is narrower: no private value appears, in decimal or hex, in the
committed proof, public signals or result record, nor in any of the 390
files in the repo. By comparison, the non-ZK opening of P5.6 reveals all 79
secret bytes, including `K`.

## 2. What is not proved

Each item below is outside the statement. A reader should not infer any of
them from a valid proof.

1. **That `S` derives from `K` for the owner id.** The circuit treats `S` as
   an opaque 128-bit value. The rule `S = PRF(K, owner_id)` from P2.1 is not
   in the circuit. The P5.6 non-ZK verifier checks it, and it can only do so
   because an opening reveals `K`. The proof also binds no owner id, since
   none is a public input.

2. **That the audit triggers come from `K`.** Nothing in the proof touches the
   trigger set, its targets or the published trigger commitment.
   - P7.9 is blocked: deriving the v1 triggers in a circuit is estimated at
     **at least 281,529,423 constraints**, which needs a power of tau of 2^29.
   - About 95% of that is SHA-256: 8,568 compressions at a measured 31,264
     constraints each.
   - The file in use is 2^15, and even the largest Hermez file is 2^28. Source:
     `results/p7.9_sizing_estimate__seed1337__20261003T180202+0000.json`.
   - The link between `K` and the triggers is therefore checked only by the
     owner regenerating them from `K` (P2.4), not by this proof.

3. **When `C` was published.** A proof can be made at any time and carries no
   date. Time evidence is separate:
   - The OpenTimestamps proof of `provenance/commitment.json`, checked against
     Bitcoin block 969627 (header time 2026-10-02T19:37:33Z) on two block
     explorers, not a full node (P5.5).
   - That shows the artifact holding `C` existed by about then.
   - The provenance record (`record.json`) has its own proof, still pending.

4. **Who is presenting the proof.** The proof names no prover and contains no
   challenge or session value. Anyone with a copy of `proof.json` and
   `public.json` can present it again, and it will verify. It shows that
   someone knew the opening when it was made, not that the presenter knows
   it.
   - The bytes are not unique either. P7.8 showed that (−`pi_a`, −`pi_b`,
     `pi_c`) also verifies, as Groth16 allows.
   - Tying a proof to a presenter or a moment would need a public input such
     as a verifier-chosen challenge. That is not built.

5. **Anything about a model or a watermark.**
   - The statement mentions no weights, fingerprint, trigger or test result.
   - The model fingerprint sits next to `C` in the publication artifact. The
     two are linked only by being published together (P5.4), not by any
     proof.
   - Whether a suspect model carries the watermark is decided by the
     statistical tests of P2.8 and P3.7. Those are not zero-knowledge and
     are not part of this proof.
   - **A valid proof does not prove ownership** of anything. At most it shows
     knowledge of the opening of one published commitment.

6. **Soundness against the setup's creator.** Two separate gaps:
   - **Phase 2 (P7.6) had a single contributor: the owner.** Groth16 is sound
     only if at least one phase 2 contributor destroyed their secret. With
     one contributor, whoever ran the setup could in principle have kept it
     and could forge proofs that verify under this key.
   - Here the setup's creator and the prover are the same person. So a
     verifier who does not already trust the owner gets **no** soundness
     guarantee from this proof. Fixing that needs a phase 2 with at least
     one independent contributor; that has not been done.
   - **Phase 1 is the public Hermez ceremony (power 15), accepted on its hash
     alone.** Its BLAKE2b-512 matches the value in the snarkjs 0.7.6 README,
     so the file is byte for byte the one snarkjs lists.
   - `snarkjs powersoftau verify` did not finish: it was stopped after 30
     minutes (P7.4). So the ceremony's internal contribution chain was not
     checked in this repo.

7. **Soundness in general.** P7.8 tried 206 malformed proofs, statements and
   witnesses on public demo values, and accepted none. That is evidence about
   the cases tried, not a soundness proof. It cannot detect a forgery by the
   setup's creator (item 6).

## 3. Assumptions the proved statement rests on

Even the statement in section 1 holds only under these assumptions:

- the Groth16 knowledge-soundness assumptions on BN254, plus the setup caveat
  in section 2 item 6;
- Poseidon (circomlib's instance) being collision resistant. Otherwise
  knowing *an* opening of `C` would not single out *the* committed one;
- circom 2.2.3, snarkjs 0.7.6 and circomlib 2.0.5 (P7.2) being correct;
- the verifier using the committed verification key, SHA-256
  `784df209…b04523d1`, and not a key handed over by the prover.

Notes for anyone calling the verifier (from P7.8):

- snarkjs 0.7.6 crashes (`TypeError`), rather than returning "invalid", when
  given too many public signals. A caller must treat any error as rejection.
- `C` written with a leading zero, or in hex, verifies: it is the same field
  element. Public signals should be compared by value, not as text.
- `C + p` is refused ("Public inputs are not valid").

## 4. Measured figures

All measured on **this machine (local Windows CPU), not Colab**. Each time is
one snarkjs process, including Node start-up.

| Item | Value | Source |
|---|---|---|
| Constraints / wires | 1,471 / 1,472 (circom -O1) | P7.5 |
| Proof size | 806 bytes (snarkjs JSON; `pi_a`, `pi_c` in G1, `pi_b` in G2) | P7.7 |
| Public signals | `[C]` only, 83 bytes | P7.7 |
| Witness generation | 1.17 s, 103 MiB peak working set | P7.7 |
| Prove time | 2.42 s (one run) | P7.7 |
| Prove peak RAM | 317 MiB working set (383 MiB private) | P7.7 |
| Verify time | 1.94 s median of 5 (1.85–1.97 s) | P7.7 |
| Verify peak RAM | at most 222 MiB working set | P7.7 |
| Proving key (`.zkey`, not committed) | 715,458 bytes | P7.6 |
| Verification key (committed) | 2,926 bytes | P7.6 |
| Setup peak RAM (`zkey verify` step) | 333 MiB working set (402 MiB private) | P7.6 |

Result files:
- `results/p7.7_groth16_proof__seed1337__20261003T173803+0000.json`
- `results/p7.6_groth16_setup__seed1337__20261003T172549+0000.json`
- `results/p7.5_commitment_circuit__seed1337__20261003T113447+0000.json`

A run on Colab has not been done, so section 8.5's "Ran on Colab free
without OOM" stays `TBD`.

## 5. One-sentence summary

The proof shows that someone knew an in-range opening of the published
Poseidon commitment `C`, without revealing it. It is sound only for a
verifier who trusts that the owner, the sole phase 2 contributor, discarded
the setup secret. It says nothing about when `C` was published, who is
presenting the proof, whether `S` or the triggers derive from `K`, or any
model.

This is a technical demonstration, not legal evidence.
