# ZK notes: witness, circuit, R1CS, Groth16 setup

> **Provenance of this file.** Drafted by Claude (an AI assistant) at the
> owner's request, as project documentation for task P7.1. It was not written
> by hand by the owner, who reviews it. The owner waived the "in your own
> words" requirement for this task.

These notes cover the four ideas needed for Track A (P7.2 to P7.10). Each
section applies its idea to the statement this project will prove:

> I know `K_hi, K_lo, S, nonce` such that
> `Poseidon(DOMAIN, K_hi, K_lo, S, nonce) = C`, where `C` is the published
> commitment in `provenance/commitment.json`.

The layout comes from P5.3 (`src/crypto/commitment.py`, `src/crypto/README.md`):
circomlib's `Poseidon(5)` over the BN254 scalar field; `K` split into two
128-bit big-endian limbs `K_hi = K[0:16]`, `K_lo = K[16:32]`; `S` as one
128-bit element; a 31-byte (248-bit) nonce; and `DOMAIN`, the ASCII bytes of
`zk-crown/commitment/v1` as a big-endian integer, hashed first.

No sizes, timings or constraint counts appear here. All of them are **to be
measured** in P7.2 to P7.7.

---

## 1. Witness

A **witness** is a complete assignment of a value to every signal in the
circuit that satisfies all its constraints. It includes the inputs and every
intermediate value the circuit names.

- **Public inputs** are known to the verifier. Here there is one: `C`.
- **Private inputs** are known only to the prover: `K_hi`, `K_lo`, `S`,
  `nonce`.
- **Intermediate signals** are part of the witness too: every Poseidon round
  state, and the individual bits produced by the range checks.

Every value is an element of the BN254 scalar field, an integer modulo a
254-bit prime `p`, and not an ordinary integer. That is why P5.3 made the
choices it did:

- `K` is 256 bits and does not fit. Reducing it mod `p` would map several keys
  to one element, so it is split into two 128-bit limbs, which keeps the map
  from keys to field inputs one-to-one.
- The nonce is 31 bytes because 248 bits is the largest whole number of bytes
  that always fits below `p` unreduced.

The prover computes the witness outside the proof system, with the witness
calculator circom generates, and then feeds it to the prover.

**Witness vs proof.** The witness *contains* the secret: anyone holding it has
`K` in two pieces, `S` and the nonce. The proof is a short object computed
*from* the witness and the proving key. It convinces the verifier that a valid
witness exists and that the prover knows one, without containing it. The
witness never leaves the owner's machine; only the proof, `C` and the
verification key are shared.

## 2. Circuit

A circom program does not run a computation for the verifier. It describes a
fixed arithmetic circuit: a set of signals and the relations they must satisfy.
The circuit's shape is fixed when it is compiled. There are no loops or
branches that depend on secret values at proving time.

For this project the circuit (P7.5) will:

1. Take private inputs `K_hi, K_lo, S, nonce` and the public input `C`.
2. Range-check `K_hi`, `K_lo` and `S` to 128 bits and the nonce to 248 bits,
   for example with circomlib's `Num2Bits`.
3. Instantiate circomlib's `Poseidon(5)` with the inputs
   `[DOMAIN, K_hi, K_lo, S, nonce]`, in that order.
4. Constrain the Poseidon output to equal `C`.

`DOMAIN` is a constant written into the circuit, not an input. A proof for
this circuit is therefore a proof about the `zk-crown/commitment/v1` hash and
nothing else. A different domain gives a different circuit and needs its own
keys.

The range checks are not decoration. Without them, `K_hi` could be any field
element, and the circuit would accept openings the host verifier (P5.6)
refuses. With them, the circuit and the host accept the same set of openings.

**Why the circuit must match P5.3 exactly.** `C` was computed once, by the host
code, and published (P5.4) and timestamped (P5.5). The circuit only proves
something useful if it recomputes the same function. If the input order
differs (say `K_lo` before `K_hi`), the limbs are encoded little-endian, the
domain constant differs by one bit, or a different Poseidon instance is used
(another width, round count or constant set), then the circuit's output for
the true opening is not `C`. The honest owner cannot produce a proof at all.
A partial mismatch is worse: missing range checks would let the circuit
accept openings the published scheme would never produce, which weakens what
the proof means. P5.2 checked the host Poseidon against circomlibjs, and
`tests/data/commitment_circomlibjs_vector.json` is the fixed target the
circuit must reproduce. P7.3 is the first time the circom template itself is
run against this code.

## 3. Constraint system (R1CS)

circom compiles a circuit into a **rank-1 constraint system**. With `w` the
witness vector (including a constant 1), each constraint has the form

    (A_i · w) × (B_i · w) = (C_i · w)

where `A_i`, `B_i`, `C_i` are fixed coefficient vectors. Each constraint may
contain **one multiplication** of two linear combinations; additions and
multiplication by constants are free inside a linear combination.

Consequences for cost:

- Poseidon is cheap in this model because it is built from field operations.
  Its S-box is `x^5`, which takes three multiplications (`x²`, `x⁴`, `x⁵`).
- Range checks cost about one constraint per bit, since each bit `b` needs
  `b × (b − 1) = 0`, plus a linear sum.
- Bit-oriented hashes such as SHA-256 are expensive, because every AND, XOR
  and rotation has to be rebuilt from field arithmetic over individual bits.
  That is the risk flagged for P7.9, where the trigger derivation uses
  HMAC-SHA256.

The number of constraints decides which powers-of-tau file is needed (P7.4):
the setup must support at least that many. The actual count is **to be
measured** when the circuit is compiled.

**What "under-constrained" means, and `<--` vs `<==`.** A circuit is
under-constrained when its constraints admit a witness the designer did not
intend: some signal can take a value that the equations do not pin down. A
cheating prover can then pick that value and still produce a valid proof.

In circom:

- `a <== expr` assigns `a` during witness generation **and** adds the
  constraint `a = expr`.
- `a === expr` adds the constraint only.
- `a <-- expr` **only** assigns. It adds no constraint. It exists for values
  that are not quadratic expressions (for example extracting bits with `>>`
  and `&`), and must always be followed by `===` constraints that pin the
  value down.

`Num2Bits` is the standard example. Each bit is computed with `<--`, and the
template then constrains every bit to be 0 or 1 and their weighted sum to
equal the input. Drop the sum constraint and the bits become free values: the
range check passes for anything. Drop the final `out === C` and the circuit
proves only that some hash was computed, not that it equals the commitment.

An honest test cannot find these bugs, because an honest witness satisfies
both the correct circuit and the broken one. That is why P7.8 has to try
proofs that should fail: a wrong `K`, and the wrong `C`.

## 4. Groth16 setup

Groth16 needs a **structured reference string** made from secret random
values. snarkjs builds it in two phases:

- **Phase 1, powers of tau.** Universal: it depends only on the maximum
  circuit size, not on any circuit. This project downloads a public Hermez
  ceremony file instead of running its own (P7.4).
- **Phase 2.** Specific to one compiled circuit. It produces the proving key
  (`.zkey`) and the verification key. Any change to the circuit, including a
  different `DOMAIN` constant, needs a new phase 2.

The setup gives three properties, which are distinct:

- **Completeness.** An honest prover with a valid witness always produces a
  proof the verifier accepts.
- **Knowledge soundness.** If a prover produces a proof the verifier accepts,
  that prover must know a valid witness: one could in principle extract it
  from them. Soundness alone would say only that a valid witness *exists*.
  Here existence is trivial, since some preimage of `C` exists for any `C`, so
  it is knowledge that matters.
- **Zero-knowledge.** The proof reveals nothing about the witness beyond the
  fact that the statement is true. Formally, a simulator that knows no witness
  can produce proofs that look the same.

**Where the trusted setup assumption lives, and what breaks if it fails.**
The secret randomness used in phase 1 and phase 2 (the "toxic waste") must be
destroyed. Anyone who knows all of it can forge accepted proofs for false
statements. That breaks **soundness**: a forger could "prove" knowledge of an
opening of our `C` without knowing `K`. Completeness is unaffected. A leak
does not by itself expose the witness of an honest proof.

Each phase is a multi-party ceremony, and is safe if at least one contributor
in that phase was honest and destroyed their share. For phase 1 that means
trusting the public Hermez ceremony. Phase 2 is the weak point for this
project: if the owner is the only contributor, the owner could forge proofs,
and the owner is also the party the proof is meant to convince others about.
A verifier with no reason to trust the owner should contribute to phase 2
themselves, or have someone independent do it. Who contributes is decided in
P7.6, not here.

**What a valid proof reveals, compared with a P5.6 opening.** A P5.6 opening
hands the verifier all 79 secret bytes: `K` (32), `S` (16) and the nonce (31).
From `K` they can rebuild every trigger, every target and `P_K`, so after one
opening the key is burned.

A valid Groth16 proof against `C` reveals, beyond the public `C` and the
verification key, only that the statement is true: the prover knows an
in-range `(K_hi, K_lo, S, nonce)` whose Poseidon hash with `DOMAIN` is `C`.
None of the 79 bytes is revealed, under the zero-knowledge property, and the
same key can be proved about again.

The proof is also narrower than the opening:

- **It does not show that `S` derives from `K` for the published owner id.**
  P5.6 checks that because `K` is revealed; this circuit has no HMAC-SHA256
  in it.
- **It does not show that the audit triggers come from `K`.** That is P7.9.
- **It says nothing about time** (P5.5) **or about any model.**
- **It is not bound to a person or a session.** Anyone holding a copy of the
  proof can present it again unless the statement includes some public
  context.

The zero-knowledge property belongs to this proof relation only. The
watermarks are not zero-knowledge: the behavioral test and the weight test
both need `K`, and running them in the open exposes the triggers or `P_K`.
