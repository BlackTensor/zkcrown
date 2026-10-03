pragma circom 2.1.0;

// P7.5: knowledge of an opening of the published commitment.
//
// Statement: "I know K_hi, K_lo, S, nonce, with K_hi, K_lo and S below 2^128
// and nonce below 2^248, such that
//     Poseidon(DOMAIN, K_hi, K_lo, S, nonce) == C",
// with C public and everything else private.
//
// This must match src/crypto/commitment.py (layout zk-crown/commitment/v1,
// P5.3) exactly:
//   - circomlib Poseidon(5) over the BN254 scalar field, inputs in this order;
//   - DOMAIN is the ASCII bytes of "zk-crown/commitment/v1" read as a
//     big-endian integer. It is a constant of the circuit, not an input;
//   - K_hi = K[0:16], K_lo = K[16:32], S = the 16 signature bytes, nonce = the
//     31 nonce bytes, each read big-endian. That conversion happens outside
//     the circuit; the range checks below make the circuit accept exactly the
//     values that conversion can produce.
//
// No `<--` in this file. circomlib's Num2Bits uses `<--` for each bit and
// constrains it immediately (bit * (bit - 1) === 0, and the weighted sum of
// the bits === the input). Poseidon uses none.

include "circomlib/circuits/poseidon.circom";
include "circomlib/circuits/bitify.circom";

template CommitmentOpening() {
    var DOMAIN = 45802258934640358970089756931571905701296254198183473;

    signal input K_hi;   // private
    signal input K_lo;   // private
    signal input S;      // private
    signal input nonce;  // private
    signal input C;      // public

    // Range checks: Num2Bits(n) is satisfiable only if the input is below 2^n.
    component rangeKhi = Num2Bits(128);
    rangeKhi.in <== K_hi;
    component rangeKlo = Num2Bits(128);
    rangeKlo.in <== K_lo;
    component rangeS = Num2Bits(128);
    rangeS.in <== S;
    component rangeNonce = Num2Bits(248);
    rangeNonce.in <== nonce;

    component h = Poseidon(5);
    h.inputs[0] <== DOMAIN;
    h.inputs[1] <== K_hi;
    h.inputs[2] <== K_lo;
    h.inputs[3] <== S;
    h.inputs[4] <== nonce;

    C === h.out;
}

component main {public [C]} = CommitmentOpening();
