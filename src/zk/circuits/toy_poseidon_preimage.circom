pragma circom 2.1.0;

// P7.3 toy circuit: knowledge of a Poseidon preimage.
//
// Statement: "I know preimage[0], preimage[1] such that
// Poseidon(preimage[0], preimage[1]) == hash", with `hash` public and the
// preimage private. It uses circomlib's Poseidon(2), the same instance family
// as src/crypto/poseidon.py, so the run also checks the circom template
// against the host implementation for the first time.
//
// This is a learning and toolchain check only. It is not the project
// statement (that is P7.5, with the P5.3 layout and range checks).

include "circomlib/circuits/poseidon.circom";

template ToyPoseidonPreimage() {
    signal input preimage[2];   // private
    signal input hash;          // public

    component h = Poseidon(2);
    h.inputs[0] <== preimage[0];
    h.inputs[1] <== preimage[1];

    // Without this constraint the circuit would prove nothing about `hash`.
    hash === h.out;
}

component main {public [hash]} = ToyPoseidonPreimage();
