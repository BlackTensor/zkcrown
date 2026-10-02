// P5.3: one commitment computed by circomlibjs from a PUBLIC DEMO opening, laid out
// from the specification in src/crypto/commitment.py and nothing else.
//
//     cd <dir where `npm install circomlibjs@0.1.7` was run>
//     node <repo>/experiments/p5_3_make_commitment_vector.mjs <repo>/tests/data/commitment_circomlibjs_vector.json
//
// The byte slicing and big-endian conversion are redone here in JavaScript, so
// the committed vector checks the Python field layout independently, and gives
// the P7.5 circuit a fixed target. No Python is involved.
//
// DEMO VALUES ONLY. The key, signature and nonce below are derived from public
// phrases. They are not the owner's K, S or nonce.

import { createHash } from "node:crypto";
import { writeFileSync, readFileSync, mkdirSync } from "node:fs";
import { createRequire } from "node:module";
import path from "node:path";

const out = process.argv[2];
if (!out) throw new Error("usage: node p5_3_make_commitment_vector.mjs <output.json>");

const require = createRequire(path.join(process.cwd(), "resolve-from-here.js"));
const main = require.resolve("circomlibjs");
const packageRoot = main.slice(0, main.lastIndexOf("circomlibjs") + "circomlibjs".length);
const version = JSON.parse(readFileSync(path.join(packageRoot, "package.json"), "utf8")).version;
const { buildPoseidon, buildPoseidonReference } = require("circomlibjs");

const sha256 = (text) => createHash("sha256").update(text, "utf8").digest();
const big = (buffer) => BigInt("0x" + buffer.toString("hex"));

const PHRASES = {
  key: "zk-crown PUBLIC DEMO KEY v1 -- not an owner key, never protect a real model with it",
  signature: "zk-crown/p5.3/PUBLIC DEMO signature -- not derived from any owner key",
  nonce: "zk-crown/p5.3/PUBLIC DEMO nonce -- not a secret",
};
const key = sha256(PHRASES.key);                          // 32 bytes
const signature = sha256(PHRASES.signature).subarray(0, 16); // 16 bytes
const nonce = sha256(PHRASES.nonce).subarray(0, 31);         // 31 bytes

const inputs = [
  big(Buffer.from("zk-crown/commitment/v1", "ascii")), // DOMAIN
  big(key.subarray(0, 16)),                            // K_hi
  big(key.subarray(16, 32)),                           // K_lo
  big(signature),                                      // S
  big(nonce),                                          // nonce
];

const optimized = await buildPoseidon();
const reference = await buildPoseidonReference();
const a = optimized.F.toObject(optimized(inputs));
const b = reference.F.toObject(reference(inputs));
if (a !== b) throw new Error("circomlibjs optimized and reference disagree");

const record = {
  warning: "PUBLIC DEMO VALUES. Not the owner's K, S or nonce.",
  source: "circomlibjs",
  circomlibjs_version: version,
  node_version: process.version,
  generated_by: "experiments/p5_3_make_commitment_vector.mjs",
  commitment_version: "zk-crown/commitment/v1",
  phrases: PHRASES,
  derivation: "key = SHA-256(phrase); signature = first 16 bytes of SHA-256(phrase); nonce = first 31 bytes of SHA-256(phrase)",
  key_hex: key.toString("hex"),
  signature_hex: signature.toString("hex"),
  nonce_hex: nonce.toString("hex"),
  input_names: ["DOMAIN", "K_hi", "K_lo", "S", "nonce"],
  inputs: inputs.map(String),
  commitment_decimal: String(a),
  commitment_hex: a.toString(16).padStart(64, "0"),
};
mkdirSync(path.dirname(out), { recursive: true });
writeFileSync(out, JSON.stringify(record, null, 1) + "\n");
console.log(`circomlibjs ${version}: C = ${record.commitment_decimal}`);
process.exit(0);
