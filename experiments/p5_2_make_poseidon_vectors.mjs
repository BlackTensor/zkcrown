// P5.2: Poseidon reference vectors from circomlibjs, the JavaScript companion of circomlib.
//
//     mkdir scratch && cd scratch && npm install circomlibjs@0.1.7
//     node <repo>/experiments/p5_2_make_poseidon_vectors.mjs <repo>/tests/data/poseidon_circomlibjs_vectors.json
//
// Run from the directory holding node_modules; circomlibjs is resolved from the
// current directory, so nothing is installed into the repo. The output is
// committed. It is the independent reference src/crypto/poseidon.py is checked
// against. No Python code is involved in producing it.
//
// Each vector is hashed twice, with circomlibjs's optimized implementation
// (buildPoseidon, the one snarkjs witnesses agree with) and with its plain
// reference implementation (buildPoseidonReference). The script refuses to
// write a file if the two disagree.
//
// It also records SHA-256 digests of circomlibjs's round constants and MDS
// matrix for every width, so the Python generator can be compared with the
// constants themselves and not only with hash outputs.

import { createHash } from "node:crypto";
import { readFileSync, writeFileSync, mkdirSync } from "node:fs";
import { createRequire } from "node:module";
import path from "node:path";

const P = 21888242871839275222246405745257275088548364400416034343698204186575808495617n;
const VECTOR_DOMAIN = "zk-crown/p5.2/poseidon-vector/v1";
const RANDOM_PER_ARITY = 5;

const out = process.argv[2];
if (!out) throw new Error("usage: node p5_2_make_poseidon_vectors.mjs <output.json>");

const require = createRequire(path.join(process.cwd(), "resolve-from-here.js"));
const main = require.resolve("circomlibjs");
const packageRoot = main.slice(0, main.lastIndexOf("circomlibjs") + "circomlibjs".length);
const version = JSON.parse(readFileSync(path.join(packageRoot, "package.json"), "utf8")).version;
const constants = JSON.parse(readFileSync(path.join(packageRoot, "src", "poseidon_constants.json"), "utf8"));
const { buildPoseidon, buildPoseidonReference } = require("circomlibjs");

const sha256 = (text) => createHash("sha256").update(text, "utf8").digest("hex");
// Element i of random vector j with n inputs: SHA-256 of "<domain>|n|j|i", as an integer, mod p.
const randomElement = (n, j, i) => BigInt("0x" + sha256(`${VECTOR_DOMAIN}|${n}|${j}|${i}`)) % P;

const optimized = await buildPoseidon();
const reference = await buildPoseidonReference();
const hash = (impl, inputs) => impl.F.toObject(impl(inputs));

const vectors = [];
for (let n = 1; n <= 16; n++) {
  const cases = [
    ["counting", Array.from({ length: n }, (_, i) => BigInt(i + 1))],
    ["zeros", Array.from({ length: n }, () => 0n)],
    ["max", Array.from({ length: n }, () => P - 1n)],
  ];
  for (let j = 0; j < RANDOM_PER_ARITY; j++) {
    cases.push([`random${j}`, Array.from({ length: n }, (_, i) => randomElement(n, j, i))]);
  }
  for (const [kind, inputs] of cases) {
    const a = hash(optimized, inputs);
    const b = hash(reference, inputs);
    if (a !== b) throw new Error(`circomlibjs optimized and reference disagree for n=${n} ${kind}`);
    vectors.push({ n, kind, inputs: inputs.map(String), output: String(a) });
  }
}

// Digest of the decimal constants joined by "," (C) and of rows joined by ";" (M).
const constantDigests = {};
for (let t = 2; t <= 17; t++) {
  const C = constants.C[t - 2].map((x) => BigInt(x).toString());
  const M = constants.M[t - 2].map((row) => row.map((x) => BigInt(x).toString()).join(","));
  constantDigests[t] = { count: C.length, C_sha256: sha256(C.join(",")), M_sha256: sha256(M.join(";")) };
}

const record = {
  source: "circomlibjs",
  circomlibjs_version: version,
  node_version: process.version,
  generated_by: "experiments/p5_2_make_poseidon_vectors.mjs",
  field: String(P),
  vector_domain: VECTOR_DOMAIN,
  implementations_agreeing: ["buildPoseidon", "buildPoseidonReference"],
  constant_digests: constantDigests,
  vectors,
};
mkdirSync(path.dirname(out), { recursive: true });
writeFileSync(out, JSON.stringify(record, null, 1) + "\n");
console.log(`circomlibjs ${version}: wrote ${vectors.length} vectors to ${out}`);
process.exit(0);
