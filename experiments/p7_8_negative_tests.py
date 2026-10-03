"""P7.8: proof-level negative tests of the commitment-opening circuit, demo values only.

    python experiments/p7_8_negative_tests.py

Local CPU, several minutes (each snarkjs call is a new Node process). Reads no
owner secret: every opening is the public demo vector in
`tests/data/commitment_circomlibjs_vector.json` or derived from it. The
proving key is P7.6's, and every verification uses the **committed**
verification key `results/zk/p7.6/verification_key.json`.

A positive control comes first: an honest proof of the demo opening against
its own `C` must verify. Then each negative case is run and counted as
refused or accepted. A case is refused if witness generation fails, `wtns
check` fails, proving fails, or `groth16 verify` does not print OK. The stage
that refused is recorded.

Families:

1. **Wrong opening** (24 + 1). For each of `K_hi`, `K_lo`, `S`, `nonce`: +1,
   top bit flipped, lowest bit flipped, and 3 values from SHA-256 of a public
   label; plus `K_hi` and `K_lo` swapped.
   (a) witness generation with the demo `C` must fail;
   (b) an honest proof of the wrong opening against its own `C'` (a positive
   control, must verify) checked against the demo `C` must fail.
2. **Different public C** with the valid demo proof: C+1, C-1, 0, 1, p-1,
   the published owner `C` (public), C with one bit flipped at 6 positions,
   and 5 hash-derived values.
3. **Proof bit flips**: each of the 8 affine coordinates of `pi_a`, `pi_b`,
   `pi_c` with one bit flipped at 9 positions; the projective z coordinates
   changed; `pi_a` and `pi_c` swapped; `pi_a` negated.
4. **Altered public signals** (other than family 2): an extra signal, an
   empty list, and C + p (a different integer, same field element).
   Encodings of the *same* value (leading zero, hex) are reported separately
   as representation checks, not negatives.
5. **Out-of-range inputs**: for each private input, 2^n, 2^n + value, and
   p - 1 (n = 128, nonce 248), with `C` recomputed by the host Poseidon for
   the exact values, so only the range check can refuse.
6. **Adversarial witnesses**: a valid demo `.wtns` is edited directly and
   then proved with snarkjs (which does not check constraints when proving):
   each input wire +1, 10 internal wires +1, every wire but the constant
   replaced with hash-derived values, and, for each private input, an honest
   witness for a changed value whose `C` wire is overwritten with the demo
   `C`. Each is `wtns check`ed and, after proving, verified against the demo
   `C` and against whatever public signal the proof carries.

Groth16 proofs are malleable: (-pi_a, -pi_b, pi_c) is another valid proof of
the same statement. That is checked and reported as malleability, not as a
negative case.

This is evidence about the cases tried, not a soundness proof.
"""

from __future__ import annotations

import argparse
import copy
import glob
import hashlib
import json
import shutil
import struct
import tempfile
import time
from pathlib import Path

import _bootstrap  # noqa: F401  -- puts the repo root on sys.path

from src.crypto.commitment import DOMAIN_ELEMENT
from src.crypto.poseidon import BN254_SCALAR_FIELD as P
from src.crypto.poseidon import poseidon
from src.utils.results import git_info, repo_root, write_result
from src.utils.seeding import DEFAULT_SEED
from src.zk.toolchain import CIRCUITS_DIR, Toolchain

CIRCUIT = CIRCUITS_DIR / "commitment_opening.circom"
ZKEY = repo_root() / "zk" / "keys" / "commitment_opening_final.zkey"
VKEY = repo_root() / "results" / "zk" / "p7.6" / "verification_key.json"
VECTOR = repo_root() / "tests" / "data" / "commitment_circomlibjs_vector.json"
PUBLICATION = repo_root() / "provenance" / "commitment.json"
OUT_DIR = repo_root() / "results" / "zk" / "p7.8"
BN254_BASE_FIELD = 21888242871839275222246405745257275088696311157297823662689037894645226208583
PRIVATE = ("K_hi", "K_lo", "S", "nonce")
BITS = {"K_hi": 128, "K_lo": 128, "S": 128, "nonce": 248}
LABEL = b"zk-crown/p7.8/demo-derived/v1\x00"
FLIP_BITS = (0, 1, 2, 31, 64, 128, 200, 252, 253)


def derived(tag: str, i: int, bits: int) -> int:
    """A public value below 2^bits: SHA-256(LABEL || tag || i), widened by counter, truncated."""
    out = b""
    j = 0
    while len(out) * 8 < bits:
        out += hashlib.sha256(LABEL + tag.encode() + struct.pack(">II", i, j)).digest()
        j += 1
    return int.from_bytes(out, "big") >> (len(out) * 8 - bits)


def demo() -> tuple[dict, int]:
    v = json.loads(VECTOR.read_text(encoding="utf-8"))
    inputs = [int(x) for x in v["inputs"]]
    if v["input_names"] != ["DOMAIN", *PRIVATE] or inputs[0] != DOMAIN_ELEMENT:
        raise SystemExit("unexpected demo vector layout")
    return dict(zip(PRIVATE, inputs[1:])), int(v["commitment_decimal"])


def host_c(private: dict) -> int:
    return poseidon([DOMAIN_ELEMENT, *(private[k] for k in PRIVATE)])


# --- .wtns files (snarkjs binfile: "wtns", version, sections) ----------------

def read_wtns(path: Path) -> tuple[bytes, int, list[int]]:
    """(header bytes up to the witness section's data, n8, witness values)."""
    b = path.read_bytes()
    if b[:4] != b"wtns":
        raise SystemExit("not a wtns file")
    n_sections = struct.unpack_from("<I", b, 8)[0]
    pos, n8, n_wit, data_at = 12, None, None, None
    for _ in range(n_sections):
        typ, size = struct.unpack_from("<IQ", b, pos)
        pos += 12
        if typ == 1:
            n8 = struct.unpack_from("<I", b, pos)[0]
            q = int.from_bytes(b[pos + 4:pos + 4 + n8], "little")
            n_wit = struct.unpack_from("<I", b, pos + 4 + n8)[0]
            if q != P:
                raise SystemExit("wtns prime is not the BN254 scalar field")
        elif typ == 2:
            data_at = pos
        pos += size
    values = [int.from_bytes(b[data_at + i * n8:data_at + (i + 1) * n8], "little") for i in range(n_wit)]
    return b[:data_at], n8, values


def write_wtns(path: Path, header: bytes, n8: int, values: list[int]) -> None:
    path.write_bytes(header + b"".join((v % P).to_bytes(n8, "little") for v in values))


def wire_index(sym: Path) -> dict[str, int]:
    out = {}
    for line in sym.read_text(encoding="utf-8").splitlines():
        _, w, _, name = line.split(",", 3)
        out[name] = int(w)
    return out


# --- the runner ---------------------------------------------------------------

class Runner:
    def __init__(self, work_dir: Path):
        self.tc = Toolchain(work_dir)
        self.dir = Path(work_dir)
        self.cases: list[dict] = []
        self.controls: list[dict] = []
        self.representations: list[dict] = []
        self.n = 0

    def _name(self) -> str:
        self.n += 1
        return f"c{self.n:04d}"

    def verify(self, public: list, proof: dict) -> tuple[bool, str]:
        name = self._name()
        pub, prf = self.dir / f"{name}_public.json", self.dir / f"{name}_proof.json"
        pub.write_text(json.dumps(public), encoding="utf-8")
        prf.write_text(json.dumps(proof), encoding="utf-8")
        ok = self.tc.verify(VKEY, pub, prf, name)
        out = self.tc.steps[-1].output
        reason = "ok" if ok else ("invalid proof" if "Invalid proof" in out
                                  else "public inputs not valid" if "Public inputs are not valid" in out
                                  else "verifier error")
        return ok, reason

    def witness(self, wasm: Path, inputs: dict) -> tuple[Path | None, str]:
        wtns, step = self.tc.witness(wasm, inputs, self._name(), check=False)
        if step.returncode != 0:
            where = "Num2Bits" if "Num2Bits" in step.output else "CommitmentOpening" if "CommitmentOpening" in step.output else "?"
            return None, f"witness generation failed ({where} assertion)" if "Assert Failed" in step.output else "witness generation failed"
        return wtns, ""

    def wtns_check(self, r1cs: Path, wtns: Path) -> bool:
        return self.tc.snarkjs(f"wtns_check_{self._name()}", "wtns", "check", str(r1cs), str(wtns),
                               check=False).returncode == 0

    def prove(self, wtns: Path) -> tuple[dict | None, list | None]:
        name = self._name()
        prf, pub = self.dir / f"{name}_p.json", self.dir / f"{name}_s.json"
        step = self.tc.snarkjs(f"prove_{name}", "groth16", "prove", str(ZKEY), str(wtns), str(prf), str(pub),
                               check=False)
        if step.returncode != 0:
            return None, None
        return json.loads(prf.read_text(encoding="utf-8")), json.loads(pub.read_text(encoding="utf-8"))

    def record(self, family: str, case: str, accepted: bool, stage: str) -> None:
        self.cases.append({"family": family, "case": case, "accepted": accepted, "refused_at": None if accepted else stage})

    def control(self, case: str, ok: bool) -> None:
        self.controls.append({"case": case, "verified": ok})
        if not ok:
            raise SystemExit(f"positive control failed: {case}")


def flip(value: int, bit: int) -> int:
    return value ^ (1 << bit)


def run(work_dir: Path) -> dict:
    p76 = json.loads(Path(sorted(glob.glob(str(repo_root() / "results" / "p7.6_groth16_setup__*.json")))[-1])
                     .read_text(encoding="utf-8"))
    for path, key in [(ZKEY, "zkey_final"), (VKEY, "verification_key_json")]:
        if hashlib.sha256(path.read_bytes()).hexdigest() != p76["metrics"]["sha256"][key]:
            raise SystemExit(f"{path.name} differs from P7.6")

    r = Runner(work_dir)
    out = r.tc.compile(CIRCUIT)
    if hashlib.sha256(out["r1cs"].read_bytes()).hexdigest() != p76["metrics"]["sha256"]["r1cs"]:
        raise SystemExit("R1CS differs from P7.6")
    wasm, r1cs = out["wasm"], out["r1cs"]
    wires = wire_index(out["sym"])
    private, C = demo()
    if host_c(private) != C:
        raise SystemExit("host Poseidon disagrees with the demo vector")

    # Positive control: the honest demo proof.
    demo_wtns, err = r.witness(wasm, {**private, "C": C})
    if demo_wtns is None:
        raise SystemExit(err)
    proof, public = r.prove(demo_wtns)
    if public != [str(C)]:
        raise SystemExit("demo public signals are not [C]")
    r.control("honest demo proof against demo C", r.verify(public, proof)[0])

    # 1. Wrong opening.
    wrong = []
    for k in PRIVATE:
        n = BITS[k]
        wrong += [(f"{k}+1", {**private, k: private[k] + 1}),
                  (f"{k} top bit flipped", {**private, k: flip(private[k], n - 1)}),
                  (f"{k} low bit flipped", {**private, k: flip(private[k], 0)})]
        wrong += [(f"{k} derived #{i}", {**private, k: derived(k, i, n)}) for i in range(3)]
    wrong.append(("K_hi and K_lo swapped", {**private, "K_hi": private["K_lo"], "K_lo": private["K_hi"]}))
    for label, priv in wrong:
        if priv == private:
            raise SystemExit(f"{label} equals the demo opening")
        wt, err = r.witness(wasm, {**priv, "C": C})
        r.record("1a wrong opening, witness for demo C", label, wt is not None, err)
        c2 = host_c(priv)
        wt2, err2 = r.witness(wasm, {**priv, "C": c2})
        if wt2 is None:
            raise SystemExit(f"{label}: honest witness for its own C failed: {err2}")
        p2, s2 = r.prove(wt2)
        r.control(f"{label}: honest proof against its own C'", r.verify(s2, p2)[0])
        ok, why = r.verify([str(C)], p2)
        r.record("1b wrong opening, own proof checked against demo C", label, ok, f"verify: {why}")

    # 2. Different public C with the valid demo proof.
    owner_c = int(json.loads(PUBLICATION.read_text(encoding="utf-8"))["commitment"]["decimal"])
    others = [("C+1", C + 1), ("C-1", C - 1), ("0", 0), ("1", 1), ("p-1", P - 1), ("published owner C", owner_c)]
    others += [(f"C bit {b} flipped", flip(C, b)) for b in (0, 1, 64, 128, 200, 252)]
    others += [(f"derived #{i}", derived("C", i, 253)) for i in range(5)]
    for label, value in others:
        if value % P == C:
            raise SystemExit(f"{label} equals C")
        ok, why = r.verify([str(value)], proof)
        r.record("2 different public C", label, ok, f"verify: {why}")

    # 3. Proof bit flips.
    coords = [("pi_a", (0,)), ("pi_a", (1,)), ("pi_b", (0, 0)), ("pi_b", (0, 1)), ("pi_b", (1, 0)), ("pi_b", (1, 1)),
              ("pi_c", (0,)), ("pi_c", (1,))]

    def get(p, key, idx):
        x = p[key]
        for i in idx:
            x = x[i]
        return x

    def put(p, key, idx, value):
        x = p[key]
        for i in idx[:-1]:
            x = x[i]
        x[idx[-1]] = value

    for key, idx in coords:
        for b in FLIP_BITS:
            bad = copy.deepcopy(proof)
            put(bad, key, idx, str(flip(int(get(proof, key, idx)), b)))
            ok, why = r.verify(public, bad)
            r.record("3 proof bit flip", f"{key}{list(idx)} bit {b}", ok, f"verify: {why}")
    structural = []
    for key, idx, val in [("pi_a", (2,), "0"), ("pi_a", (2,), "2"), ("pi_c", (2,), "0"), ("pi_c", (2,), "2"),
                          ("pi_b", (2, 0), "0"), ("pi_b", (2, 0), "2"), ("pi_b", (2, 1), "1")]:
        bad = copy.deepcopy(proof)
        put(bad, key, idx, val)
        structural.append((f"{key}{list(idx)} = {val}", bad))
    swapped = copy.deepcopy(proof)
    swapped["pi_a"], swapped["pi_c"] = proof["pi_c"], proof["pi_a"]
    structural.append(("pi_a and pi_c swapped", swapped))
    neg_a = copy.deepcopy(proof)
    neg_a["pi_a"][1] = str(BN254_BASE_FIELD - int(proof["pi_a"][1]))
    structural.append(("pi_a negated", neg_a))
    for label, bad in structural:
        ok, why = r.verify(public, bad)
        r.record("3 proof structure", label, ok, f"verify: {why}")

    # Malleability (not a negative): (-A, -B, C) is a valid proof of the same statement.
    neg_ab = copy.deepcopy(neg_a)
    for j in range(2):
        neg_ab["pi_b"][1][j] = str((BN254_BASE_FIELD - int(proof["pi_b"][1][j])) % BN254_BASE_FIELD)
    mall_ok, _ = r.verify(public, neg_ab)

    # 4. Altered public signals (shape and field size).
    for label, pub in [("extra signal [C, 0]", [str(C), "0"]), ("empty list", []), ("C + p", [str(C + P)]),
                       ("C + 2p", [str(C + 2 * P)])]:
        ok, why = r.verify(pub, proof)
        r.record("4 altered public signals", label, ok, f"verify: {why}")
    for label, pub in [("C with a leading zero", ["0" + str(C)]), ("C as 0x hex", [hex(C)])]:
        ok, why = r.verify(pub, proof)
        r.representations.append({"case": label, "same_field_element": True, "accepted": ok})

    # 5. Out-of-range private inputs (C recomputed by the host, so only Num2Bits can refuse).
    for k in PRIVATE:
        n = BITS[k]
        for label, value in [(f"{k} = 2^{n}", 2**n), (f"{k} = 2^{n} + demo value", 2**n + private[k]),
                             (f"{k} = p-1", P - 1)]:
            priv = {**private, k: value}
            wt, err = r.witness(wasm, {**priv, "C": host_c(priv)})
            stage = err if "Num2Bits" in err else (err or "accepted")
            r.record("5 out-of-range input", label, wt is not None, stage)

    # 6. Adversarial witnesses, proved without a constraint check.
    header, n8, values = read_wtns(demo_wtns)
    if values[0] != 1 or values[wires["main.C"]] != C:
        raise SystemExit("unexpected demo witness layout")
    if any(values[wires[f"main.{k}"]] != private[k] for k in PRIVATE):
        raise SystemExit("unexpected private input wires")
    internal = [i for i in range(len(values)) if i not in {0, *(wires[f"main.{k}"] for k in (*PRIVATE, "C"))}]
    picks = [internal[(j * (len(internal) - 1)) // 9] for j in range(10)]
    edits = [(f"wire {name} + 1", {wires[name]: values[wires[name]] + 1})
             for name in ("main.C", *(f"main.{k}" for k in PRIVATE))]
    edits += [(f"internal wire {i} + 1", {i: values[i] + 1}) for i in picks]
    edits.append(("every wire but the constant hash-derived", {i: derived("wire", i, 253) for i in range(1, len(values))}))
    adversarial = [(label, {**dict(enumerate(values)), **change}) for label, change in edits]
    for k in PRIVATE:
        priv = {**private, k: private[k] + 1}
        wt, err = r.witness(wasm, {**priv, "C": host_c(priv)})
        if wt is None:
            raise SystemExit(err)
        _, _, honest = read_wtns(wt)
        honest[wires["main.C"]] = C
        adversarial.append((f"honest witness for {k}+1 with the C wire set to demo C", dict(enumerate(honest))))
    for label, wmap in adversarial:
        path = work_dir / f"adv_{r._name()}.wtns"
        write_wtns(path, header, n8, [wmap[i] for i in range(len(values))])
        if r.wtns_check(r1cs, path):
            r.record("6 adversarial witness: wtns check", label, True, "")
        else:
            r.record("6 adversarial witness: wtns check", label, False, "wtns check failed")
        prf, pub = r.prove(path)
        if prf is None:
            r.record("6 adversarial witness: proof vs demo C", label, False, "prove failed")
            continue
        ok_c, why_c = r.verify([str(C)], prf)
        r.record("6 adversarial witness: proof vs demo C", label, ok_c, f"verify: {why_c}")
        if pub != [str(C)]:
            ok_own, why_own = r.verify(pub, prf)
            r.record("6 adversarial witness: proof vs its own public signal", label, ok_own, f"verify: {why_own}")

    return {"r": r, "C": C, "proof": proof, "public": public, "malleated_verified": mall_ok,
            "n_witness": len(values), "internal_wires_edited": picks}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="recorded only; cases are fixed")
    args = parser.parse_args()
    git = git_info()
    start = time.perf_counter()

    if (OUT_DIR / "demo_proof.json").exists():
        raise SystemExit(f"{OUT_DIR} already holds a demo proof; refusing to replace it")
    with tempfile.TemporaryDirectory(prefix="zkcrown_p7_8_") as tmp:
        res = run(Path(tmp))
    r = res["r"]
    families: dict[str, dict] = {}
    for c in r.cases:
        f = families.setdefault(c["family"], {"tried": 0, "accepted": 0, "refused_at": {}})
        f["tried"] += 1
        f["accepted"] += int(c["accepted"])
        if not c["accepted"]:
            f["refused_at"][c["refused_at"]] = f["refused_at"].get(c["refused_at"], 0) + 1
    tried, accepted = len(r.cases), sum(c["accepted"] for c in r.cases)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "demo_proof.json").write_text(json.dumps(res["proof"], indent=1), encoding="utf-8")
    (OUT_DIR / "demo_public.json").write_text(json.dumps(res["public"], indent=1), encoding="utf-8")
    path = write_result(
        name="p7.8_negative_tests",
        seed=args.seed,
        task="P7.8",
        params={
            "circuit": "src/zk/circuits/commitment_opening.circom",
            "proving_key": "zk/keys/commitment_opening_final.zkey (P7.6, SHA-256 checked)",
            "verification_key": "results/zk/p7.6/verification_key.json (committed, SHA-256 checked)",
            "inputs": "public demo vector tests/data/commitment_circomlibjs_vector.json and values derived from it; "
                      "no owner secret read",
            "derived_values": "SHA-256(" + LABEL.decode().rstrip("\x00") + "\\0 || tag || u32 i || u32 j), truncated",
            "refused_means": "witness generation, wtns check or proving failed, or groth16 verify did not print OK",
            "scope": "evidence about the cases tried, not a soundness proof",
        },
        metrics={
            "cases_tried": tried,
            "cases_accepted": accepted,
            "families": families,
            "positive_controls": {"count": len(r.controls), "all_verified": all(c["verified"] for c in r.controls)},
            "representations_of_same_C": r.representations,
            "malleability_neg_a_neg_b_verified": res["malleated_verified"],
            "witness_wires": res["n_witness"],
            "internal_wires_edited": res["internal_wires_edited"],
            "cases": r.cases,
            "snarkjs_calls": len(r.tc.steps),
        },
        notes="Demo values only. Accepted count over the negative cases tried; not a soundness proof. "
              "Same-value encodings and Groth16 malleability reported separately.",
        duration_seconds=time.perf_counter() - start,
        git=git,
    )
    print(f"{tried} negative cases, {accepted} accepted; {len(r.controls)} positive controls verified; "
          f"same-C encodings accepted: {[x['accepted'] for x in r.representations]}; "
          f"(-A,-B,C) verified: {res['malleated_verified']}")
    for name, f in families.items():
        print(f"  {name}: {f['tried']} tried, {f['accepted']} accepted, refused at {f['refused_at']}")
    print(f"wrote {path}")
    if accepted:
        raise SystemExit("some negative cases were accepted; see the record")


if __name__ == "__main__":
    main()
