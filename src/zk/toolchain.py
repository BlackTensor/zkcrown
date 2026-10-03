"""Thin wrapper over the pinned Track A command line tools (P7.2).

circom compiles a circuit; snarkjs runs the powers of tau, the Groth16 setup,
witness generation, proving and verification. Every call is a subprocess with
the exact arguments recorded, and a non-zero exit raises `ToolError` with the
tool's output. Nothing here knows about `K` or any project secret.

Paths: circom is `tools/bin/circom(.exe)`, snarkjs is run as
`node zk/node_modules/snarkjs/build/cli.cjs`, and `zk/node_modules` is the
circom include path, so circuits write `include "circomlib/circuits/...";`.
See `zk/README.md` for the install.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
CIRCOM = REPO_ROOT / "tools" / "bin" / ("circom.exe" if sys.platform == "win32" else "circom")
NODE_MODULES = REPO_ROOT / "zk" / "node_modules"
SNARKJS_CLI = NODE_MODULES / "snarkjs" / "build" / "cli.cjs"
CIRCUITS_DIR = REPO_ROOT / "src" / "zk" / "circuits"


class ToolError(RuntimeError):
    """A circom or snarkjs command failed."""


@dataclass
class Step:
    """One recorded command: what ran, how long it took, whether it succeeded."""

    name: str
    argv: list[str]
    seconds: float
    returncode: int
    output: str = field(repr=False)


def toolchain_available() -> bool:
    return CIRCOM.is_file() and SNARKJS_CLI.is_file() and shutil.which("node") is not None


class Toolchain:
    """Runs circom and snarkjs commands in a working directory and logs each step."""

    def __init__(self, work_dir: Path | str):
        if not toolchain_available():
            raise ToolError("Track A toolchain not installed; see zk/README.md")
        self.work_dir = Path(work_dir)
        self.work_dir.mkdir(parents=True, exist_ok=True)
        self.node = shutil.which("node")
        self.steps: list[Step] = []

    def _run(self, name: str, argv: list[str], *, check: bool = True) -> Step:
        start = time.perf_counter()
        proc = subprocess.run(argv, cwd=self.work_dir, capture_output=True, text=True, timeout=1800)
        step = Step(name, argv, time.perf_counter() - start, proc.returncode, (proc.stdout or "") + (proc.stderr or ""))
        self.steps.append(step)
        if check and proc.returncode != 0:
            raise ToolError(f"{name} failed (exit {proc.returncode}):\n{step.output[-2000:]}")
        return step

    def snarkjs(self, name: str, *args: str, check: bool = True) -> Step:
        return self._run(name, [self.node, str(SNARKJS_CLI), *args], check=check)

    # --- circuit ---------------------------------------------------------

    def compile(self, circuit: Path) -> dict[str, Path]:
        """circom --r1cs --wasm --sym. Returns the paths of the outputs."""
        self._run("circom_compile", [str(CIRCOM), str(circuit), "--r1cs", "--wasm", "--sym",
                                     "-l", str(NODE_MODULES), "-o", str(self.work_dir)])
        stem = circuit.stem
        return {"r1cs": self.work_dir / f"{stem}.r1cs", "sym": self.work_dir / f"{stem}.sym",
                "wasm": self.work_dir / f"{stem}_js" / f"{stem}.wasm"}

    def r1cs_info(self, r1cs: Path) -> dict[str, int]:
        """Parse `snarkjs r1cs info` into integers (constraints, wires, inputs...)."""
        out = self.snarkjs("r1cs_info", "r1cs", "info", str(r1cs)).output
        info = {}
        for label, key in [("# of Wires", "wires"), ("# of Constraints", "constraints"),
                           ("# of Private Inputs", "private_inputs"), ("# of Public Inputs", "public_inputs"),
                           ("# of Labels", "labels"), ("# of Outputs", "outputs")]:
            m = re.search(re.escape(label) + r":\s*(\d+)", out)
            if m:
                info[key] = int(m.group(1))
        return info

    # --- setup -----------------------------------------------------------

    def local_powers_of_tau(self, power: int, name: str = "pot") -> Path:
        """A single-contributor powers of tau made here. For toy circuits only.

        Its toxic waste exists in this process's memory while it runs, so it
        gives no soundness against whoever ran it. P7.4 replaces it with a
        public ceremony file for the real circuit.
        """
        p0 = self.work_dir / f"{name}_0000.ptau"
        p1 = self.work_dir / f"{name}_0001.ptau"
        final = self.work_dir / f"{name}_final.ptau"
        self.snarkjs("ptau_new", "powersoftau", "new", "bn128", str(power), str(p0))
        self.snarkjs("ptau_contribute", "powersoftau", "contribute", str(p0), str(p1),
                     "--name=zk-crown toy contribution", "-e=" + _entropy())
        self.snarkjs("ptau_prepare_phase2", "powersoftau", "prepare", "phase2", str(p1), str(final))
        return final

    def groth16_setup(self, r1cs: Path, ptau: Path, name: str) -> dict[str, Path]:
        """Phase 2 for one circuit: setup, one contribution, export the verification key."""
        z0 = self.work_dir / f"{name}_0000.zkey"
        z1 = self.work_dir / f"{name}_final.zkey"
        vkey = self.work_dir / f"{name}_verification_key.json"
        self.snarkjs("groth16_setup", "groth16", "setup", str(r1cs), str(ptau), str(z0))
        self.snarkjs("zkey_contribute", "zkey", "contribute", str(z0), str(z1),
                     "--name=zk-crown phase 2 contribution", "-e=" + _entropy())
        self.snarkjs("zkey_verify", "zkey", "verify", str(r1cs), str(ptau), str(z1))
        self.snarkjs("zkey_export_vkey", "zkey", "export", "verificationkey", str(z1), str(vkey))
        return {"zkey": z1, "vkey": vkey}

    # --- prove and verify -------------------------------------------------

    def witness(self, wasm: Path, inputs: dict, name: str, *, check: bool = True) -> tuple[Path, Step]:
        """Write the input JSON and compute the witness. Field elements go in as decimal strings."""
        inp = self.work_dir / f"{name}_input.json"
        inp.write_text(json.dumps(_decimal(inputs)), encoding="utf-8")
        wtns = self.work_dir / f"{name}.wtns"
        step = self.snarkjs(f"wtns_calculate_{name}", "wtns", "calculate", str(wasm), str(inp), str(wtns), check=check)
        return wtns, step

    def check_witness(self, r1cs: Path, wtns: Path) -> Step:
        return self.snarkjs("wtns_check", "wtns", "check", str(r1cs), str(wtns))

    def prove(self, zkey: Path, wtns: Path, name: str) -> dict[str, Path]:
        proof = self.work_dir / f"{name}_proof.json"
        public = self.work_dir / f"{name}_public.json"
        self.snarkjs(f"groth16_prove_{name}", "groth16", "prove", str(zkey), str(wtns), str(proof), str(public))
        return {"proof": proof, "public": public}

    def verify(self, vkey: Path, public: Path, proof: Path, name: str) -> bool:
        """True only if snarkjs exits 0 and prints its OK line."""
        step = self.snarkjs(f"groth16_verify_{name}", "groth16", "verify", str(vkey), str(public), str(proof),
                            check=False)
        return step.returncode == 0 and "OK!" in step.output


def _decimal(value):
    if isinstance(value, bool):
        raise TypeError("booleans are not field elements")
    if isinstance(value, int):
        return str(value)
    if isinstance(value, (list, tuple)):
        return [_decimal(v) for v in value]
    if isinstance(value, dict):
        return {k: _decimal(v) for k, v in value.items()}
    return value


def _entropy() -> str:
    """Fresh OS randomness for a ceremony contribution. Never stored."""
    import secrets

    return secrets.token_hex(32)
