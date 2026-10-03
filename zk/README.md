# Track A toolchain (P7.2)

Exact versions, installed locally inside the repo. Nothing is installed
system-wide.

| Tool | Version | Source | Location (gitignored) |
|---|---|---|---|
| circom | 2.2.3 | prebuilt `circom-windows-amd64.exe`, GitHub release `v2.2.3` of `iden3/circom` | `tools/bin/circom.exe` |
| snarkjs | 0.7.6 | npm, pinned in `package.json`, locked in `package-lock.json` | `zk/node_modules/` |
| circomlib | 2.0.5 | npm, pinned in `package.json`, locked in `package-lock.json` | `zk/node_modules/` |

The circom binary's SHA-256 must be
`e43f132ee6f0aa79b705beceb59c2a7e6a54d7bdeab917ca34e9fc1951d185e1`, the
digest GitHub publishes for that release asset.

## Reproduce

From the repo root, with Node and the GitHub CLI available:

```sh
mkdir -p tools/bin
gh release download v2.2.3 -R iden3/circom -p circom-windows-amd64.exe -D tools/bin
mv tools/bin/circom-windows-amd64.exe tools/bin/circom.exe
(cd zk && npm ci)
python experiments/p7_2_toolchain_versions.py
```

`npm ci` installs exactly what the lockfile records. The last command checks
every version and the binary's hash, and writes a result record.

On Linux (for example Colab), use the `circom-linux-amd64` asset of the same
release, saved as `tools/bin/circom`. Its hash is not pinned here. The check
script skips the hash comparison off Windows but still requires version 2.2.3.

Run snarkjs as `node zk/node_modules/snarkjs/build/cli.cjs ...` or
`npx snarkjs ...` from `zk/`.

snarkjs and circomlib are GPL-3.0 licensed.
