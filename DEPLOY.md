# Deploying the zk-Crown dashboard (Streamlit Community Cloud, free tier)

This is the procedure for P9.15, and the record of the deployment.

## Deployment (P9.15, 2026-10-07)

- **Live app:** <https://zkcrown-akfez8euv8rfndkfhweeco.streamlit.app/>. The Data integrity page is at <https://zkcrown-akfez8euv8rfndkfhweeco.streamlit.app/integrity>.
- **Repository:** <https://github.com/BlackTensor/zkcrown>, public, default branch `master`. The deployed tree is commit `3e5259f`, which passed the pre-deploy check.
- **Host settings:** Streamlit Community Cloud, free tier. Entry point `app/streamlit_app.py`, Python version as chosen at deploy time (see CLAUDE.md, P9.15), **no secrets configured**.
- The deploy itself was done by the owner. The signed tag `provenance-commitment-v1` has not been pushed.

## What the hosted app is

- **Entry point:** `app/streamlit_app.py`. Theme: `.streamlit/config.toml`.
- **Requirements:** Community Cloud reads `app/requirements.txt`, the file next to the entry point. It pins `streamlit` only. numpy comes with Streamlit and is used by one module, `app/fingerprint_live.py`.
- **Data:** the app reads only committed files under `results/`, `figures/` and `provenance/` that are listed in `app/manifest.json`. Each read is re-hashed against the recorded SHA-256, and any mismatch is shown as an error.
- **Bundled model:** exactly one, the owner's dual-watermarked `W*`, as `results/p9.15_dual_W_star.npz` (1,245,090 bytes). Its fingerprint equals the one the published commitment names. It is used for one live check: the verbatim copy's fingerprint on the Live audit page.

**Not on the host** (it is not in git, and the app has no way to read it):

- the master key `K`, the commitment nonce, the record signing key, the GPG private key;
- the real trigger bundle;
- every other model's weights;
- the snarkjs and ezkl toolchains, the Groth16 proving key, the EZKL proving key, the SRS.

**Live versus replayed:**

- The watermark checks need `K`, so they are replayed from the committed audit. So are the zero-knowledge proof checks.
- The commitment check, the grade, the verbatim copy's fingerprint, the consistency checks and the data-integrity hashes run live.
- Every page states which applies, under its title.

## Before any push

1. Commit everything. The pre-deploy check requires a clean working tree.
2. Run the pre-deploy check:

   ```
   .venv/Scripts/python.exe experiments/p9_15_predeploy_check.py
   ```

   It must end with every check `PASS`. It checks the committed tree, then writes `results/p9.15_predeploy_check__*.json`. Commit that record, then regenerate the manifest (`python app/build_manifest.py`) and commit it: a new tracked results file always makes the manifest stale. Those two files are the only changes after the check. It checks:

   - a clean tree;
   - nothing tracked under `secrets/` except the empty `secrets/.gitkeep`;
   - no secret-looking file names;
   - no model weights except the approved bundle;
   - the bundle is the published model, and not the behavioral-only P2.3 model or clean `W`;
   - no secret values in any tracked file (needs the owner's secrets on the machine; otherwise reported as NOT RUN, which also fails);
   - every page states what is live and what is replayed;
   - minimal requirements;
   - the manifest is current.

   Listed under "kept by owner decision", not failing: `results/zk/p8.1/zk_model.onnx`, the small MNIST model's weights, committed in P8.1 before this rule existed. The owner decided on 2026-10-07 to keep it in the repository.
3. Optionally run the app tests (`pytest tests/test_dashboard*.py`) and look at the app locally with `streamlit run app/streamlit_app.py`.

## Publishing a repository makes all of it public

A public GitHub repository publishes every tracked file and the whole history, not just the app. That includes:

- `CLAUDE.md`, with the owner's email, which is already in the GPG user ID;
- all result records and figures;
- the provenance files and the public GPG key;
- the experiment and source code.

Secrets are not tracked and are not published. A private repository can also be deployed on Community Cloud if the owner grants it access to the repository. Check the current free-tier terms when deploying.

## Deploying

1. Create an empty repository on GitHub. Do not add a README or a license there, so it does not conflict with this history.
2. Push from this machine. The commands below are an example; the owner runs them:

   ```
   git remote add origin https://github.com/BlackTensor/zkcrown.git
   git push -u origin master
   ```

   The tag `provenance-commitment-v1` is pushed only with `git push origin provenance-commitment-v1`. Pushing it gives the tag third-party time evidence (the host records the push). It still holds the older, pending proof.
3. On share.streamlit.io, sign in with GitHub and create a new app:
   - repository `BlackTensor/zkcrown`, branch `master`;
   - main file path `app/streamlit_app.py`;
   - in Advanced settings, choose Python 3.11 (the version the app and its tests run on locally) if offered, and add **no secrets**. The app reads no secrets.
4. Deploy, then check the live app:
   - Data integrity: every listed file verifies.
   - Live audit, Verbatim copy, Run the audit: the Model fingerprint step says `live` and `Passed`.
   - Each page shows its "On this page" line.

## After deploying

- Each later commit that changes a file the app reads must regenerate `app/manifest.json` (`python app/build_manifest.py`), or the app shows a data error for that file.
- Do not add any other model file to the repository. The pre-deploy check fails on it.
