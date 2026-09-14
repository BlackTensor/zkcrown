"""P3.6: produce the final dual-watermarked `W*`, post-hoc weight watermark at alpha 0.1.

    python experiments/p3_6_make_dual_model.py

CPU only, a few minutes.

Decision (owner, P3.6): the weight watermark is embedded **post-hoc** into the
behavioral model from P2.3, at **alpha = 0.1**. Reasons, from measurements
already committed:

- P3.4 and P3.5 measured exactly this setup. At alpha 0.1 the correct key gives
  z = 10.29 with 126 of 128 bits right. Wrong keys stay within |z| <= 2.12, and
  the paired accuracy drop against the P2.3 model is indistinguishable from
  zero on test and holdout. Test loss had not yet started to rise, which it
  does from 0.15. Alpha 0.1 is the largest grid value where that holds.
- No retraining. The P2.3 behavioral model is kept, and no GPU run is needed.
- Blind extraction still works, because the extractor never needs the clean model.

Alpha was chosen after seeing P3.5's test-set drops, so this model's test
accuracy is an optimistic estimate. Embedding during training is not measured,
and it is in the Icebox as a candidate only if Phase 4 shows the post-hoc
watermark does not survive attacks.

Steps:

1. Load the P2.3 `W*` and the P0.5 `W` by hash. Derive `P_K` and `S` from
   `secrets/K.bin` and ``PROJECT_OWNER_ID``.
2. Embed at alpha 0.1 without BN recalibration. Save the state dict atomically
   to ``results/p3.6_dual_wm_W_star.pt`` (gitignored) and record its SHA-256.
   A second save to a temporary file must give the same bytes.
3. Reload that file strictly into a fresh `main_model`. **Every measurement
   below uses the reloaded model**, not the in-memory weights.
4. Weight watermark: blind extraction with `K`. The correlation must equal
   P3.5's alpha 0.1 value.
5. Behavioral watermark: the triggers are regenerated from `K`, and the script
   refuses to report unless their bundle digest equals P2.3's. Measured: WDR
   (P2.4), the P2.8 p-value, and the base-image control, each for both the
   dual model and the P2.3 model. The dual model is called final only if the
   behavioral detection test still rejects H0 at alpha 1e-6, i.e. fired >= 29
   of 100. That criterion was fixed before this run. Anything less is
   recorded with ``final: false`` and the script exits with status 1.
6. Accuracy per image on test and holdout. It must equal P3.5's alpha 0.1
   counts. Paired drops against the P2.3 model and against `W`.

The result file holds aggregates and the weights hash only.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import tempfile
import time
from pathlib import Path

import _bootstrap  # noqa: F401  -- puts the repo root on sys.path
import numpy as np
import torch
from make_master_key import DEFAULT_KEY_PATH, load_key
from p2_4_measure_wdr import P2_3_BUNDLE_SHA256, W_CLEAN_SHA256, W_STAR_SHA256, load_main_model, measure_all, sha256_file
from p3_5_alpha_sweep import paired_pp, score_split

from src.data import cifar10_loaders
from src.data.cifar10 import cifar10_split_indices
from src.models import main_model
from src.utils.results import repo_root, write_result
from src.utils.seeding import DEFAULT_SEED, set_seed
from src.watermark.carrier import carrier_layout
from src.watermark.signature import PROJECT_OWNER_ID, derive_signature
from src.watermark.significance import detection_test
from src.watermark.triggers import DEFAULT_AMPLITUDE, DEFAULT_N
from src.watermark.weight_embedding import derive_carrier_projection, embed_weight_watermark
from src.watermark.weight_extraction import extract_weight_watermark

ALPHA = 0.1
"""Chosen by the owner in P3.6 after P3.5. Not tuned further."""
BEHAVIORAL_GATE_ALPHA = "1e-6"
"""The dual model is final only if the P2.8 test still rejects at this level."""
OUTPUT = "results/p3.6_dual_wm_W_star.pt"
P3_5_RESULT = "results/p3.5_alpha_sweep__seed1337__20260914T084524+0000.json"


def save_state(state: dict, path: Path) -> str:
    """Write `state` atomically to `path`. Returns the file's SHA-256."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(state, tmp)
    os.replace(tmp, path)
    return sha256_file(path)


def p3_5_row(alpha: float) -> dict:
    rows = json.loads((repo_root() / P3_5_RESULT).read_text(encoding="utf-8"))["metrics"]["rows"]
    matches = [r for r in rows if r["alpha"] == alpha]
    if len(matches) != 1:
        raise SystemExit(f"P3.5 has no single row for alpha {alpha}")
    return matches[0]


def behavioral_gate(fired: int, n: int) -> dict:
    test = detection_test(fired, n)
    return {**test.summary(), "gate_alpha": BEHAVIORAL_GATE_ALPHA, "passes_gate": test.rejects(BEHAVIORAL_GATE_ALPHA)}


def main(argv: list[str] | None = None) -> dict:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--key", type=Path, default=DEFAULT_KEY_PATH)
    parser.add_argument("--data-root", type=Path, default=repo_root() / "data")
    parser.add_argument("--w-star", type=Path, default=repo_root() / "results" / "p2.3_behavioral_wm_W_star.pt")
    parser.add_argument("--w-clean", type=Path, default=repo_root() / "results" / "p0.5_clean_baseline_W.pt")
    parser.add_argument("--output", type=Path, default=repo_root() / OUTPUT)
    args = parser.parse_args(argv)

    backends = set_seed(args.seed)
    started = time.perf_counter()
    device = torch.device("cpu")
    behavioral = load_main_model(args.w_star, W_STAR_SHA256)
    baseline = load_main_model(args.w_clean, W_CLEAN_SHA256)
    layout = carrier_layout(main_model())
    key = load_key(args.key)
    projection = derive_carrier_projection(key, layout)
    signature = derive_signature(key, PROJECT_OWNER_ID)

    # 2. embed and save
    dual_state, embedding = embed_weight_watermark(behavioral.state_dict(), layout, projection, signature, ALPHA)
    weights_sha256 = save_state(dual_state, args.output)
    with tempfile.TemporaryDirectory() as tmp:
        # torch.save stores the file name inside the archive, so compare under the same name.
        second = save_state(dual_state, Path(tmp) / args.output.name)
    if second != weights_sha256:
        raise SystemExit("saving the same state dict twice gave different bytes")

    # 3. reload; everything below measures the file on disk
    dual = load_main_model(args.output, weights_sha256)
    reloaded = dual.state_dict()
    for name, tensor in dual_state.items():
        if not torch.equal(reloaded[name], tensor):
            raise SystemExit(f"reloaded tensor {name} differs from the embedded weights")

    # 4. weight watermark
    reference = p3_5_row(ALPHA)
    extraction = extract_weight_watermark(reloaded, layout, projection, signature)
    if abs(extraction.correlation - reference["detection"]["correlation"]) > 1e-12:
        raise SystemExit(f"correlation {extraction.correlation} differs from P3.5's {reference['detection']['correlation']}")
    behavioral_extraction = extract_weight_watermark(behavioral.state_dict(), layout, projection, signature)

    # 5. behavioral watermark
    from torchvision.datasets import CIFAR10

    cifar = CIFAR10(str(args.data_root), train=True, download=False)
    pool, _ = cifar10_split_indices(len(cifar.data))
    digest, triggers = measure_all(
        key, cifar.data, np.asarray(cifar.targets), pool, {"dual": dual, "behavioral_only": behavioral},
        n=DEFAULT_N, amplitude=DEFAULT_AMPLITUDE, device=device,
    )
    if digest != P2_3_BUNDLE_SHA256:
        raise SystemExit(f"K regenerates trigger bundle {digest}, expected {P2_3_BUNDLE_SHA256}. Refusing to report.")
    gates = {name: behavioral_gate(s["triggers"]["fired"], s["triggers"]["n"]) for name, s in triggers.items()}

    # 6. accuracy
    loaders = cifar10_loaders(args.data_root, eval_batch_size=500, num_workers=0, download=False)
    accuracy = {}
    for split in ("test", "holdout"):
        d = score_split(dual, loaders[split])
        b = score_split(behavioral, loaders[split])
        c = score_split(baseline, loaders[split])
        if d["correct_count"] != reference[split]["correct"]:
            raise SystemExit(f"{split}: {d['correct_count']} correct, P3.5 recorded {reference[split]['correct']}")
        accuracy[split] = {
            "accuracy": d["accuracy"], "correct": d["correct_count"], "n": d["n"], "loss": d["loss"],
            "behavioral_only_accuracy": b["accuracy"], "baseline_W_accuracy": c["accuracy"],
            "vs_behavioral_only": paired_pp(b["correct"], d["correct"]),
            "vs_W": paired_pp(c["correct"], d["correct"]),
        }

    final = gates["dual"]["passes_gate"]
    metrics = {
        "final": final,
        "weights_sha256": weights_sha256,
        "weights_path": OUTPUT,
        "save_is_deterministic": True,
        "embedding": embedding.to_dict(),
        "weight_watermark": {
            "dual": {**extraction.to_dict(), "z": extraction.correlation * math.sqrt(extraction.rows)},
            "behavioral_only": {**behavioral_extraction.to_dict(),
                                "z": behavioral_extraction.correlation * math.sqrt(behavioral_extraction.rows)},
            "matches_p3_5": True,
        },
        "behavioral_watermark": {"trigger_bundle_sha256": digest, "scores": triggers, "detection_test": gates},
        "accuracy": accuracy,
        "matches_p3_5_accuracy": True,
    }
    path = write_result(
        name="p3.6_dual_wm",
        seed=args.seed,
        task="P3.6",
        params={
            "decision": "post-hoc weight watermark on the P2.3 behavioral model (owner, P3.6)",
            "alpha": ALPHA,
            "alpha_note": "chosen after P3.5's test-set drops; test accuracy of this model is an optimistic estimate",
            "bn_recalibration": False,
            "host": {"task": "P2.3", "weights_sha256": W_STAR_SHA256},
            "baseline": {"task": "P0.5/P0.6", "weights_sha256": W_CLEAN_SHA256},
            "owner_id": PROJECT_OWNER_ID,
            "carrier_digest": layout.digest(),
            "behavioral_gate": f"P2.8 test must reject at alpha {BEHAVIORAL_GATE_ALPHA} (fired >= 29 of 100), fixed before the run",
            "measured_from": "weights reloaded from the saved file",
            "p3_5_reference": P3_5_RESULT,
            "device": "cpu",
        },
        metrics=metrics,
        seeded_backends=backends,
        duration_seconds=time.perf_counter() - started,
        notes=(
            "Final dual-watermarked W*: P2.3 behavioral model plus post-hoc weight watermark at alpha 0.1. One host, "
            "one key, one embedding. Weights file is gitignored; only its hash is here. No weight-watermark threshold "
            "yet (P3.7). Aggregates only."
        ),
    )

    print(f"dual W*: {args.output}  sha256 {weights_sha256}")
    w = metrics["weight_watermark"]
    print(f"weight watermark  dual corr {w['dual']['correlation']:+.4f} z {w['dual']['z']:.2f} bits {w['dual']['bit_matches']}  "
          f"| P2.3 model corr {w['behavioral_only']['correlation']:+.4f} z {w['behavioral_only']['z']:.2f}")
    for name, s in triggers.items():
        t, g = s["triggers"], gates[name]
        print(f"behavioral {name:16s} fired {t['fired']}/{t['n']}  mean target prob {t['target_probability_mean']:.4f} "
              f"(min {t['target_probability_min']:.4f})  p {g['p_value']:.3g}  bases fired {s['base_images']['fired']}  "
              f"passes gate at {BEHAVIORAL_GATE_ALPHA}: {g['passes_gate']}")
    for split, a in accuracy.items():
        v, c = a["vs_behavioral_only"], a["vs_W"]
        print(f"{split:8s} {a['accuracy']:.2%}  drop vs P2.3 {v['drop_pp']:+.2f} pp [{v['ci95_pp'][0]:+.2f}, {v['ci95_pp'][1]:+.2f}] "
              f"p {v['mcnemar_exact_p']:.3g}  drop vs W {c['drop_pp']:+.2f} pp [{c['ci95_pp'][0]:+.2f}, {c['ci95_pp'][1]:+.2f}] "
              f"p {c['mcnemar_exact_p']:.3g}")
    print("FINAL" if final else "NOT FINAL: behavioral watermark failed the gate")
    print("wrote", path)
    if not final:
        raise SystemExit(1)
    return {"path": path, "metrics": metrics}


if __name__ == "__main__":
    main()
