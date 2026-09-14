"""P3.4: the weight watermark extracts with the correct `K` and not with wrong keys.

    python experiments/p3_4_key_specificity.py

CPU only, several minutes. The steps:

1. Load two real hosts by hash: `W*` from P2.3, the behavioral model that is
   carried forward, and clean `W` from P0.5.
2. Derive `P_K` and `S` from `secrets/K.bin` and the project owner id
   ``PROJECT_OWNER_ID``. Embed the weight watermark post-hoc (P3.2) at each
   alpha in a small grid. Alpha 0 is the unwatermarked control.
3. Extract (P3.3) from every embedded model with:
   - **the correct key**: `P_K` and `S` from `K`;
   - **M wrong keys**: each wrong key ``K'_j`` derives its own ``P_K'`` and
     ``S'``, which is what an auditor holding the wrong key would do;
   - **wrong projection, true S**: ``P_K'`` scored against the real `S`. This
     costs nothing extra, and it shows the projection key, not only knowing
     `S`, is what makes extraction work.
4. Report, for each host and alpha, the correct-key correlation, the
   distribution of wrong-key correlations, and the gap between them, as a raw
   difference and in wrong-key standard deviations. No detection threshold
   or p-value is derived. That is P3.7, with 1,000 keys and a fitted null.

The alpha grid is fixed before running and is **not a selection**. The
accuracy cost of each alpha is not measured here (P3.5). Choosing alpha and
post-hoc versus during-training is P3.6.

Wrong keys are ``SHA-256("zk-crown/p3.4/wrong-key/v1" || u64 seed || u64 j)``,
public, and a different family from P2.8's null keys and from P3.7's. The
script refuses to run if one equals `K`. The result file holds aggregate
correlations only. `S`, `P_K`, the fingerprints and the watermarked weights
stay in memory.
"""

from __future__ import annotations

import argparse
import hashlib
import math
import time
from collections.abc import Mapping
from pathlib import Path

import _bootstrap  # noqa: F401  -- puts the repo root on sys.path
import numpy as np
import torch
from make_master_key import DEFAULT_KEY_PATH, load_key
from p2_4_measure_wdr import W_CLEAN_SHA256, W_STAR_SHA256, load_main_model

from src.models import main_model
from src.utils.results import repo_root, write_result
from src.utils.seeding import DEFAULT_SEED, set_seed
from src.watermark.carrier import CarrierLayout, carrier_layout
from src.watermark.signature import PROJECT_OWNER_ID, derive_signature
from src.watermark.weight_embedding import derive_carrier_projection, embed_weight_watermark
from src.watermark.weight_extraction import extract_weight_watermark, recover_fingerprint, score_fingerprint

WRONG_KEY_LABEL = b"zk-crown/p3.4/wrong-key/v1"
ALPHAS = (0.0, 0.005, 0.01, 0.02, 0.05, 0.1)
"""Fixed before running. Spans a weak to a strong watermark relative to the
host term. Not tuned, not accuracy-checked (P3.5)."""
DEFAULT_WRONG_KEYS = 100


def wrong_key(seed: int, j: int) -> bytes:
    """The j-th public wrong key. Not `K`, never secret."""
    return hashlib.sha256(WRONG_KEY_LABEL + seed.to_bytes(8, "big") + j.to_bytes(8, "big")).digest()


def summarise(values: list[float]) -> dict:
    a = np.asarray(values, dtype=np.float64)
    return {
        "count": int(a.size),
        "mean": float(a.mean()),
        "sd": float(a.std(ddof=1)) if a.size > 1 else 0.0,
        "min": float(a.min()),
        "max": float(a.max()),
        "max_abs": float(np.abs(a).max()),
    }


def run_specificity(
    hosts: Mapping[str, Mapping[str, torch.Tensor]],
    layout: CarrierLayout,
    key: bytes,
    owner_id: str,
    alphas: tuple[float, ...],
    wrong_keys: list[bytes],
    *,
    log=print,
) -> dict:
    """Embed with `key` at each alpha, extract with `key` and every wrong key.

    Returns per host, per alpha: the embedding summary, the correct-key
    extraction, wrong-key and wrong-projection correlation summaries, and the gaps.
    """
    if key in wrong_keys or len(set(wrong_keys)) != len(wrong_keys):
        raise ValueError("wrong keys must be distinct and differ from K")
    if len(wrong_keys) < 2:
        raise ValueError("need at least two wrong keys to estimate a spread")
    projection = derive_carrier_projection(key, layout)
    signature = derive_signature(key, owner_id)

    marked: dict[tuple[str, float], dict[str, torch.Tensor]] = {}
    rows: dict[str, dict[str, dict]] = {name: {} for name in hosts}
    for name, state in hosts.items():
        for alpha in alphas:
            state_alpha, embedding = embed_weight_watermark(state, layout, projection, signature, alpha)
            marked[(name, alpha)] = state_alpha
            correct = extract_weight_watermark(state_alpha, layout, projection, signature)
            rows[name][repr(alpha)] = {
                "alpha": alpha,
                "embedding": embedding.to_dict(),
                "correct_key": {**correct.to_dict(), "z": correct.correlation * math.sqrt(correct.rows)},
                "_wrong_corr": [],
                "_wrong_bits": [],
                "_wrong_proj_true_s_corr": [],
            }

    started = time.perf_counter()
    for j, k_wrong in enumerate(wrong_keys):
        p_wrong = derive_carrier_projection(k_wrong, layout)
        s_wrong = derive_signature(k_wrong, owner_id)
        if s_wrong.value == signature.value:
            raise ValueError("a wrong key derived the true S")
        for (name, alpha), state_alpha in marked.items():
            y = recover_fingerprint(state_alpha, layout, p_wrong)
            corr, _, _, bits = score_fingerprint(y, s_wrong)
            corr_true_s, _, _, _ = score_fingerprint(y, signature)
            row = rows[name][repr(alpha)]
            row["_wrong_corr"].append(corr)
            row["_wrong_bits"].append(bits)
            row["_wrong_proj_true_s_corr"].append(corr_true_s)
        if (j + 1) % 10 == 0 or j + 1 == len(wrong_keys):
            log(f"  wrong keys {j + 1}/{len(wrong_keys)}  ({time.perf_counter() - started:.0f} s)")

    rows_per_projection = projection.rows
    for name in rows:
        for row in rows[name].values():
            wrong = summarise(row.pop("_wrong_corr"))
            bits = row.pop("_wrong_bits")
            wrong_true_s = summarise(row.pop("_wrong_proj_true_s_corr"))
            correct_corr = row["correct_key"]["correlation"]
            gap = correct_corr - wrong["mean"]
            row["wrong_keys"] = {
                "correlation": wrong,
                "z_max_abs": wrong["max_abs"] * math.sqrt(rows_per_projection),
                "bit_matches": {"mean": float(np.mean(bits)), "min": int(min(bits)), "max": int(max(bits))},
            }
            row["wrong_projection_true_signature"] = {
                "correlation": wrong_true_s,
                "z_max_abs": wrong_true_s["max_abs"] * math.sqrt(rows_per_projection),
            }
            row["gap"] = {
                "correct_minus_wrong_mean": gap,
                "in_wrong_key_sd": gap / wrong["sd"] if wrong["sd"] > 0 else math.inf,
                "correct_minus_wrong_max": correct_corr - wrong["max"],
                "correct_above_every_wrong_key": bool(correct_corr > wrong["max"]),
            }
    return rows


def main(argv: list[str] | None = None) -> dict:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--key", type=Path, default=DEFAULT_KEY_PATH)
    parser.add_argument("--w-star", type=Path, default=repo_root() / "results" / "p2.3_behavioral_wm_W_star.pt")
    parser.add_argument("--w-clean", type=Path, default=repo_root() / "results" / "p0.5_clean_baseline_W.pt")
    parser.add_argument("--wrong-keys", type=int, default=DEFAULT_WRONG_KEYS)
    args = parser.parse_args(argv)

    backends = set_seed(args.seed)
    started = time.perf_counter()
    hosts = {
        "W_star": load_main_model(args.w_star, W_STAR_SHA256).state_dict(),
        "W_clean": load_main_model(args.w_clean, W_CLEAN_SHA256).state_dict(),
    }
    layout = carrier_layout(main_model())
    key = load_key(args.key)
    wrong = [wrong_key(args.seed, j) for j in range(args.wrong_keys)]

    rows = run_specificity(hosts, layout, key, PROJECT_OWNER_ID, ALPHAS, wrong)

    path = write_result(
        name="p3.4_key_specificity",
        seed=args.seed,
        task="P3.4",
        params={
            "hosts": {"W_star": W_STAR_SHA256, "W_clean": W_CLEAN_SHA256},
            "owner_id": PROJECT_OWNER_ID,
            "embedding": "post-hoc W + alpha * P_K^T * S on the carrier (P3.2)",
            "extraction": "blind, per-tensor centred, normalised correlation (P3.3)",
            "carrier_digest": layout.digest(),
            "dim": layout.dim,
            "rows": 128,
            "alphas": list(ALPHAS),
            "alphas_note": "fixed before running; not a selection; accuracy not measured (P3.5)",
            "wrong_keys": {
                "count": args.wrong_keys,
                "derivation": "SHA-256('zk-crown/p3.4/wrong-key/v1' || u64 seed || u64 j), public, never K",
                "each_derives": "its own P_K' and S' for the same owner id",
            },
            "z": "correlation * sqrt(128); a scale, not a p-value",
            "device": "cpu",
        },
        metrics={"hosts": rows},
        seeded_backends=backends,
        duration_seconds=time.perf_counter() - started,
        notes=(
            "Key specificity of the weight watermark on real hosts, post-hoc embedding. No threshold or p-value: "
            "the null distribution and threshold are P3.7. Accuracy of the embedded models is not measured (P3.5). "
            "Aggregates only; S, P_K, fingerprints and watermarked weights are not in this file."
        ),
    )

    for name, by_alpha in rows.items():
        print(f"\n{name}")
        print("  alpha   correct corr  bits   wrong mean   wrong sd  wrong max|z|  gap     gap/sd  above all  wrongP+trueS max|z|")
        for row in by_alpha.values():
            c, w, g = row["correct_key"], row["wrong_keys"], row["gap"]
            print(
                f"  {row['alpha']:<6}  {c['correlation']:+.4f}      {c['bit_matches']:>3}   {w['correlation']['mean']:+.4f}"
                f"     {w['correlation']['sd']:.4f}   {w['z_max_abs']:.2f}         {g['correct_minus_wrong_mean']:+.4f}"
                f"  {g['in_wrong_key_sd']:+7.2f}  {str(g['correct_above_every_wrong_key']):9s}"
                f"  {row['wrong_projection_true_signature']['z_max_abs']:.2f}"
            )
    print("wrote", path)
    return {"path": path, "metrics": {"hosts": rows}}


if __name__ == "__main__":
    main()
