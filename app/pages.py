"""The dashboard's pages (P9.5): name, section, icon, and which task fills each one.

Every page except Data integrity is a placeholder until its task (P9.6 to
P9.14) builds the panel. Descriptions say what the panel will show, in words;
they carry no measured values.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PageSpec:
    slug: str
    title: str
    section: str
    icon: str
    tasks: str
    summary: str
    plan: tuple[str, ...]
    default: bool = False


PAGES: tuple[PageSpec, ...] = (
    PageSpec(
        "overview", "Overview", "Start", ":material/home:", "P9.6",
        "A neural network carries a secret ownership identity, is attacked the way a thief would attack it, and is "
        "then audited against a commitment published before the theft.",
        ("The pipeline at a glance: watermark, attack, commit, prove, audit.",
         "Each stage links to its own page.",
         "Headline figures read from the committed results."),
        default=True),
    PageSpec(
        "audit", "Live audit", "Audit", ":material/policy:", "P9.7, P9.8",
        "Pick a suspect model and follow the five checks one by one, ending in a graded verdict.",
        ("Suspects: a verbatim copy, quantized, pruned, fine-tuned and distilled thefts, and unrelated models.",
         "Watermark checks are replayed from the recorded audit, and every step says whether it is live or replayed.",
         "The grade is recomputed from the verdict shown, with its caveats and the separate owner evidence.",
         "What private data the audit used, taken from the verdict itself."),
    ),
    PageSpec(
        "evidence", "Evidence strength", "Audit", ":material/query_stats:", "P9.9",
        "Each p-value drawn against its null distribution, so the gap between a watermarked model and chance is "
        "visible.",
        ("Behavioral test: the binomial bound, the wrong-key histograms and the detection thresholds.",
         "Weight test: the proven tail bound and the recorded null aggregates.",
         "The suspect's own statistic marked on each."),
    ),
    PageSpec(
        "attacks", "Attack lab", "Experiments", ":material/swords:", "P9.10",
        "Every attack setting from the robustness study, with its accuracy cost and both detection tests.",
        ("Filter by attack family and setting.",
         "Accuracy cost beside each outcome.",
         "The evidence tier for every row."),
    ),
    PageSpec(
        "triggers", "Trigger gallery", "Experiments", ":material/grid_view:", "P9.11",
        "Public demo-key triggers next to their clean base images. The real triggers are never shown.",
        ("Demo-key trigger set and per-image detail.",
         "Image statistics and the amplitude sweep."),
    ),
    PageSpec(
        "provenance", "Provenance", "Cryptography", ":material/verified:", "P9.12",
        "The published commitment, its independent Bitcoin timestamp, the signed record, and the theft timeline.",
        ("Commitment, model fingerprint, block attestation, record signature and tag status.",
         "The simulated theft, the thief's backdated counter-claim, and what is not symmetric between the two."),
    ),
    PageSpec(
        "zk", "Zero-knowledge", "Cryptography", ":material/lock:", "P9.13",
        "The two proof tracks: what each one proves, what it costs on this machine, and what it does not prove.",
        ("Track A: knowledge of an opening of the published commitment (Circom).",
         "Track B: inference of the small MNIST model (EZKL).",
         "The tampered proofs that were rejected."),
    ),
    PageSpec(
        "limits", "Honest limits", "Cryptography", ":material/report:", "P9.14",
        "Where the watermarks fail and what the cryptography does not cover, stated plainly.",
        ("Distillation, and channel pruning followed by fine-tuning.",
         "The single-contributor setup, the blocked trigger circuit, the unfinished ceremony check, the record's "
         "later timestamp."),
    ),
    PageSpec(
        "integrity", "Data integrity", "About", ":material/fact_check:", "P9.5",
        "Every file this dashboard may read, checked against the SHA-256 recorded when it was committed.",
        (),
    ),
)

SECTIONS: tuple[str, ...] = tuple(dict.fromkeys(p.section for p in PAGES))
