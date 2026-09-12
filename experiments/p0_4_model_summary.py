"""P0.4: report main_model's parameter counts.

CLAUDE.md 0.3 -- every number that appears in the repo has to come from a
script in `experiments/`. The parameter count in section 8.2 comes from here.

This measures architecture, not performance. Clean accuracy is P0.5/P0.6 and
needs a GPU; nothing here touches a dataset.

    python experiments/p0_4_model_summary.py
"""

from __future__ import annotations

import argparse

import _bootstrap  # noqa: F401  -- puts the repo root on sys.path
import torch

from src.models import CIFAR10_INPUT_SHAPE, count_parameters, main_model
from src.utils.results import write_result
from src.utils.seeding import DEFAULT_SEED, set_seed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--width", type=int, default=32)
    parser.add_argument(
        "--student-width",
        type=int,
        default=16,
        help="width of the P4.7 distillation student, reported for comparison",
    )
    args = parser.parse_args()

    backends = set_seed(args.seed)

    model = main_model(width=args.width)
    counts = count_parameters(model)
    student_counts = count_parameters(main_model(width=args.student_width))

    # Confirm the forward pass works at the documented input shape, so the
    # reported counts describe a model that actually runs.
    model.eval()
    with torch.no_grad():
        logits = model(torch.zeros(2, *CIFAR10_INPUT_SHAPE))
    assert logits.shape == (2, model.num_classes), logits.shape

    layer_params = {
        name: tuple(param.shape) for name, param in model.named_parameters()
    }

    metrics = {
        "total_params": counts["total"],
        "trainable_params": counts["trainable"],
        "conv_and_linear_weights": counts["conv_and_linear_weights"],
        "student_total_params": student_counts["total"],
        "output_shape": list(logits.shape),
        "state_dict_entries": len(model.state_dict()),
    }
    params = {
        "architecture": "MainModel",
        "width": args.width,
        "student_width": args.student_width,
        "dataset": "CIFAR-10",
        "input_shape": list(CIFAR10_INPUT_SHAPE),
        "num_classes": model.num_classes,
        "layer_shapes": {k: list(v) for k, v in layer_params.items()},
    }

    path = write_result(
        name="p0.4_main_model_summary",
        seed=args.seed,
        task="P0.4",
        params=params,
        metrics=metrics,
        seeded_backends=backends,
        notes="Architecture only. No training, no dataset, no accuracy measured.",
    )

    print(model)
    print()
    for key, value in metrics.items():
        print(f"{key:>26} : {value}")
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
