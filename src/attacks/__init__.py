"""Removal attacks: prune, quantize, fine-tune, distill.

`harness` defines the attack interface and registry (P4.1); importing it
registers the ``none`` control. Each Phase 4 attack module registers itself
when imported here. `evaluation` scores an attacked model into the standard
row.
"""

from src.attacks import harness  # noqa: F401  -- registers the 'none' control
from src.attacks import prune  # noqa: F401  -- P4.2 magnitude pruning
from src.attacks import structured_prune  # noqa: F401  -- P4.3 structured (filter) pruning
from src.attacks import quantize  # noqa: F401  -- P4.4 post-training quantization
from src.attacks import finetune  # noqa: F401  -- P4.5 fine-tuning on the attacker holdout
from src.attacks import prune_finetune  # noqa: F401  -- P4.6 prune, then fine-tune with the mask fixed
from src.attacks import distill  # noqa: F401  -- P4.7 knowledge distillation into a fresh student
