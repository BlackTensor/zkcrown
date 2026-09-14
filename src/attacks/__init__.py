"""Removal attacks: prune, quantize, fine-tune, distill.

`harness` defines the attack interface and registry (P4.1); importing it
registers the ``none`` control. Each Phase 4 attack module registers itself
when imported here. `evaluation` scores an attacked model into the standard
row.
"""

from src.attacks import harness  # noqa: F401  -- registers the 'none' control
from src.attacks import prune  # noqa: F401  -- P4.2 magnitude pruning
