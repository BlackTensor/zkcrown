"""The attack interface every Phase 4 attack plugs into (P4.1).

An attack takes the thief's copy of a model and returns modified weights. The
harness (`experiments/run_attack_suite.py`) applies it from a config, then
scores the result with `src.attacks.evaluation` and writes one standard row.

The contract
------------
An attack is a function registered under a name::

    @register_attack("magnitude_prune", strength="sparsity", description="...")
    def magnitude_prune(state_dict, arch, strength, params, context) -> AttackOutput: ...

- ``state_dict``: a private copy of the source weights. The attack may change
  it in place or build a new one.
- ``arch``: the `main_model` constructor arguments of the source, for example
  ``{"width": 32}``.
- ``strength``: one number, the knob the attack sweeps (sparsity, epochs,
  learning rate, ...). Its meaning is the registered ``strength`` name.
  Everything else goes in ``params``.
- ``context``: an `AttackContext` with the device, seed and data location.

**An attack never receives `K`.** `AttackContext` has no key field, and the
function signature has no slot for one. The attacker in this threat model does
not know `K` (P2.8, P3.7 assumptions). The owner's key enters only in
evaluation, which runs locally. That split also keeps `K` off Colab: a `[GPU]`
attack is run there with ``run_attack_suite.py apply``, which needs no key,
and its saved weights are scored here with ``evaluate``.

The attack returns an `AttackOutput`: the attacked ``state_dict``, the
``arch`` it loads into (a distilled student may be narrower), and ``info``,
aggregate facts about what the attack did (achieved sparsity, epochs run,
final training loss). `apply_attack` checks that the weights load strictly
into ``main_model(**arch)``, so every row describes a model that exists.

Not decided here: how each attack works. P4.2 to P4.8 register their own
functions. The only attack registered by P4.1 is ``none``, the control: it
returns the weights unchanged, and its row must reproduce the unattacked
numbers.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch

from src.models import main_model
from src.utils.seeding import set_seed

ROW_VERSION = "attack-row/v1"
ALLOWED_ARCH_KEYS = frozenset({"width"})
"""`main_model` arguments an attack may set. Dropout does not change eval-mode outputs."""


@dataclass(frozen=True)
class AttackConfig:
    """One attack at one strength.

    Attributes:
        attack: registered attack name.
        strength: the swept value, finite.
        params: every other setting, JSON-serialisable. Part of the record.
    """

    attack: str
    strength: float
    params: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.attack, str) or not self.attack:
            raise ValueError("attack must be a non-empty string")
        if isinstance(self.strength, bool) or not isinstance(self.strength, (int, float)):
            raise TypeError(f"strength must be a number, got {type(self.strength).__name__}")
        if not math.isfinite(self.strength):
            raise ValueError(f"strength must be finite, got {self.strength}")
        if not isinstance(self.params, dict):
            raise TypeError("params must be a dict")
        object.__setattr__(self, "strength", float(self.strength))

    def label(self) -> str:
        """Short, filename-friendly name, e.g. ``magnitude_prune_0.5``."""
        return f"{self.attack}_{self.strength:g}"

    def to_dict(self) -> dict[str, Any]:
        return {"attack": self.attack, "strength": self.strength, "params": dict(self.params)}


def expand_config(config: Mapping[str, Any]) -> list[AttackConfig]:
    """Read a config dict into one `AttackConfig` per strength.

    Accepted shape::

        {"attack": "magnitude_prune", "strengths": [0.1, 0.5, 0.9], "params": {...}}

    ``"strength": x`` is accepted in place of ``"strengths"``. Unknown keys are
    refused, so a typo cannot silently drop a setting.
    """
    unknown = set(config) - {"attack", "strength", "strengths", "params"}
    if unknown:
        raise ValueError(f"unknown config keys: {sorted(unknown)}")
    if ("strength" in config) == ("strengths" in config):
        raise ValueError("give exactly one of 'strength' or 'strengths'")
    strengths = [config["strength"]] if "strength" in config else list(config["strengths"])
    if not strengths:
        raise ValueError("'strengths' is empty")
    if len(set(strengths)) != len(strengths):
        raise ValueError("'strengths' has duplicates")
    params = dict(config.get("params") or {})
    return [AttackConfig(attack=config["attack"], strength=s, params=params) for s in strengths]


@dataclass(frozen=True)
class AttackContext:
    """What an attack may use besides the weights. Deliberately no key."""

    device: torch.device
    seed: int
    data_root: Path
    num_workers: int = 0


@dataclass
class AttackOutput:
    """The attacked model.

    Attributes:
        state_dict: weights that load strictly into ``main_model(**arch)``.
        arch: `main_model` arguments, a subset of ``ALLOWED_ARCH_KEYS``.
        info: aggregate facts about the attack, JSON-serialisable. No weights.
        runtime_model: optional (P4.4). The model as the thief actually runs
            it, when loading ``state_dict`` into `main_model` does not
            reproduce its outputs, for example with quantized activations.
            If set, accuracy and the behavioral watermark are scored on it,
            while weight extraction still reads ``state_dict``, the shipped
            weights in the owner's layout. Such an output cannot be saved and
            re-scored from its weights, so ``run_attack_suite.py apply``
            refuses it.
    """

    state_dict: dict[str, torch.Tensor]
    arch: dict[str, Any]
    info: dict[str, Any] = field(default_factory=dict)
    runtime_model: torch.nn.Module | None = None


AttackFn = Callable[[dict, dict, float, dict, AttackContext], AttackOutput]


@dataclass(frozen=True)
class AttackSpec:
    name: str
    fn: AttackFn
    strength: str
    description: str

    def to_dict(self) -> dict[str, str]:
        return {"name": self.name, "strength_meaning": self.strength, "description": self.description}


_REGISTRY: dict[str, AttackSpec] = {}


def register_attack(name: str, *, strength: str, description: str) -> Callable[[AttackFn], AttackFn]:
    """Decorator: register `fn` under `name`. Names are unique."""

    def decorate(fn: AttackFn) -> AttackFn:
        if not name or not name.replace("_", "").isalnum() or name.lower() != name:
            raise ValueError(f"attack name {name!r} must be lower-case letters, digits and underscores")
        if name in _REGISTRY:
            raise ValueError(f"attack {name!r} is already registered")
        _REGISTRY[name] = AttackSpec(name=name, fn=fn, strength=strength, description=description)
        return fn

    return decorate


def get_attack(name: str) -> AttackSpec:
    if name not in _REGISTRY:
        raise KeyError(f"no attack named {name!r}; registered: {sorted(_REGISTRY)}")
    return _REGISTRY[name]


def available_attacks() -> list[str]:
    return sorted(_REGISTRY)


def clone_state(state_dict: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Detached CPU copy that shares no storage with the input."""
    return {name: tensor.detach().to("cpu").clone() for name, tensor in state_dict.items()}


def load_model(state_dict: Mapping[str, torch.Tensor], arch: Mapping[str, Any]) -> torch.nn.Module:
    """``main_model(**arch)`` with `state_dict` loaded strictly, in eval mode."""
    unknown = set(arch) - ALLOWED_ARCH_KEYS
    if unknown:
        raise ValueError(f"arch has keys an attack may not set: {sorted(unknown)}")
    model = main_model(**arch)
    model.load_state_dict(state_dict, strict=True)
    return model.eval()


def apply_attack(
    config: AttackConfig,
    source_state: Mapping[str, torch.Tensor],
    source_arch: Mapping[str, Any],
    context: AttackContext,
) -> AttackOutput:
    """Run `config` on a private copy of the source weights.

    Seeds every backend with ``context.seed`` first, so a rerun of the same
    config gives the same weights. The source is never modified. The output
    is checked to load strictly into its declared architecture.
    """
    spec = get_attack(config.attack)
    set_seed(context.seed)
    output = spec.fn(clone_state(source_state), dict(source_arch), config.strength, dict(config.params), context)
    if not isinstance(output, AttackOutput):
        raise TypeError(f"attack {config.attack!r} returned {type(output).__name__}, expected AttackOutput")
    output.state_dict = clone_state(output.state_dict)
    load_model(output.state_dict, output.arch)
    if output.runtime_model is not None:
        if not isinstance(output.runtime_model, torch.nn.Module):
            raise TypeError(f"attack {config.attack!r} returned a runtime_model that is not an nn.Module")
        output.runtime_model.eval()
    return output


@register_attack(
    "none",
    strength="unused; must be 0",
    description="Control. Returns the source weights unchanged; its row must reproduce the unattacked model.",
)
def no_attack(state_dict, arch, strength, params, context) -> AttackOutput:
    if strength != 0 or params:
        raise ValueError("the 'none' attack takes strength 0 and no params")
    return AttackOutput(state_dict=state_dict, arch=arch, info={"changed_tensors": 0})
