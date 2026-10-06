"""The repo's own statistics code, as the dashboard uses it (P9.7, P9.9).

The dashboard computes derived numbers (a grade, a binomial null, the weight
bound) with the same functions the experiments used, never with copies. This
module is the only place the app loads code from ``src/``, and only these three
modules, none of which needs torch:

- `grading`: ``src/auditor/grading.py``, loaded by path, because importing the
  ``src.auditor`` package would load the auditor engine and torch.
- `significance`: ``src.watermark.significance`` (P2.8, the exact binomial test).
- `weight_significance`: ``src.watermark.weight_significance`` (P3.7, the
  proven bound). These two import numpy through ``src.watermark``; numpy is a
  dependency of Streamlit itself.
"""

from __future__ import annotations

import importlib
import importlib.util
from functools import lru_cache
from types import ModuleType

from app.data import REPO_ROOT

GRADING_SOURCE = "src/auditor/grading.py"
MODULES = ("src.watermark.significance", "src.watermark.weight_significance")


@lru_cache(maxsize=None)
def grading() -> ModuleType:
    spec = importlib.util.spec_from_file_location("zk_crown_auditor_grading", REPO_ROOT / GRADING_SOURCE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@lru_cache(maxsize=None)
def significance() -> ModuleType:
    return importlib.import_module(MODULES[0])


@lru_cache(maxsize=None)
def weight_significance() -> ModuleType:
    return importlib.import_module(MODULES[-1])
