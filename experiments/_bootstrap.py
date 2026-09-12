"""Put the repo root on `sys.path`.

Every script in `experiments/` imports this first::

    import _bootstrap  # noqa: F401

`python experiments/foo.py` adds `experiments/` to `sys.path` but not the repo
root, so `import src...` fails without this. Importing by name works because
this file sits next to the caller.

The alternative -- making the repo pip-installable -- would mean an extra
install step inside every Colab session for no benefit, since notebooks already
`chdir` into the checkout.

`tests/` gets the same treatment from the root `conftest.py`.
"""

import sys
from pathlib import Path

_ROOT = str(Path(__file__).resolve().parents[1])
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
