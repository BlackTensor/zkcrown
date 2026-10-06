"""Put the repo root on `sys.path` so tests can `import src...` from anywhere."""

import sys
from pathlib import Path

ROOT = str(Path(__file__).resolve().parent)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# Downloaded third-party code (P9.4's torch.hub checkout under data/) ships its
# own tests; they are not this project's.
collect_ignore_glob = ["data/*"]
