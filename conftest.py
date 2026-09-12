"""Put the repo root on `sys.path` so tests can `import src...` from anywhere."""

import sys
from pathlib import Path

ROOT = str(Path(__file__).resolve().parent)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
