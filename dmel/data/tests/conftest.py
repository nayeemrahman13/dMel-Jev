"""Test bootstrap: make the repo root importable when pytest is invoked from
anywhere (dmel is an implicit namespace package — no __init__.py)."""

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
