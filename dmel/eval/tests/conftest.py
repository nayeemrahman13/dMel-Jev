"""Make ``dmel.eval`` importable from the repo root regardless of install mode.

dmel/ is an implicit namespace package (no __init__.py), so pytest's rootdir
insertion does not reliably cover it; pin the repo root explicitly.
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
