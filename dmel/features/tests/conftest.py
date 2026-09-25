"""Bootstrap the repo root onto sys.path.

dmel is a source-checkout implicit-namespace package resolved from the repo
root; a plain ``pytest dmel/features/tests`` invocation needs this because it
does not put the repo root on ``sys.path``.
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
