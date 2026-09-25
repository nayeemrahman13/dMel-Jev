import sys
from pathlib import Path

# Make the repo root importable (implicit `dmel` namespace) regardless of how
# pytest is invoked.
REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
