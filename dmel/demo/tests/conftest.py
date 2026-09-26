"""Path bootstrap for the dmel.demo tests.

A plain ``pytest dmel/demo/tests`` invocation does not put the repo root on
``sys.path`` — add it so ``import dmel.demo...`` and the sibling ``dmel.*``
areas resolve regardless of how pytest was invoked.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


@pytest.fixture(scope="session")
def fixtures_dir() -> Path:
    """The committed pilot-fixture samples (stems, labels, dMel caches)."""
    return REPO_ROOT / "dmel" / "data" / "fixtures"


@pytest.fixture(scope="session")
def checkpoint_path() -> Path:
    return REPO_ROOT / "dmel" / "demo" / "checkpoints" / "w256_seed0_best.pt"
