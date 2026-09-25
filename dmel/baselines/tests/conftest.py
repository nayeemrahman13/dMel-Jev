"""Path bootstrap for the dmel.baselines tests.

A plain ``pytest dmel/baselines/tests`` invocation does not put the repo root
on ``sys.path`` — add it so ``import dmel.baselines...`` and the vendored
``dmel.runtime`` pieces both resolve regardless of how pytest was invoked.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


@pytest.fixture(scope="session")
def repo_root() -> Path:
    return REPO_ROOT


@pytest.fixture(scope="session")
def fixtures_dir() -> Path:
    return REPO_ROOT / "dmel" / "data" / "fixtures"


@pytest.fixture(scope="session")
def inject_pcm() -> "np.ndarray":
    """The committed scripted barge-in clip ('hey stop') as int16 PCM."""
    pytest.importorskip("onnxruntime")
    import numpy as np

    from dmel.runtime.audio import Pcm
    from dmel.runtime.config import RealtimeConfig

    return Pcm.load_wav_i16(RealtimeConfig.INJECT_WAV)
