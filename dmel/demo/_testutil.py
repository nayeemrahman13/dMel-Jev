"""Shared helpers for the dmel.demo tests (synthetic clips, fixture loaders).

Fixture readers reuse the EVAL harness's own loaders (``dmel.eval.labels``)
so the loopback comparisons exercise exactly the offline code path rather
than a test-local reimplementation of it.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from dmel.demo.frontend import FRAME_SAMPLES
from dmel.eval.labels import load_labels, read_pcm
from dmel.features.tests._synth import sine, silence


def clip_with_burst() -> np.ndarray:
    """A short clip whose second half carries a loud sine burst.

    Shape-shifting audio (silence -> burst) exercises both the frontend's
    steady-state and transient windows in streaming/offline comparisons.
    """
    return np.concatenate([silence(400.0), sine(220.0, 400.0, amp=0.5)])


def fixture_mix(fixtures_dir: Path, sample_id: str) -> np.ndarray:
    """The fixture's mix stem (agent+user+background — demo-mic semantics)."""
    return read_pcm(fixtures_dir / f"{sample_id}.mix.pcm")


def fixture_cache(fixtures_dir: Path, sample_id: str) -> np.ndarray:
    """The fixture's committed dMel token cache (tokenized from the mix stem)."""
    return np.load(fixtures_dir / f"{sample_id}.dmel.npy", allow_pickle=False)


def fixture_flags(fixtures_dir: Path, sample_id: str) -> list[bool]:
    """Per-50ms-step agent_speaking flags from the fixture's labels parquet."""
    labels = load_labels(fixtures_dir / f"{sample_id}.labels.parquet")
    return [bool(value) for value in labels["agent_speaking"]]
