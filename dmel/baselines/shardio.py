"""Shared shard I/O for arm A tooling (runner, calibration, threshold sweep)."""

from __future__ import annotations

from pathlib import Path
from typing import Iterator

import numpy as np

from dmel.baselines.policy import FRAME_MS, FRAME_SAMPLES, BargeInPolicy


def load_mix(path: Path) -> np.ndarray:
    """Read a 16 kHz mono int16 mix."""
    return np.fromfile(path, dtype=np.int16)


def sample_id_of(mix_path: Path) -> str:
    """Sample id from a ``<sample_id>.mix.pcm`` path.

    ``Path.stem`` only strips one suffix (yielding ``<id>.mix``), so strip the
    full contract suffix explicitly.
    """
    name = mix_path.name
    if not name.endswith(".mix.pcm"):
        raise ValueError(f"{name}: expected <sample_id>.mix.pcm")
    return name[: -len(".mix.pcm")]


def load_labels_table(path: Path):
    """Read a labels parquet (lazy pyarrow import — policy-only consumers skip it)."""
    import pyarrow.parquet as pq

    return pq.read_table(path)


def labels_column(table, name: str) -> list:
    if name not in table.column_names:
        raise ValueError(f"labels parquet has no {name!r} column (found {table.column_names})")
    return table.column(name).to_pylist()


def agent_speaking_series(table) -> list[bool]:
    return [bool(v) for v in labels_column(table, "agent_speaking")]


def iter_decisions(
    mix: np.ndarray, agent_speaking: list[bool], *, threshold_ms: float
) -> Iterator[dict]:
    """Stream a sample through the policy; yield one decision record per frame.

    Records carry the frozen probs key (``p_stop``) plus per-step diagnostics
    from the policy attributes for the decision logs.
    """
    policy = BargeInPolicy(stop_threshold_ms=threshold_ms)
    frame_count = min(mix.size // FRAME_SAMPLES, len(agent_speaking))
    for index in range(frame_count):
        frame = mix[index * FRAME_SAMPLES : (index + 1) * FRAME_SAMPLES]
        result = policy.step(frame, None, agent_speaking=agent_speaking[index])
        yield {
            "t_ms": result["t_ms"],
            "action": result["action"],
            "p_stop": result["probs"]["p_stop"],
            "speech_prob": policy.last_speech_prob,
            "speech_ms": policy.last_speech_ms,
            "latched": policy.latched,
            "stop_threshold_ms": policy.stop_threshold_ms,
            "frame_ms": FRAME_MS,
        }
