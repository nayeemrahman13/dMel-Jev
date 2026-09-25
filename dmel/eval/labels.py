"""Shard/sample discovery and label/audio loading for the dMel eval harness.

Consumes the contract shard layout (data engine's output format):
``shard-XXXXX/manifest.json``, ``<sample_id>.{user,agent,mix}.pcm`` and
``<sample_id>.labels.parquet``. Nothing here depends on other dmel areas' code;
the dMel feature cache (``<sample_id>.dmel.npy``) is consumed by path only, with
tolerant shape handling since the features area owns its internal format.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

PCM_SAMPLE_RATE = 16_000
SAMPLES_PER_FRAME = 800  # 16 kHz * 50 ms hop


@dataclass(frozen=True)
class SamplePaths:
    """File locations for one synthesized conversation clip."""

    sample_id: str
    shard_dir: Path
    labels: Path
    user_pcm: Path | None
    agent_pcm: Path | None
    dmel_npy: Path | None


def discover_samples(root: str | Path) -> list[SamplePaths]:
    """Find samples under ``root``.

    ``root`` may be a single shard/fixtures directory (contains
    ``*.labels.parquet`` directly) or a parent directory of shard directories.
    Raises FileNotFoundError when nothing loadable is found.
    """
    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(f"shard dir does not exist: {root}")
    shard_dirs = _shard_dirs(root)
    samples: list[SamplePaths] = []
    for shard_dir in shard_dirs:
        for labels_path in sorted(shard_dir.glob("*.labels.parquet")):
            sample_id = labels_path.name[: -len(".labels.parquet")]
            samples.append(
                SamplePaths(
                    sample_id=sample_id,
                    shard_dir=shard_dir,
                    labels=labels_path,
                    user_pcm=_existing(shard_dir / f"{sample_id}.user.pcm"),
                    agent_pcm=_existing(shard_dir / f"{sample_id}.agent.pcm"),
                    dmel_npy=_existing(shard_dir / f"{sample_id}.dmel.npy"),
                )
            )
    if not samples:
        raise FileNotFoundError(f"no *.labels.parquet samples found under {root}")
    return samples


def _shard_dirs(root: Path) -> list[Path]:
    if any(root.glob("*.labels.parquet")):
        return [root]
    subdirs = sorted(d for d in root.iterdir() if d.is_dir() and any(d.glob("*.labels.parquet")))
    if not subdirs:
        raise FileNotFoundError(f"no shard dirs with *.labels.parquet under {root}")
    return subdirs


def _existing(path: Path) -> Path | None:
    return path if path.is_file() else None


def load_labels(path: str | Path) -> pd.DataFrame:
    """Read one sample's per-frame labels parquet into a DataFrame, time-sorted."""
    import pyarrow.parquet as pq

    df = pq.read_table(path).to_pandas()
    if "frame_index" in df.columns:
        df = df.sort_values("frame_index").reset_index(drop=True)
    return df


def read_pcm(path: str | Path) -> np.ndarray:
    """Read raw int16 mono PCM (16 kHz)."""
    return np.fromfile(path, dtype=np.int16)


def frame_view(pcm: np.ndarray, index: int) -> np.ndarray:
    """The ``index``-th 800-sample (50 ms) frame, zero-padded if the audio ends short."""
    frame = pcm[index * SAMPLES_PER_FRAME : (index + 1) * SAMPLES_PER_FRAME]
    if len(frame) < SAMPLES_PER_FRAME:
        frame = np.pad(frame, (0, SAMPLES_PER_FRAME - len(frame)))
    return frame


def load_dmel_steps(path: Path | None, n_steps: int) -> tuple[list[np.ndarray | None], str | None]:
    """Per-step dMel token ids from a features-area cache, or Nones.

    Returns ``(steps, warning)`` where ``warning`` explains a fully-None result.
    The cache is expected to be an array whose first axis is the 50 ms step;
    anything else is unusable without features-area knowledge, so the runner
    feeds ``None`` token ids (contract: policies must accept None).
    """
    if path is None:
        return [None] * n_steps, None
    arr = np.load(path, allow_pickle=False)
    if arr.ndim < 1 or arr.shape[0] < n_steps:
        return [None] * n_steps, (
            f"{path.name}: dmel cache shape {arr.shape} does not cover {n_steps} steps; "
            "passing dmel_token_ids=None"
        )
    return [np.asarray(arr[i]) for i in range(n_steps)], None
