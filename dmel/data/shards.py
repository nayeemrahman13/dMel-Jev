"""Shard IO: manifests, raw PCM stems, and per-frame label parquet files.

Shard layout (contract):

    shard-XXXXX/manifest.json
    shard-XXXXX/<sample_id>.{user,agent,mix}.pcm
    shard-XXXXX/<sample_id>.labels.parquet

Parquet tables are written through pyarrow with a fixed schema and no
embedded pandas metadata, keeping outputs byte-identical across runs.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from dmel.data.annotation import LABEL_COLUMNS

_LABEL_SCHEMA = pa.schema(
    [
        ("frame_index", pa.int32()),
        ("t_ms", pa.int32()),
        ("agent_speaking", pa.int8()),
        ("speech_present", pa.int8()),
        ("primary_user", pa.int8()),
        ("backchannel", pa.int8()),
        ("interrupt_intent", pa.int8()),
        ("action", pa.string()),
        ("earliest_reasonable_stop_ms", pa.int32()),
        ("overlap_onset_ms", pa.int32()),
        ("scenario", pa.string()),
    ]
)


def write_pcm(path: Path, pcm: np.ndarray) -> None:
    """Write int16 mono PCM (raw, headerless)."""
    path.write_bytes(np.ascontiguousarray(pcm, dtype=np.int16).tobytes())


def write_labels_parquet(path: Path, columns: dict[str, np.ndarray]) -> None:
    """Write one row per 50 ms frame with the fixed contract schema."""
    arrays = [
        pa.array(columns[name], type=_LABEL_SCHEMA.field(name).type)
        for name in LABEL_COLUMNS
    ]
    table = pa.Table.from_arrays(arrays, schema=_LABEL_SCHEMA)
    pq.write_table(table, path, compression="snappy")


def write_manifest(path: Path, manifest: dict) -> None:
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")


def shard_dir(root: Path, shard_index: int) -> Path:
    return root / f"shard-{shard_index:05d}"
