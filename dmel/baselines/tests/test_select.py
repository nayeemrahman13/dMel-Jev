"""Validation-split threshold selection tests (contract v1 protocol)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("pyarrow")

from dmel.baselines.policy import FRAME_SAMPLES
from dmel.baselines.select import manifest_splits, select_operating_point

LEAD_FRAMES = 10
SPEECH_FRAMES = 8


def _write_val_sample(shard: Path, sample_id: str, inject_pcm: np.ndarray) -> None:
    """Labels: the stop span covers exactly the user speech frames."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    speech = np.zeros((LEAD_FRAMES + SPEECH_FRAMES) * FRAME_SAMPLES, dtype=np.int16)
    speech[LEAD_FRAMES * FRAME_SAMPLES :] = inject_pcm[: SPEECH_FRAMES * FRAME_SAMPLES]
    speech.tofile(shard / f"{sample_id}.mix.pcm")

    frame_count = speech.size // FRAME_SAMPLES
    table = pa.table(
        {
            "frame_index": pa.array(range(frame_count), type=pa.int64()),
            "t_ms": pa.array([i * 50 for i in range(frame_count)], type=pa.int64()),
            "agent_speaking": pa.array([True] * frame_count, type=pa.bool_()),
            "action": pa.array(["KEEP"] * LEAD_FRAMES + ["STOP_TTS"] * SPEECH_FRAMES),
        }
    )
    pq.write_table(table, shard / f"{sample_id}.labels.parquet")


def _write_manifest(shard: Path, payload: dict) -> None:
    (shard / "manifest.json").write_text(json.dumps(payload), encoding="utf-8")


@pytest.fixture(scope="module")
def selected(tmp_path_factory: pytest.TempPathFactory, inject_pcm: "np.ndarray") -> dict:
    shard = tmp_path_factory.mktemp("shard-sel")
    _write_val_sample(shard, "v1", inject_pcm)
    _write_val_sample(shard, "t1", inject_pcm)  # test split — must not enter the sweep
    _write_manifest(shard, {"samples": {"v1": {"split": "val"}, "t1": {"split": "test"}}})
    return select_operating_point(shard, manifest_path=None, thresholds_ms=[50.0, 100.0, 600.0])


def test_selection_uses_validation_only(selected: dict) -> None:
    assert selected["samples"] == ["v1"]
    assert selected["split"] == "val"


def test_selected_point_beats_never_stop(selected: dict) -> None:
    # A 600 ms gate never fires on an 8-speech-frame span; F-beta must be 0
    # there and the selected point must beat it.
    by_threshold = {c["threshold_ms"]: c for c in selected["candidates"]}
    assert by_threshold[600.0]["f_beta"] == 0.0
    assert selected["selected_threshold_ms"] <= 100.0
    assert selected["selected"]["f_beta"] >= 0.5
    assert selected["selected"]["recall"] >= 0.5


def test_manifest_list_shape_and_missing_split(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps([{"id": "a", "split": "train"}, {"id": "b", "split": "val"}]))
    assert manifest_splits(manifest) == {"a": "train", "b": "val"}

    manifest.write_text(json.dumps({"samples": [{"id": "a"}]}))
    with pytest.raises(ValueError, match="split"):
        manifest_splits(manifest)


def test_no_val_samples_raises(tmp_path: Path, inject_pcm: np.ndarray) -> None:
    shard = tmp_path / "shard-noval"
    shard.mkdir()
    _write_val_sample(shard, "t1", inject_pcm)
    _write_manifest(shard, {"samples": {"t1": {"split": "test"}}})
    with pytest.raises(ValueError, match="validation"):
        select_operating_point(shard, manifest_path=None, thresholds_ms=[50.0])


def test_no_stop_labels_raises(tmp_path: Path, inject_pcm: np.ndarray) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    shard = tmp_path / "shard-nolabelstop"
    shard.mkdir()
    sample = tmp_path / "silence.mix.pcm"
    np.zeros(FRAME_SAMPLES * 4, dtype=np.int16).tofile(sample)
    (shard / "v1.mix.pcm").write_bytes(sample.read_bytes())
    table = pa.table(
        {
            "frame_index": pa.array(range(4), type=pa.int64()),
            "t_ms": pa.array([0, 50, 100, 150], type=pa.int64()),
            "agent_speaking": pa.array([True] * 4, type=pa.bool_()),
            "action": pa.array(["KEEP"] * 4),
        }
    )
    pq.write_table(table, shard / "v1.labels.parquet")
    _write_manifest(shard, {"samples": {"v1": {"split": "val"}}})
    with pytest.raises(ValueError, match="STOP_TTS"):
        select_operating_point(shard, manifest_path=None, thresholds_ms=[50.0])
