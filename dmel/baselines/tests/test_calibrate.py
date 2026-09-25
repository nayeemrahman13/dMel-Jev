"""Calibration tool tests on a synthetic contract-format shard."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("pyarrow")

from dmel.baselines.calibrate import calibrate_shard, markdown_table
from dmel.baselines.policy import FRAME_SAMPLES

LEAD_FRAMES = 10
SPEECH_FRAMES = 8


def _write_sample(shard: Path, sample_id: str, inject_pcm: np.ndarray) -> None:
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
            # v1 acoustic reference: user/background speech, agent excluded.
            "speech_present": pa.array([False] * LEAD_FRAMES + [True] * SPEECH_FRAMES),
            "scenario": pa.array(["interruption"] * frame_count),
        }
    )
    pq.write_table(table, shard / f"{sample_id}.labels.parquet")


@pytest.fixture(scope="module")
def calibrated(tmp_path_factory: pytest.TempPathFactory, inject_pcm: "np.ndarray") -> dict:
    shard = tmp_path_factory.mktemp("shard-cal")
    _write_sample(shard, "c1", inject_pcm)
    return calibrate_shard(shard, prob_threshold=0.5)


def test_report_structure(calibrated: dict) -> None:
    assert calibrated["prob_threshold"] == 0.5
    assert set(calibrated["scenarios"]) == {"interruption"}
    for section in (calibrated["scenarios"]["interruption"], calibrated["overall"]):
        assert set(section) >= {"tp", "fp", "fn", "tn", "precision", "recall", "f1"}
        assert 0.0 <= section["precision"] <= 1.0
        assert 0.0 <= section["recall"] <= 1.0
        assert 0.0 <= section["f1"] <= 1.0


def test_reference_counts_sum_to_frame_count(calibrated: dict) -> None:
    overall = calibrated["overall"]
    # One sample, 18 frames: every frame lands in exactly one confusion bucket.
    assert overall["tp"] + overall["fp"] + overall["fn"] + overall["tn"] == LEAD_FRAMES + SPEECH_FRAMES


def test_speech_frames_are_detected(calibrated: dict) -> None:
    overall = calibrated["overall"]
    assert overall["recall"] >= 0.5, calibrated
    assert overall["tp"] > 0
    # The kokoro flag must be consistent with the recall it is derived from.
    assert calibrated["kokoro_slice_required"] == (calibrated["user_speech_recall"] < 0.9)


def test_markdown_renders_rows(calibrated: dict) -> None:
    table = markdown_table(calibrated)
    assert "| scenario |" in table
    assert "| interruption |" in table
    assert "| **overall** |" in table


def test_missing_scenario_column_raises(tmp_path: Path, inject_pcm: np.ndarray) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    shard = tmp_path / "shard-noscn"
    shard.mkdir()
    _write_sample(shard, "c1", inject_pcm)
    labels_path = shard / "c1.labels.parquet"
    table = pa.table({"speech_present": pa.array([True, False])})
    pq.write_table(table, labels_path)
    with pytest.raises(ValueError, match="scenario"):
        calibrate_shard(shard, prob_threshold=0.5)


def test_json_serializable(calibrated: dict, tmp_path: Path) -> None:
    (tmp_path / "r.json").write_text(json.dumps(calibrated))
    assert True
