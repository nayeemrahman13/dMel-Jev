"""Runner tests: synthetic contract shard → per-frame decision logs."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("pyarrow")

from dmel.baselines.policy import FRAME_SAMPLES
from dmel.baselines.runner import run_shard

LEAD_FRAMES = 10
SPEECH_FRAMES = 8


def _write_sample(shard: Path, sample_id: str, inject_pcm: np.ndarray) -> None:
    """mix.pcm = 500 ms silence then speech; agent speaks (flag) throughout."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    speech = np.zeros((LEAD_FRAMES + SPEECH_FRAMES) * FRAME_SAMPLES, dtype=np.int16)
    speech[
        LEAD_FRAMES * FRAME_SAMPLES : LEAD_FRAMES * FRAME_SAMPLES
        + min(inject_pcm.size, SPEECH_FRAMES * FRAME_SAMPLES)
    ] = inject_pcm[: SPEECH_FRAMES * FRAME_SAMPLES]
    speech.tofile(shard / f"{sample_id}.mix.pcm")

    frame_count = speech.size // FRAME_SAMPLES
    table = pa.table(
        {
            "frame_index": pa.array(range(frame_count), type=pa.int64()),
            "t_ms": pa.array([i * 50 for i in range(frame_count)], type=pa.int64()),
            "agent_speaking": pa.array([True] * frame_count, type=pa.bool_()),
        }
    )
    pq.write_table(table, shard / f"{sample_id}.labels.parquet")


def test_run_shard_well_formed_logs(tmp_path: Path, inject_pcm: np.ndarray) -> None:
    shard = tmp_path / "shard-00001"
    shard.mkdir()
    _write_sample(shard, "s1", inject_pcm)

    out = tmp_path / "out"
    written = run_shard(shard, out, threshold_ms=200.0)
    assert written == [out / "s1.armA.jsonl"]

    lines = [json.loads(line) for line in written[0].read_text().splitlines()]
    assert len(lines) == LEAD_FRAMES + SPEECH_FRAMES
    assert [r["t_ms"] for r in lines] == [50 * i for i in range(len(lines))]
    for record in lines:
        assert record["action"] in {"KEEP", "STOP_TTS"}
        assert 0.0 <= record["p_stop"] <= 1.0
        assert 0.0 <= record["speech_prob"] <= 1.0
        assert record["stop_threshold_ms"] == 200.0
        assert isinstance(record["latched"], bool)
    assert all(r["action"] == "KEEP" for r in lines[:LEAD_FRAMES])
    assert any(r["action"] == "STOP_TTS" for r in lines[LEAD_FRAMES:])
    # STOP frames saturate the frozen p_stop score; KEEP frames stay below 1.
    for record in lines:
        if record["action"] == "STOP_TTS":
            assert record["p_stop"] == 1.0


def test_run_shard_requires_labels(tmp_path: Path, inject_pcm: np.ndarray) -> None:
    shard = tmp_path / "shard-00002"
    shard.mkdir()
    inject_pcm[:FRAME_SAMPLES].tofile(shard / "orphan.mix.pcm")
    with pytest.raises(FileNotFoundError, match="labels"):
        run_shard(shard, tmp_path / "out", threshold_ms=200.0)


def test_cli_module_entrypoint(tmp_path: Path, repo_root: Path, inject_pcm: np.ndarray) -> None:
    shard = tmp_path / "shard-00003"
    shard.mkdir()
    _write_sample(shard, "s1", inject_pcm)

    env = {
        **os.environ,
        "PYTHONPATH": os.pathsep.join(
            [
                str(repo_root),
                str(repo_root / "apps" / "realtime"),
                str(repo_root / "packages" / "protocol" / "python"),
            ]
        ),
    }
    proc = subprocess.run(
        [sys.executable, "-m", "dmel.baselines.runner", "--shard", str(shard)],
        cwd=repo_root,
        env=env,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr
    summary = json.loads(proc.stdout.strip().splitlines()[-1])
    assert summary["sample"] == "s1"
    assert summary["frames"] == LEAD_FRAMES + SPEECH_FRAMES
    assert summary["stop_frames"] > 0
    assert (shard / "decisions" / "s1.armA.jsonl").is_file()
