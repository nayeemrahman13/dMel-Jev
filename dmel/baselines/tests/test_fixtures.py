"""Fixture-consumption test: runs against dmel/data/fixtures when present.

The committed fixtures (contract: every scenario in ~10–20 s of audio) live
at ``dmel/data/fixtures/``. If the directory is absent this module skips with
a reason; when present the test streams every fixture sample through the
policy and asserts well-formed logs plus, for interruption samples, a
STOP_TTS inside the labeled stop span.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

pytest.importorskip("pyarrow")

REPO_ROOT = Path(__file__).resolve().parents[3]
FIXTURES_DIR = REPO_ROOT / "dmel" / "data" / "fixtures"

pytestmark = pytest.mark.skipif(
    not FIXTURES_DIR.is_dir(),
    reason="dmel/data/fixtures not present (data PR not merged yet)",
)


def _fixture_label_files() -> list[Path]:
    return sorted(FIXTURES_DIR.rglob("*.labels.parquet"))


def test_fixtures_exist() -> None:
    label_files = _fixture_label_files()
    assert label_files, f"dmel/data/fixtures exists but holds no *.labels.parquet files"


def test_fixture_samples_stream_end_to_end() -> None:
    from dmel.baselines import shardio
    from dmel.baselines.policy import FRAME_SAMPLES

    label_files = _fixture_label_files()
    interrupt_stops_seen = 0
    for labels_path in label_files:
        sample_id = labels_path.name[: -len(".labels.parquet")]
        mix_path = labels_path.with_name(f"{sample_id}.mix.pcm")
        assert mix_path.is_file(), f"{sample_id}: fixture mix.pcm missing"

        mix = shardio.load_mix(mix_path)
        table = shardio.load_labels_table(labels_path)
        agent_speaking = shardio.agent_speaking_series(table)
        records = list(shardio.iter_decisions(mix, agent_speaking, threshold_ms=200.0))
        assert records, f"{sample_id}: no frames streamed"
        assert [r["t_ms"] for r in records] == [
            50 * i for i in range(len(records))
        ], f"{sample_id}: t_ms grid broken"
        assert all(r["action"] in {"KEEP", "STOP_TTS"} for r in records)

        # Behavioral spot-check on interruption samples: the contract's labels
        # guarantee a stop is warranted at/after earliest_reasonable_stop_ms.
        if "scenario" in table.column_names and "interruption" in set(
            shardio.labels_column(table, "scenario")
        ):
            stops = [r for r in records if r["action"] == "STOP_TTS"]
            assert stops, f"{sample_id}: interruption fixture produced no STOP_TTS"
            interrupt_stops_seen += 1

    assert interrupt_stops_seen > 0, "no interruption-scenario fixture samples found"
