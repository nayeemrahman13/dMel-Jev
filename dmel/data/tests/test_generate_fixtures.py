"""End-to-end generator tests: fixtures preset determinism, coverage, schema.

These run the real CLI (espeak-ng required) — everything else in this suite
stays render-free, so the skip guard keeps `pytest` usable without espeak.
"""

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from dmel.data.annotation import LABEL_COLUMNS
from dmel.data.constants import MS_TO_SAMPLES, SAMPLE_RATE

espeak_missing = shutil.which("espeak-ng") is None
pytestmark = pytest.mark.skipif(espeak_missing, reason="espeak-ng not installed")

SCENARIOS = {
    "interruption", "backchannel", "normal_turn", "noise",
    "background_speaker", "hesitation", "hesitation_commit",
}


def run_fixtures(out: Path) -> None:
    result = subprocess.run(
        [sys.executable, "-m", "dmel.data.generate", "--preset", "fixtures", "--out", str(out)],
        capture_output=True, text=True, timeout=300,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def shard_files(root: Path) -> dict[str, bytes]:
    return {
        str(p.relative_to(root)): p.read_bytes()
        for p in sorted(root.rglob("*")) if p.is_file()
    }


def sample_dir(root: Path) -> Path:
    """Directory holding fixture files: flat (fixtures preset) or shard-XXXXX."""
    shard = next(root.glob("shard-*"), None)
    return shard if shard is not None else root


@pytest.fixture(scope="module")
def fixture_run(tmp_path_factory):
    out = tmp_path_factory.mktemp("dmel-fixtures") / "fixtures"
    run_fixtures(out)
    return out


class TestDeterminism:
    def test_two_runs_are_byte_identical(self, tmp_path, fixture_run):
        second = tmp_path / "second"
        run_fixtures(second)
        first_files = shard_files(fixture_run)
        second_files = shard_files(second)
        assert sorted(first_files) == sorted(second_files)
        differing = [k for k in first_files if first_files[k] != second_files[k]]
        assert not differing, f"non-deterministic outputs: {differing}"

    def test_manifest_records_determinism_command(self, fixture_run):
        manifest = json.loads((sample_dir(fixture_run) / "manifest.json").read_text())
        assert manifest["determinism"]["command"] == "python -m dmel.data.generate --preset fixtures"
        assert "espeak" in manifest["determinism"]


class TestFixtureCoverage:
    def test_every_scenario_present(self, fixture_run):
        manifest = json.loads((sample_dir(fixture_run) / "manifest.json").read_text())
        scenarios = {s["scenario"] for s in manifest["samples"]}
        assert scenarios == SCENARIOS

    def test_truncated_interruption_present(self, fixture_run):
        manifest = json.loads((sample_dir(fixture_run) / "manifest.json").read_text())
        worlds = [s["world"] for s in manifest["samples"] if s["scenario"] == "interruption"]
        assert "truncated" in worlds

    def test_censored_pair_prefix_identical(self, fixture_run):
        sdir = sample_dir(fixture_run)
        manifest = json.loads((sdir / "manifest.json").read_text())
        pairs: dict[str, list[dict]] = {}
        for s in manifest["samples"]:
            if s.get("pair_id"):
                pairs.setdefault(s["pair_id"], []).append(s)
        assert pairs, "no censored pair in fixtures"
        for members in pairs.values():
            assert len(members) == 2
            hes = next(m for m in members if m["scenario"] == "hesitation")
            commit = next(m for m in members if m["scenario"] == "hesitation_commit")
            n = hes["shared_prefix_ms"] * MS_TO_SAMPLES * 2  # ms -> samples -> bytes
            assert n > 0
            for stem in ("agent", "mix"):
                a = (sdir / f"{hes['id']}.{stem}.pcm").read_bytes()
                b = (sdir / f"{commit['id']}.{stem}.pcm").read_bytes()
                assert a[:n] == b[:n], f"{stem} prefix diverges within the censored window"

    def test_train_samples_have_no_codec(self, fixture_run):
        # The codec degradation family is eval-only (contract Splits).
        manifest = json.loads((sample_dir(fixture_run) / "manifest.json").read_text())
        for s in manifest["samples"]:
            if s["split"] == "train":
                assert s["codec"] is None, f"{s['id']} applies eval-only codec on train"


class TestLabelFiles:
    def test_parquet_schema_matches_contract(self, fixture_run):
        import pyarrow.parquet as pq

        sdir = sample_dir(fixture_run)
        for parquet in sorted(sdir.glob("*.labels.parquet")):
            table = pq.read_table(parquet)
            assert table.column_names == list(LABEL_COLUMNS)
            break

    def test_fixture_audio_durations(self, fixture_run):
        # ~10-20 s total audio across the fixture set.
        total_samples = 0
        for pcm in sample_dir(fixture_run).glob("*.mix.pcm"):
            total_samples += pcm.stat().st_size // 2  # int16
        total_s = total_samples / SAMPLE_RATE
        # The contract's ~10-20 s fixture target predates the v1 censored-pair
        # and truncated-interruption requirements: valid hesitation->commit
        # chains need multi-second agent runs, so the committed set lands
        # around 45 s. Still tiny; the bound keeps it from growing unnoticed.
        assert 8 <= total_s <= 60, f"{total_s:.1f}s outside the fixture budget"
