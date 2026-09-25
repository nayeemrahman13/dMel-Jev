"""Fixture-consumer tests: run against dmel/data/fixtures when the data PR has merged.

Skipped cleanly while fixtures are absent (the data task owns that directory).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from dmel.features.config import DmelConfig
from dmel.features.tokenizer import n_steps_for_samples, read_pcm_i16, tokenize

REPO_ROOT = Path(__file__).resolve().parents[3]
FIXTURES = REPO_ROOT / "dmel" / "data" / "fixtures"

pytestmark = pytest.mark.skipif(
    not FIXTURES.is_dir(), reason="dmel/data/fixtures not merged yet (data task)"
)


def _mix_files() -> list[Path]:
    return sorted(FIXTURES.glob("*.mix.pcm"))


def test_fixtures_exist_when_directory_present():
    assert _mix_files(), f"{FIXTURES} exists but contains no *.mix.pcm"


def test_fixture_token_alignment():
    cfg = DmelConfig()
    for path in _mix_files()[:5]:
        pcm = read_pcm_i16(path)
        tokens = tokenize(pcm, cfg)
        assert tokens.shape == (n_steps_for_samples(len(pcm), cfg), cfg.tokens_per_step)


def test_fixture_labels_row_count_matches_steps():
    pyarrow = pytest.importorskip("pyarrow")
    cfg = DmelConfig()
    checked = 0
    for labels_path in sorted(FIXTURES.glob("*.labels.parquet"))[:5]:
        sample_id = labels_path.name.removesuffix(".labels.parquet")
        mix_path = FIXTURES / f"{sample_id}.mix.pcm"
        if not mix_path.exists():
            continue
        pcm = read_pcm_i16(mix_path)
        n_rows = pyarrow.parquet.read_table(labels_path).num_rows
        assert n_rows == n_steps_for_samples(len(pcm), cfg), (
            f"{sample_id}: labels.parquet has {n_rows} rows but the tokenizer "
            f"produces {n_steps_for_samples(len(pcm), cfg)} steps — the data and "
            "features areas disagree on frame alignment"
        )
        checked += 1
    assert checked > 0, "no labels.parquet had a matching mix.pcm"
