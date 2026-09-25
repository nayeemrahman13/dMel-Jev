"""Fixture-set conformance: run the policy interface over the committed
contract fixtures (dmel/data/fixtures/). Skipped if the fixtures directory
is absent — the synthetic scaffolding tests carry the same assertions
until then."""

from pathlib import Path

import numpy as np
import pytest

from dmel.models.config import DmelModelConfig
from dmel.models.model import DmelBargeInNet
from dmel.models.policy import PROBS_SCHEMA, CheckpointedBargeInPolicy
from dmel.training.dataset import scan_samples
from dmel.training.features_cache import ensure_token_caches

FIXTURES_ROOT = Path("dmel/data/fixtures")

pytestmark = pytest.mark.skipif(
    not FIXTURES_ROOT.exists() or not any(FIXTURES_ROOT.glob("**/*.labels.parquet")),
    reason="dmel/data/fixtures not present yet (data PR pending)",
)


def test_policy_interface_over_fixtures():
    # Committed fixtures ship pcm + labels only; token caches are materialized
    # on demand by the features precompute (deterministic, skips existing).
    ensure_token_caches(FIXTURES_ROOT)
    samples = scan_samples(FIXTURES_ROOT)
    assert samples, "fixtures directory exists but contains no samples"
    base = DmelModelConfig()
    policy = CheckpointedBargeInPolicy(DmelBargeInNet(base), base)
    for sample in samples[:3]:
        policy.reset()
        for step in range(min(10, sample.n_steps)):
            frame_pcm = np.zeros(800, dtype=np.int16)
            result = policy.step(
                frame_pcm, sample.tokens[step], agent_speaking=bool(sample.agent_speaking[step])
            )
            assert set(result["probs"]) == set(PROBS_SCHEMA)
            assert result["action"] in ("KEEP", "STOP_TTS")
