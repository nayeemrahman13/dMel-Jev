"""Dataset loader tests: manifest-driven splits, validation, windowing."""

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest
import torch

from dmel.models.config import DmelModelConfig
from dmel.models.model import DmelBargeInNet
from dmel.training.dataset import (
    DatasetError,
    WindowDataset,
    balanced_action_weights,
    group_by_split,
    scan_samples,
)
from dmel.training.synthetic import (
    SYNTHETIC_TOKENS_PER_STEP,
    SYNTHETIC_VOCAB_SIZE,
    scaffold_features,
    write_synthetic_shard,
)


def test_manifest_split_is_honored(tmp_path):
    shard = write_synthetic_shard(tmp_path, n_samples=10, n_steps=64, seed=1)
    samples = scan_samples(shard)
    grouped = group_by_split(samples)
    assert len(grouped["train"]) > 0
    assert len(grouped["val"]) > 0
    # The synthetic manifest is the assignment's only source; no local re-split.
    assert len(samples) == sum(len(v) for v in grouped.values())


def test_missing_manifest_fails_loudly(tmp_path):
    shard = write_synthetic_shard(tmp_path, n_samples=4, n_steps=64, seed=2)
    (shard / "manifest.json").unlink()
    with pytest.raises(DatasetError, match="manifest"):
        scan_samples(shard)


def test_sample_without_split_fails_loudly(tmp_path):
    import json

    shard = write_synthetic_shard(tmp_path, n_samples=4, n_steps=64, seed=3)
    manifest = json.loads((shard / "manifest.json").read_text())
    del manifest["synth-0001"]["split"]
    (shard / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(DatasetError, match="synth-0001"):
        scan_samples(shard)


def test_missing_token_cache_fails_loudly(tmp_path):
    shard = write_synthetic_shard(tmp_path, n_samples=4, n_steps=64, seed=4)
    (shard / "synth-0000.dmel.npy").unlink()
    with pytest.raises(DatasetError, match="token cache"):
        scan_samples(shard)


def test_window_dataset_layout(tmp_path):
    shard = write_synthetic_shard(tmp_path, n_samples=6, n_steps=96, seed=6)
    samples = scan_samples(shard)
    dataset = WindowDataset(samples, window_steps=48, train=False)
    # Non-overlapping, deterministic windows; tails dropped.
    assert len(dataset) == sum(96 // 48 for _ in samples)
    item = dataset[0]
    assert item["tokens"].shape == (48, samples[0].tokens.shape[1])
    assert item["action"].shape == (48,)
    assert set(item["agent_speaking"].shape) == {48}
    for name in ("speech_present", "primary_user", "backchannel", "interrupt_intent"):
        assert name in item


def test_balanced_action_weights_split_classes(tmp_path):
    shard = write_synthetic_shard(tmp_path, n_samples=6, n_steps=96, seed=7)
    samples = [s for s in scan_samples(shard) if s.split == "train"]
    dataset = WindowDataset(samples, window_steps=48, train=False)
    weights = balanced_action_weights(dataset).numpy()
    is_stop = np.array(
        [int(dataset[i]["action"].sum()) > 0 for i in range(len(dataset))]
    )
    n_stop = int(is_stop.sum())
    n_keep = len(dataset) - n_stop
    if n_stop == 0 or n_keep == 0:
        assert np.allclose(weights, 1.0)
        return
    # Inverse frequency, uniform within class: the stop/keep weight ratio
    # equals the keep/stop window-count ratio.
    stop_weight = float(weights[is_stop].mean())
    keep_weight = float(weights[~is_stop].mean())
    assert stop_weight / keep_weight == pytest.approx(n_keep / n_stop, rel=1e-6)


def test_scaffold_features_match_synthetic_shard(tmp_path):
    """Regression for the Modal GPU smoke crash: the model geometry used
    with synthetic shards must match the geometry write_synthetic_shard
    emits — the runner once built a contract-geometry model against
    scaffold data and died on the input projection shape check."""
    shard = write_synthetic_shard(tmp_path, n_samples=4, n_steps=64, seed=8)
    base = DmelModelConfig()
    features = scaffold_features(base.features)
    assert features.token_vocab_size == SYNTHETIC_VOCAB_SIZE
    assert features.tokens_per_step == SYNTHETIC_TOKENS_PER_STEP

    samples = scan_samples(shard)
    dataset = WindowDataset(
        samples,
        window_steps=32,
        train=False,
        token_vocab_size=features.token_vocab_size,
    )
    batch = torch.utils.data.default_collate([dataset[i] for i in range(2)])
    model = DmelBargeInNet(
        DmelModelConfig(
            features=features,
            model=replace(base.model, arm="lstm"),
            policy=base.policy,
            loss_weights=base.loss_weights,
        )
    )
    model.eval()
    with torch.no_grad():
        outputs = model(batch["tokens"], batch["agent_speaking"])
    assert outputs["action"].shape == batch["action"].shape + (2,)
