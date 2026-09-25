"""Contract guardrails for the DmelConfig surface."""

from __future__ import annotations

import pytest

from dmel.features.config import DEFAULT_CONFIG_PATH, DmelConfig


def test_default_yaml_matches_dataclass_defaults():
    assert DmelConfig.from_yaml(DEFAULT_CONFIG_PATH) == DmelConfig()


def test_derived_sizes_match_contract():
    cfg = DmelConfig()
    assert (cfg.window_samples, cfg.hop_samples, cfg.step_samples) == (400, 160, 800)
    assert cfg.subframes_per_step == 5
    assert cfg.n_levels == 16
    assert cfg.n_groups == 80
    assert cfg.tokens_per_step == 400
    assert cfg.vocab_size == 1280


@pytest.mark.parametrize(
    "overrides",
    [
        {"quant_bits": 0},
        {"quant_bits": 17},
        {"bins_per_group": 0},
        {"bins_per_group": 3},  # 80 % 3 != 0
        {"mel_min_db": 50.0, "mel_max_db": -70.0},
        {"mel_min_db": 0.0, "mel_max_db": 0.0},
        {"n_fft": 256},  # < window_samples (400)
        {"hop_ms": 7.0},  # 50 % 7 != 0
        {"window": "hamming"},
        {"fmin_hz": -1.0},
        {"fmax_hz": 9000.0},  # > Nyquist
        {"sample_rate_hz": 0},
        {"log_eps": 0.0},
    ],
)
def test_invalid_configs_raise(overrides: dict):
    with pytest.raises(ValueError):
        DmelConfig(**overrides)


def test_unknown_yaml_key_rejected():
    with pytest.raises(ValueError):
        DmelConfig.from_mapping({"n_mels": 80, "scenario": "interruption"})


def test_yaml_type_coercion():
    assert DmelConfig.from_mapping({"window_ms": 25, "fmax_hz": None}) == DmelConfig()
