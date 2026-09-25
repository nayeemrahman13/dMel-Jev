"""Guardrails for the geometry documented in docs/dmel_geometry.md.

Asserts the tokenizer-side constants (sample rate, mel window/hop, subframes
per step, tokens per step, vocabulary size, codebook) and the token ID
layout/formula against the live code, so the document cannot silently rot.
Torch-free by design: dmel/features declares only numpy + PyYAML.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import yaml

from dmel.features.config import DEFAULT_CONFIG_PATH, DmelConfig
from dmel.features.tokenizer import log_mel_db, quantize_levels, tokenize
from _synth import sine

# Constants asserted below, per docs/dmel_geometry.md (@ a543b54).
EXPECTED = {
    "sample_rate_hz": 16_000,
    "window_samples": 400,  # 25 ms
    "hop_samples": 160,  # 10 ms
    "step_samples": 800,  # 50 ms
    "subframes_per_step": 5,
    "n_mels": 80,
    "n_fft": 512,
    "n_levels": 16,  # 4-bit
    "bins_per_group": 1,
    "n_groups": 80,
    "tokens_per_step": 400,  # 5 subframes x 80 channels
    "vocab_size": 1280,  # 80 bin-groups x 16 levels
    "mel_min_db": -70.0,
    "mel_max_db": 50.0,
}


def test_features_geometry_constants():
    cfg = DmelConfig()
    assert cfg.sample_rate_hz == EXPECTED["sample_rate_hz"]
    assert (cfg.window_samples, cfg.hop_samples, cfg.step_samples) == (
        EXPECTED["window_samples"],
        EXPECTED["hop_samples"],
        EXPECTED["step_samples"],
    )
    assert cfg.subframes_per_step == EXPECTED["subframes_per_step"]
    assert (cfg.n_mels, cfg.n_fft) == (EXPECTED["n_mels"], EXPECTED["n_fft"])
    assert cfg.n_levels == EXPECTED["n_levels"]
    assert (cfg.bins_per_group, cfg.n_groups) == (
        EXPECTED["bins_per_group"],
        EXPECTED["n_groups"],
    )
    assert cfg.tokens_per_step == EXPECTED["tokens_per_step"]
    assert cfg.vocab_size == EXPECTED["vocab_size"]
    assert (cfg.mel_min_db, cfg.mel_max_db) == (
        EXPECTED["mel_min_db"],
        EXPECTED["mel_max_db"],
    )


def test_training_config_yaml_matches_features_geometry():
    """dmel/training/config.yaml features section must equal the live geometry."""
    cfg = DmelConfig()
    training_yaml = Path(__file__).resolve().parents[3] / "dmel" / "training" / "config.yaml"
    raw = yaml.safe_load(training_yaml.read_text())["features"]
    assert raw["token_vocab_size"] == cfg.vocab_size
    assert raw["tokens_per_step"] == cfg.tokens_per_step


def test_step_token_count_and_steps_rounding():
    """One 50 ms step tokenizes to (n_steps, 400); a 125 ms clip rounds to 3 steps."""
    cfg = DmelConfig()
    tokens = tokenize(sine(440.0, dur_ms=50), cfg)
    assert tokens.shape == (1, 400)
    assert tokens.dtype == np.int32
    # 125 ms sits exactly halfway between 2 and 3 steps -> rounds up (3).
    assert tokenize(sine(440.0, dur_ms=125), cfg).shape == (3, 400)


def test_position_layout_and_id_formula():
    """Position p encodes (subframe, bin) = (p // 80, p % 80); id = bin*16 + level."""
    cfg = DmelConfig()
    pcm = sine(440.0, dur_ms=50, amp=0.25)
    tokens = tokenize(pcm, cfg)[0]

    levels = quantize_levels(log_mel_db(pcm, cfg), cfg)  # (5 subframes, 80 bins)
    for subframe in (0, 4):
        for bin_index in (0, 5, 79):
            expected = bin_index * cfg.n_levels + int(levels[subframe, bin_index])
            position = subframe * 80 + bin_index
            assert int(tokens[position]) == expected
            # The position decomposition documented in dmel_geometry.md:
            assert position // 80 == subframe
            assert position % 80 == bin_index


def test_silence_ids_are_bin_multiples():
    """Level 0 everywhere -> ids are exactly {bin*16}, tiled per subframe."""
    cfg = DmelConfig()
    silence = np.zeros(800, dtype=np.int16)
    tokens = tokenize(silence, cfg)[0]
    assert np.array_equal(tokens, np.tile(np.arange(80, dtype=np.int32) * 16, 5))


def test_default_yaml_matches_documented_geometry():
    """The shipped YAML must reproduce the documented derived constants."""
    cfg = DmelConfig.from_yaml(DEFAULT_CONFIG_PATH)
    assert cfg.tokens_per_step == 400 and cfg.vocab_size == 1280
