"""Quantization bounds, codebook edges, and level sanity on synthetic signals."""

from __future__ import annotations

import numpy as np
import pytest
from _synth import silence, sine, white_noise

from dmel.features.config import DmelConfig
from dmel.features.tokenizer import decode_tokens, log_mel_db, quantize_levels, tokenize

CFG = DmelConfig()
# Explicit 100 dB span -> 6.25 dB per level (4 bits), for exact edge math.
EXPLICIT = DmelConfig(mel_min_db=-60.0, mel_max_db=40.0)


def test_codebook_edges_and_clipping():
    x = np.array([[-60.0, -53.75, -47.5, 40.0, -100.0, 100.0, -59.9, -57.0]])
    levels = quantize_levels(x, EXPLICIT)
    assert levels.tolist() == [[0, 1, 2, 15, 0, 15, 0, 0]]


def test_silence_quantizes_to_level_zero():
    tokens = tokenize(silence(500.0), CFG)
    assert tokens.shape == (10, 400)
    # Token layout: id = group * n_levels + level. Digital silence carries
    # level 0 in every bin-group, so ids are group*16, not all-zero ids.
    expected = np.tile(np.arange(80, dtype=np.int32) * 16, (10, 5))
    assert np.array_equal(tokens, expected)
    assert not decode_tokens(tokens, CFG).any()


def test_full_scale_tone_hits_top_level():
    levels = quantize_levels(log_mel_db(sine(440.0, 500.0), CFG), CFG)
    assert levels.max() == 15


def test_louder_signal_uses_higher_levels():
    quiet = quantize_levels(log_mel_db(sine(440.0, 300.0, amp=0.05), CFG), CFG)
    loud = quantize_levels(log_mel_db(sine(440.0, 300.0, amp=0.5), CFG), CFG)
    assert loud.mean() > quiet.mean()


@pytest.mark.parametrize("amp", [0.01, 0.1, 1.0])
def test_levels_within_bounds_for_noise(amp: float):
    levels = quantize_levels(log_mel_db(white_noise(300.0, amp=amp), CFG), CFG)
    assert levels.min() >= 0
    assert levels.max() <= 15


def test_bit_depth_is_configurable():
    cfg = DmelConfig(quant_bits=2)
    assert cfg.vocab_size == 80 * 4
    tokens = tokenize(white_noise(300.0), cfg)
    assert tokens.max() < cfg.vocab_size
    assert decode_tokens(tokens, cfg).max() <= 3


def test_nonfinite_logmel_rejected():
    with pytest.raises(ValueError):
        quantize_levels(np.array([[np.nan]]), CFG)


def test_bins_per_group_is_configurable():
    cfg = DmelConfig(bins_per_group=4)
    assert cfg.n_groups == 20
    assert cfg.vocab_size == 20 * 16
    tokens = tokenize(white_noise(300.0), cfg)
    assert tokens.shape == (6, cfg.tokens_per_step)
    assert tokens.max() < cfg.vocab_size
