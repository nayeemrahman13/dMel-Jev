"""Round-trip and spectral sanity checks."""

from __future__ import annotations

import numpy as np
import pytest
from _synth import chirp, sine

from dmel.features.config import DmelConfig
from dmel.features.tokenizer import (
    decode_tokens,
    log_mel_db,
    mel_filterbank,
    tokens_from_levels,
)

CFG = DmelConfig()


def test_roundtrip_exact_for_single_bin_groups():
    rng = np.random.default_rng(7)
    levels = rng.integers(0, 16, size=(40, 80)).astype(np.int32)
    tokens = tokens_from_levels(levels, CFG)
    assert tokens.shape == (8, CFG.tokens_per_step)  # 40 subframes -> 8 steps
    decoded = decode_tokens(tokens, CFG)  # (n_steps, subframes, n_mels)
    assert np.array_equal(decoded, levels.reshape(8, 5, 80))


def test_grouped_tokens_collapse_by_max():
    cfg = DmelConfig(bins_per_group=4)
    rng = np.random.default_rng(11)
    levels = rng.integers(0, 16, size=(10, 80)).astype(np.int32)
    tokens = tokens_from_levels(levels, cfg)
    assert tokens.shape == (2, cfg.tokens_per_step)  # 10 subframes -> 2 steps
    expected = levels.reshape(2, cfg.subframes_per_step, cfg.n_groups, 4).max(axis=3)
    decoded = decode_tokens(tokens, cfg)  # (n_steps, subframes, n_groups)
    assert np.array_equal(decoded, expected)


def test_token_group_index_encodes_bin_group():
    rng = np.random.default_rng(3)
    levels = rng.integers(0, 16, size=(5, 80)).astype(np.int32)
    tokens = tokens_from_levels(levels, CFG)
    # id = group * n_levels + level; position p within a step: p // 80 = subframe.
    assert np.array_equal(tokens // 16, np.tile(np.arange(80, dtype=np.int32), 5)[None, :])


def _hz_to_mel(hz: float) -> float:
    # Standard HTK mel scale — independent re-derivation for the test.
    return 2595.0 * np.log10(1.0 + hz / 700.0)


def test_tone_energy_lands_near_its_mel_band():
    low = log_mel_db(sine(1000.0, 200.0), CFG).mean(axis=0)
    high = log_mel_db(sine(4000.0, 200.0), CFG).mean(axis=0)
    assert np.argmax(low) < np.argmax(high)
    mel_span = _hz_to_mel(8000.0) - _hz_to_mel(CFG.fmin_hz)
    for spectrum, hz in ((low, 1000.0), (high, 4000.0)):
        expected_band = (_hz_to_mel(hz) - _hz_to_mel(CFG.fmin_hz)) / (mel_span / CFG.n_mels)
        assert abs(int(np.argmax(spectrum)) - expected_band) <= 2


def test_chirp_spreads_energy_across_bins():
    chirp_mel = log_mel_db(chirp(500.0, 7000.0, 1000.0), CFG)
    sine_mel = log_mel_db(sine(1000.0, 1000.0), CFG)

    def active_bands(m: np.ndarray) -> int:
        return int((m > m.max() - 20.0).any(axis=0).sum())

    # A swept tone visits many mel bands over its duration; a pure tone only
    # lights up the bands around its frequency. Measured at a 20 dB floor:
    # chirp 62 bands vs sine 6 bands (ratio ~10); 4x leaves margin both ways.
    assert active_bands(chirp_mel) > 4 * active_bands(sine_mel)


def test_filterbank_shape_and_peak():
    bank = mel_filterbank(CFG)
    assert bank.shape == (80, 257)
    assert np.all(bank >= 0.0) and np.all(bank <= 1.0)
    peaks = bank.max(axis=1)
    assert np.all(peaks > 0.0)                       # every band responds
    assert np.all(np.diff(bank.argmax(axis=1)) >= 0)  # triangles ordered by freq
    # With several FFT bins per band, every triangle samples its unit peak;
    # at 80 bands the narrowest low-frequency bands are sub-bin-width and may
    # not (same as librosa-style filterbanks) — asserted only on a coarse bank.
    coarse = mel_filterbank(DmelConfig(n_mels=8))
    assert np.all(coarse.max(axis=1) > 0.95)


def test_decode_rejects_out_of_vocab_and_wrong_width():
    with pytest.raises(ValueError):
        decode_tokens(np.full((1, 400), 1280, dtype=np.int32), CFG)
    with pytest.raises(ValueError):
        decode_tokens(np.zeros((1, 399), dtype=np.int32), CFG)
