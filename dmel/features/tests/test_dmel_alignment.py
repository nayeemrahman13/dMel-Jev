"""Frame-count alignment: n_steps == round(duration_ms / 50) exactly."""

from __future__ import annotations

import numpy as np
import pytest
from _synth import sine

from dmel.features.config import DmelConfig
from dmel.features.tokenizer import log_mel_db, n_steps_for_samples, tokenize

CFG = DmelConfig()


@pytest.mark.parametrize(
    "n_samples",
    [0, 1, 160, 399, 400, 401, 799, 800, 801, 1199, 1200, 1201, 1600, 1601, 32000],
)
def test_step_count_matches_round_half_up(n_samples: int):
    tokens = tokenize(np.zeros(n_samples, dtype=np.int16), CFG)
    expected = (n_samples + 400) // 800
    assert n_steps_for_samples(n_samples, CFG) == expected
    assert tokens.shape == (expected, CFG.tokens_per_step)


def test_two_second_clip_gives_40_steps():
    tokens = tokenize(sine(440.0, 2000.0), CFG)
    assert tokens.shape == (40, 400)
    assert tokens.dtype == np.int32


def test_subframe_rows_match_steps():
    mel = log_mel_db(sine(440.0, 1000.0), CFG)
    assert mel.shape == (20 * CFG.subframes_per_step, CFG.n_mels)


def test_exact_tie_rounds_up():
    # 1200 samples = 75 ms, exactly halfway between 1 and 2 steps -> 2 (half-up).
    # The tie rule differs from Python's half-to-even round() only at exact
    # ties, which whole-step corpus clips never hit; it is documented in
    # n_steps_for_samples and pinned here.
    assert n_steps_for_samples(1200, CFG) == 2
    assert n_steps_for_samples(2800, CFG) == 4
    assert n_steps_for_samples(1199, CFG) == 1


def test_tail_padding_on_non_step_multiple_clip():
    # 1041 samples = 65.06 ms -> 1 step; the final subframe window extends
    # past the clip end and is zero-padded (documented edge padding).
    sig = sine(440.0, 65.0625)
    assert sig.shape[0] == 1041
    mel = log_mel_db(sig, CFG)
    assert mel.shape == (5, 80)
    # The padded final subframe still contains real signal (samples 640..1039).
    assert mel[4].max() > -20.0
    tokens = tokenize(sig, CFG)
    assert tokens.shape == (1, 400)


def test_token_ids_within_vocabulary():
    tokens = tokenize(sine(440.0, 2000.0), CFG)
    assert tokens.min() >= 0
    assert tokens.max() < CFG.vocab_size
