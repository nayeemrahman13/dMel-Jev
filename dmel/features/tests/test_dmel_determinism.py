"""Determinism: same input bytes + config => identical token ids, always."""

from __future__ import annotations

import numpy as np
from _synth import sine, white_noise

from dmel.features.config import DmelConfig
from dmel.features.precompute import main
from dmel.features.tokenizer import mel_filterbank, tokenize

CFG = DmelConfig()


def test_tokenizer_is_deterministic():
    sig = white_noise(700.0, amp=0.4)  # not a whole-step multiple
    first = tokenize(sig, CFG)
    second = tokenize(sig.copy(), CFG)
    assert np.array_equal(first, second)


def test_filterbank_is_cached_readonly_and_identical():
    first = mel_filterbank(CFG)
    second = mel_filterbank(CFG)
    assert first is second
    assert np.array_equal(first, second)
    assert not first.flags.writeable


def test_cli_writes_byte_identical_caches(tmp_path):
    shard = tmp_path / "shard-00001"
    shard.mkdir()
    (shard / "s0001.mix.pcm").write_bytes(sine(440.0, 1000.0).tobytes())
    (shard / "s0002.mix.pcm").write_bytes(white_noise(1250.0, amp=0.3).tobytes())

    assert main(["--shard", str(shard)]) == 0
    first = {p.name: p.read_bytes() for p in sorted(shard.glob("*.dmel.npy"))}

    assert main(["--shard", str(shard), "--overwrite"]) == 0
    second = {p.name: p.read_bytes() for p in sorted(shard.glob("*.dmel.npy"))}

    assert set(first) == {"s0001.dmel.npy", "s0002.dmel.npy"}
    assert first == second


def test_different_input_gives_different_tokens():
    low = tokenize(sine(440.0, 500.0), CFG)
    high = tokenize(sine(880.0, 500.0), CFG)
    assert not np.array_equal(low, high)
