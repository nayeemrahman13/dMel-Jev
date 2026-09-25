"""Deterministic synthetic test signals (contract: sine, chirp, silence, noise)."""

from __future__ import annotations

import numpy as np

SR = 16_000


def to_i16(x: np.ndarray) -> np.ndarray:
    """Float in [-1, 1] -> int16 PCM (the tokenizer's public input dtype)."""
    return np.clip(np.rint(np.asarray(x) * 32767.0), -32768, 32767).astype(np.int16)


def n_samples(dur_ms: float) -> int:
    return round(dur_ms * SR / 1000.0)


def sine(freq_hz: float, dur_ms: float, amp: float = 1.0) -> np.ndarray:
    t = np.arange(n_samples(dur_ms)) / SR
    return to_i16(amp * np.sin(2.0 * np.pi * freq_hz * t))


def chirp(f0_hz: float, f1_hz: float, dur_ms: float, amp: float = 1.0) -> np.ndarray:
    """Linear sweep from f0 to f1."""
    n = n_samples(dur_ms)
    t = np.arange(n) / SR
    total = dur_ms / 1000.0
    phase = 2.0 * np.pi * (f0_hz * t + (f1_hz - f0_hz) * t**2 / (2.0 * total))
    return to_i16(amp * np.sin(phase))


def white_noise(dur_ms: float, amp: float = 1.0, seed: int = 1234) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return to_i16(amp * rng.standard_normal(n_samples(dur_ms)))


def silence(dur_ms: float) -> np.ndarray:
    return np.zeros(n_samples(dur_ms), dtype=np.int16)
