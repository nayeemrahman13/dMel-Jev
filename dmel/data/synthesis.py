"""Speech synthesis via the vendored EspeakTTS adapter plus shaped-noise events.

The vendored adapter (``dmel.runtime.tts.EspeakTTS``) is a behavior-identical
copy of the live fallback path's adapter, per the contract — see
``dmel/runtime/README.md``.

Noise events (cough/breath/burst) are shaped-noise approximations synthesized
with numpy — no CC0 sample pack was fetched; the choice is recorded in each
shard manifest. Shapes are deterministic given their RNG.
"""

from __future__ import annotations

import subprocess
from functools import lru_cache

import numpy as np

from dmel.data.constants import SAMPLE_RATE
from dmel.runtime.tts import EspeakTTS


@lru_cache(maxsize=1)
def espeak_version() -> str:
    out = subprocess.run(
        ["espeak-ng", "--version"], check=True, capture_output=True, text=True
    )
    return out.stdout.strip().splitlines()[0]


def render_utterance(text: str, voice: str, rate: int) -> np.ndarray:
    """Render one line to int16 PCM at 16 kHz through the repo adapter."""
    tts = EspeakTTS(voice=voice, rate=rate)

    async def _collect() -> np.ndarray:
        chunks = [chunk async for chunk in tts.synthesize(text)]
        return np.concatenate(chunks)

    import asyncio

    return asyncio.run(_collect())


class RenderCache:
    """Deterministic (voice, rate, text) -> int16 PCM cache."""

    def __init__(self) -> None:
        self._cache: dict[tuple[str, int, str], np.ndarray] = {}

    def render(self, text: str, role_voice: str, rate: int) -> np.ndarray:
        key = (role_voice, rate, text)
        cached = self._cache.get(key)
        if cached is None:
            cached = render_utterance(text, role_voice, rate)
            self._cache[key] = cached
        return cached

    def __call__(self, text: str, role_voice: str, rate: int) -> np.ndarray:
        return self.render(text, role_voice, rate)


def _bandshape(pcm: np.ndarray, low_hz: float, high_hz: float, slope: float = 0.0) -> np.ndarray:
    """FFT band-shaping: keep [low_hz, high_hz], optional 1/f^slope tilt."""
    spectrum = np.fft.rfft(pcm)
    freqs = np.fft.rfftfreq(pcm.size, d=1.0 / SAMPLE_RATE)
    mask = (freqs >= low_hz) & (freqs <= high_hz)
    shaped = spectrum * mask
    if slope:
        tilt = 1.0 / np.maximum(freqs, 20.0) ** slope
        shaped = shaped * tilt
    return np.fft.irfft(shaped, n=pcm.size)


def _envelope(n: int, attack: int, decay_tail: int, rng: np.random.Generator) -> np.ndarray:
    """Attack/decay amplitude envelope in [0, 1]."""
    env = np.ones(n, dtype=np.float64)
    attack = min(attack, n // 3)
    env[:attack] = np.linspace(0.0, 1.0, attack)
    tail = min(decay_tail, n - attack)
    if tail > 0:
        env[n - tail :] *= np.linspace(1.0, 0.0, tail) ** 1.5
    if rng.random() < 0.5:  # slight double-pulse texture (e.g. two-part cough)
        bump_start = int(n * float(rng.uniform(0.35, 0.6)))
        bump = np.exp(-np.linspace(0.0, 6.0, n - bump_start))
        env[bump_start:] *= 0.3 + 0.7 * bump
    return env


def synth_noise(kind: str, n_samples: int, rng: np.random.Generator) -> np.ndarray:
    """Shaped-noise approximation of a non-speech vocal event, float in [-1, 1)."""
    noise = rng.standard_normal(n_samples)
    if kind == "cough":
        shaped = _bandshape(noise, 300.0, 3200.0, slope=0.6)
        env = _envelope(n_samples, int(0.008 * SAMPLE_RATE), int(0.55 * n_samples) + 1, rng)
        return shaped * env
    if kind == "breath":
        shaped = _bandshape(noise, 80.0, 1400.0, slope=1.2)
        env = np.sin(np.linspace(0.0, np.pi, n_samples)) ** 1.5  # slow in/out
        return shaped * env
    if kind == "burst":
        shaped = _bandshape(noise, 1000.0, 7500.0)
        env = np.exp(-np.linspace(0.0, 9.0, n_samples))  # sharp transient
        return shaped * env
    raise ValueError(f"unknown noise kind: {kind!r}")
