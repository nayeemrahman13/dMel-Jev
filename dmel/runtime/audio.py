"""Vendored PCM helpers — behavior-identical copy of the upstream gateway's
``Pcm`` (16 kHz mono int16 is the POC's native format)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf

from dmel.runtime.config import RealtimeConfig


class Pcm:
    @staticmethod
    def i16_to_f32(samples: np.ndarray) -> np.ndarray:
        return samples.astype(np.float32) / 32768.0

    @staticmethod
    def load_wav_i16(path: Path, sample_rate: int = RealtimeConfig.SAMPLE_RATE) -> np.ndarray:
        audio, rate = sf.read(path, dtype="int16", always_2d=True)
        mono = audio.mean(axis=1).astype(np.int16)
        if rate == sample_rate:
            return mono
        ratio = sample_rate / rate
        out_len = int(round(mono.size * ratio))
        x_old = np.linspace(0.0, 1.0, mono.size, endpoint=False)
        x_new = np.linspace(0.0, 1.0, out_len, endpoint=False)
        resampled = np.interp(x_new, x_old, mono.astype(np.float32))
        return np.clip(resampled, -32768, 32767).astype(np.int16)
