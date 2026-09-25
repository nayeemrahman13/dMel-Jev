"""Vendored runtime constants for the dMel POC.

Behavior-identical subset of the upstream realtime gateway's config, reduced
to the knobs the dmel areas actually consume (TTS chunking, VAD silence
windows, vendored asset paths). Kept as a class of constants so the vendored
``Pcm`` / ``EspeakTTS`` / ``SileroVAD`` copies read exactly like their
upstream originals.
"""

from __future__ import annotations

import os
from pathlib import Path

_PACKAGE_ROOT = Path(__file__).resolve().parent


class RealtimeConfig:
    SAMPLE_RATE = 16_000
    # EspeakTTS streams in 512-sample chunks (upstream agent playback chunk).
    AGENT_CHUNK_SAMPLES = 512
    # 5 × 32 ms Silero windows ≈ 160 ms of silence before SPEECH_ENDED.
    VAD_MIN_SILENCE_WINDOWS = int(os.environ.get("DMEL_VAD_SILENCE_WINDOWS", "5"))

    ASSETS_DIR = _PACKAGE_ROOT / "assets"
    MODELS_DIR = _PACKAGE_ROOT / "models"
    SILERO_ONNX = MODELS_DIR / "silero_vad.onnx"
    INJECT_WAV = ASSETS_DIR / "inject_speech.wav"
