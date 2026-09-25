"""Vendored espeak-ng WAV synthesis adapter — behavior-identical copy of the
upstream gateway's ``EspeakTTS`` (the same adapter the live fallback path
uses). Always available in this environment given the ``espeak-ng`` binary."""

from __future__ import annotations

import asyncio
import shutil
import subprocess
import tempfile
from collections.abc import AsyncIterator
from pathlib import Path

import numpy as np

from dmel.runtime.audio import Pcm
from dmel.runtime.config import RealtimeConfig


class EspeakTTS:
    backend_name = "espeak-ng"

    def __init__(self, voice: str = "en-us", rate: int = 165) -> None:
        self.voice = voice
        self.rate = rate
        binary = shutil.which("espeak-ng") or shutil.which("espeak")
        if not binary:
            raise RuntimeError("espeak-ng is not installed")
        self.binary = binary

    async def synthesize(self, text: str) -> AsyncIterator[np.ndarray]:
        pcm = await asyncio.to_thread(self._synth, text)
        hop = RealtimeConfig.AGENT_CHUNK_SAMPLES
        for start in range(0, pcm.size, hop):
            yield pcm[start : start + hop]

    def _synth(self, text: str) -> np.ndarray:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as handle:
            path = Path(handle.name)
        try:
            subprocess.run(
                [
                    self.binary,
                    "-v",
                    self.voice,
                    "-s",
                    str(self.rate),
                    "-w",
                    str(path),
                    text,
                ],
                check=True,
                capture_output=True,
            )
            return Pcm.load_wav_i16(path)
        finally:
            path.unlink(missing_ok=True)
