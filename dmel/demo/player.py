"""Agent TTS playback for the live demo, with immediate stop.

Synthesizes the demo script once via the vendored espeak-ng adapter
(``dmel.runtime.tts.EspeakTTS`` — the same adapter that generated the
training corpus), then plays it through sounddevice in 800-sample (50 ms)
ticks. The blocking output write IS the demo's 50 ms clock.

``agent_speaking`` = there is still unplayed agent audio (TTS playback
state — the contract's legitimately-available input flag). ``stop_now()``
drops the pending audio and aborts the output stream, which silences
playback immediately; at most one already-queued 50 ms block can remain
audible (PortAudio's buffer), identical for both arms, so the A/B feel
comparison stays fair.
"""

from __future__ import annotations

import asyncio

import numpy as np

from dmel.demo.frontend import FRAME_SAMPLES
from dmel.demo.mic import SAMPLE_RATE, _sounddevice

DEFAULT_SCRIPT = (
    "So this is the live barge-in demo. I am going to keep talking for a while, "
    "which means this is your chance to interrupt me. Try to cut in while I am "
    "mid-sentence, and listen to how quickly the playback stops. If nothing "
    "happens, wait for a pause between phrases, speak a little louder, and try "
    "again. Switch between the two interruption brains with the a and c keys, "
    "and notice that one of them was trained to hesitate. The other one was "
    "trained on a synthetic corpus of synthesized conversations, and it has "
    "opinions about what your voice means. Whenever you are ready, go ahead "
    "and talk over me."
)


class AgentPlayback:
    """Script playback in 50 ms ticks with immediate-stop semantics."""

    def __init__(
        self, voice: str = "en-us", rate: int = 165, device: int | str | None = None
    ) -> None:
        # Vendored adapter — raises RuntimeError when espeak-ng is not installed.
        from dmel.runtime.tts import EspeakTTS

        self._tts = EspeakTTS(voice=voice, rate=rate)
        self._sd = _sounddevice()
        self.voice = voice
        self.rate = rate
        self.device = device
        self._pending = np.zeros(0, dtype=np.int16)
        self._stream = self._sd.OutputStream(
            samplerate=SAMPLE_RATE, channels=1, dtype="int16", blocksize=0, device=device
        )
        self._script: str | None = None

    def __enter__(self) -> "AgentPlayback":
        self._stream.start()
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def speak(self, text: str | None = None) -> None:
        """Synthesize (blocking) and queue for playback."""
        self._script = text if text is not None else DEFAULT_SCRIPT
        self._pending = self._synthesize(self._script)

    @property
    def script(self) -> str | None:
        return self._script

    def _synthesize(self, text: str) -> np.ndarray:
        async def _collect():
            return [chunk async for chunk in self._tts.synthesize(text)]

        chunks = asyncio.run(_collect())
        return np.concatenate(chunks).reshape(-1).astype(np.int16, copy=False)

    @property
    def speaking(self) -> bool:
        """True while unplayed agent audio remains (the contract's input flag)."""
        return self._pending.size > 0

    def tick(self) -> bool:
        """Write the next 50 ms of agent audio (blocking — the demo clock).

        Returns the agent_speaking flag for the tick that just elapsed:
        True iff agent audio was still playing after this write consumed
        its chunk.
        """
        if self.speaking:
            chunk = self._pending[:FRAME_SAMPLES]
            if chunk.shape[0] < FRAME_SAMPLES:
                chunk = np.pad(chunk, (0, FRAME_SAMPLES - chunk.shape[0]))
            self._stream.write(chunk.reshape(-1, 1))
            self._pending = self._pending[FRAME_SAMPLES:]
            return self.speaking
        self._stream.write(np.zeros((FRAME_SAMPLES, 1), dtype=np.int16))  # keep the clock alive
        return False

    def stop_now(self) -> None:
        """Abort playback immediately (drop pending audio, kill queued blocks)."""
        self._pending = np.zeros(0, dtype=np.int16)
        try:
            self._stream.abort()
            self._stream.start()
        except Exception:  # stream already stopped — nothing to abort
            pass

    def close(self) -> None:
        self._pending = np.zeros(0, dtype=np.int16)
        try:
            self._stream.stop()
            self._stream.close()
        except Exception:  # best-effort close at shutdown
            pass
