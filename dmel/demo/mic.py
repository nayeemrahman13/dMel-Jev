"""Microphone capture for the live demo.

sounddevice (PortAudio) — cross-platform. 16 kHz mono int16, read in the
contract's 800-sample (50 ms) frames. sounddevice is imported lazily so the
rest of the demo module (and the whole test suite) works on machines without
a PortAudio library; importing THIS module is always safe.
"""

from __future__ import annotations

import numpy as np

from dmel.demo.frontend import FRAME_SAMPLES, validate_frame

SAMPLE_RATE = 16_000


class MicError(RuntimeError):
    """Raised when the microphone cannot be opened or read."""


def _sounddevice():
    try:
        import sounddevice as sd
    except OSError as exc:  # PortAudio library missing — the usual cause
        raise MicError(
            "sounddevice could not load the PortAudio library — install it "
            "(Debian/Ubuntu: sudo apt install libportaudio2; macOS: brew install "
            "portaudio) and `pip install -r dmel/demo/requirements.txt`"
        ) from exc
    return sd


def list_devices() -> str:
    """Human-readable device list for --list-devices."""
    sd = _sounddevice()
    lines = [str(sd.query_devices())]
    try:
        default_input, _ = sd.default.device
        lines.append(f"default input device: {default_input}")
    except Exception as exc:  # sd.default can raise on exotic setups — still print the list
        lines.append(f"(no default input device: {exc})")
    return "\n".join(lines)


class MicStream:
    """Continuous mic capture; each read returns exactly one 50 ms frame."""

    def __init__(self, device: int | str | None = None, sample_rate: int = SAMPLE_RATE) -> None:
        self._sd = _sounddevice()
        self.device = device
        self.sample_rate = sample_rate
        self._stream = self._sd.InputStream(
            samplerate=sample_rate,
            channels=1,
            dtype="int16",
            blocksize=0,
            device=device,
        )
        self._overflowed_once = False

    def __enter__(self) -> "MicStream":
        self._stream.start()
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def read_frame(self) -> np.ndarray:
        """Block until one 800-sample int16 frame is available."""
        data, overflowed = self._stream.read(FRAME_SAMPLES)
        if overflowed and not self._overflowed_once:
            # Real signal loss the user should know about (slow consumer);
            # warn once per session so the log stays readable.
            self._overflowed_once = True
            print("⚠ mic buffer overflowed at least once — frames were dropped by PortAudio")
        frame = np.asarray(data).reshape(-1)
        if frame.shape[0] != FRAME_SAMPLES:
            raise MicError(f"mic read returned {frame.shape[0]} samples, expected {FRAME_SAMPLES}")
        return validate_frame(frame)

    def close(self) -> None:
        try:
            self._stream.stop()
            self._stream.close()
        except Exception:  # best-effort close; PortAudio can already be gone at shutdown
            pass
