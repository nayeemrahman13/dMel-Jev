"""Vendored Silero VAD v5 via ONNX Runtime (no PyTorch) — behavior-identical
copy of the upstream gateway's wrapper, including the 512-sample window and
64-sample context constants, with the model loaded from the vendored ONNX in
``dmel/runtime/models/``."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np

from dmel.runtime.config import RealtimeConfig

WINDOW_SAMPLES = 512
CONTEXT_SAMPLES = 64


@dataclass
class VadEvent:
    kind: str
    prob: float
    t0_client: float
    t0_server: float


class StreamingVAD(Protocol):
    backend_name: str
    last_prob: float
    speech_ms: float
    peak_speech_ms: float

    def reset(self) -> None: ...

    def push(
        self,
        samples: np.ndarray,
        client_ts: float,
        server_ts: float,
    ) -> list[VadEvent]: ...

    def new_stream(self) -> StreamingVAD: ...


class SileroVAD:
    """Eager speech-start (one window above threshold) for the Phase 1 baseline."""

    def __init__(
        self,
        session: object,
        input_names: list[str],
        output_names: list[str],
        window_samples: int,
        context_samples: int,
        threshold: float = 0.5,
        end_threshold: float = 0.35,
        min_silence_windows: int | None = None,
    ) -> None:
        self._session = session
        self._input_names = input_names
        self._output_names = output_names
        self.window_samples = window_samples
        self.context_samples = context_samples
        self.threshold = threshold
        self.end_threshold = end_threshold
        self.min_silence_windows = (
            RealtimeConfig.VAD_MIN_SILENCE_WINDOWS
            if min_silence_windows is None
            else min_silence_windows
        )
        self.backend_name = "silero-onnx"
        self.last_prob = 0.0
        self.reset()

    @classmethod
    def load(cls, model_path: Path | None = None) -> SileroVAD:
        import onnxruntime as ort

        path = Path(model_path or RealtimeConfig.SILERO_ONNX)
        opts = ort.SessionOptions()
        opts.inter_op_num_threads = 1
        opts.intra_op_num_threads = 1
        session = ort.InferenceSession(str(path), sess_options=opts, providers=["CPUExecutionProvider"])
        inputs = session.get_inputs()
        outputs = session.get_outputs()
        input_names = [i.name for i in inputs]
        output_names = [o.name for o in outputs]
        audio_dim = inputs[0].shape[-1]
        if isinstance(audio_dim, int) and audio_dim == WINDOW_SAMPLES:
            context = 0
            window = WINDOW_SAMPLES
        else:
            context = CONTEXT_SAMPLES
            window = WINDOW_SAMPLES
        return cls(session, input_names, output_names, window, context)

    def new_stream(self) -> StreamingVAD:
        return SileroVAD(
            self._session,
            self._input_names,
            self._output_names,
            self.window_samples,
            self.context_samples,
            self.threshold,
            self.end_threshold,
            self.min_silence_windows,
        )

    def reset(self) -> None:
        self._state = np.zeros((2, 1, 128), dtype=np.float32)
        self._context = np.zeros((self.context_samples,), dtype=np.float32)
        self._carry = np.zeros((0,), dtype=np.float32)
        self._carry_client_ts = 0.0
        self._carry_server_ts = 0.0
        self._triggered = False
        self._silence_windows = 0
        self.last_prob = 0.0
        self.speech_ms = 0.0
        self.peak_speech_ms = 0.0

    def push(self, samples: np.ndarray, client_ts: float, server_ts: float) -> list[VadEvent]:
        audio = samples.astype(np.float32, copy=False).reshape(-1)
        if self._carry.size:
            audio = np.concatenate([self._carry, audio])
        else:
            self._carry_client_ts = client_ts
            self._carry_server_ts = server_ts

        events: list[VadEvent] = []
        offset = 0
        while offset + self.window_samples <= audio.size:
            window = audio[offset : offset + self.window_samples]
            events.extend(
                self._infer_window(window, self._carry_client_ts, self._carry_server_ts)
            )
            offset += self.window_samples
            self._carry_client_ts += 1000.0 * self.window_samples / RealtimeConfig.SAMPLE_RATE
            self._carry_server_ts = server_ts
        self._carry = audio[offset:]
        return events

    def _infer_window(
        self,
        window: np.ndarray,
        window_client_ts: float,
        window_server_ts: float,
    ) -> list[VadEvent]:
        if self.context_samples:
            effective = np.empty((self.context_samples + self.window_samples,), dtype=np.float32)
            effective[: self.context_samples] = self._context
            effective[self.context_samples :] = window
            self._context = effective[-self.context_samples :].copy()
            audio_in = effective
        else:
            audio_in = window

        feeds: dict[str, np.ndarray] = {}
        for name in self._input_names:
            if name == "input":
                feeds[name] = audio_in.reshape(1, -1)
            elif name == "state":
                feeds[name] = self._state
            elif name == "sr":
                feeds[name] = np.array([RealtimeConfig.SAMPLE_RATE], dtype=np.int64)

        outputs = self._session.run(self._output_names, feeds)  # type: ignore[union-attr]
        by_name = dict(zip(self._output_names, outputs, strict=True))
        prob = float(np.asarray(by_name.get("output", outputs[0])).reshape(-1)[0])
        state_key = "stateN" if "stateN" in by_name else self._output_names[-1]
        self._state = np.asarray(by_name[state_key], dtype=np.float32)
        self.last_prob = prob

        if prob >= self.threshold:
            if not self._triggered:
                self.peak_speech_ms = 0.0
            self.speech_ms += 1000.0 * self.window_samples / RealtimeConfig.SAMPLE_RATE
            self.peak_speech_ms = max(self.peak_speech_ms, self.speech_ms)
        else:
            self.speech_ms = 0.0

        events: list[VadEvent] = []
        if not self._triggered and prob >= self.threshold:
            self._triggered = True
            self._silence_windows = 0
            events.append(
                VadEvent("speech_started", prob, window_client_ts, window_server_ts)
            )
        elif self._triggered:
            if prob < self.end_threshold:
                self._silence_windows += 1
                if self._silence_windows >= self.min_silence_windows:
                    self._triggered = False
                    self._silence_windows = 0
                    events.append(
                        VadEvent("speech_ended", prob, window_client_ts, window_server_ts)
                    )
            else:
                self._silence_windows = 0
        return events
