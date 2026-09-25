"""Arm A: Silero VAD + duration-threshold heuristic in the contract shape.

Uses the vendored Silero ONNX wrapper (``dmel.runtime.silero``) and the
vendored ``silero_vad.onnx`` — the wrapper already buffers arbitrary chunks
into 512-sample windows with 64-sample context and tracks the continuous
above-threshold speech duration (``speech_ms``) the heuristic gates on.

Contract interface (one call per 50 ms step, 800-sample int16 frames):

    policy.reset()
    policy.step(user_pcm_frame, dmel_token_ids, *, agent_speaking,
                agent_pcm_frame=None, speaker_similarity=None)
        -> {"action": "KEEP"|"STOP_TTS", "probs": dict, "t_ms": int}

Arm A ignores ``dmel_token_ids``; ``agent_pcm_frame`` / ``speaker_similarity``
are accepted but unused (plumbing reserved for ablation arms D/E).

v1 contract points honored here:

- ``probs`` is frozen to exactly ``{"p_stop": float}`` — a monotone heuristic
  score: ``min(speech_ms / stop_threshold_ms, 1.0)``, saturated at 1.0 whenever
  the action is ``STOP_TTS``. Sweeping a p_stop cutoff ``c`` in ``[0, 1)``
  reproduces duration thresholds of ``c * stop_threshold_ms``, so the eval
  harness can sweep operating points uniformly; the policy's own action
  corresponds to the cutoff at 1.0 (the default operating point).
- Truncated-world samples: ``agent_speaking`` goes low shortly after the stop
  point. The latch clears on that transition, so the policy returns ``KEEP``
  from there — nothing left to cancel.

Known limitation, documented for eval: ``user_pcm_frame`` is what the
microphone hears, so on the synthetic corpus it contains the agent's own
playback. A raw VAD cannot separate that from user speech, so the duration
gate is what bounds — but does not eliminate — false stops; the eval harness
quantifies them per scenario.
"""

from __future__ import annotations

import numpy as np

from dmel.baselines.heuristic import KEEP, STOP_TTS, BargeInHeuristic
from dmel.runtime.silero import SileroVAD

FRAME_SAMPLES = 800
FRAME_MS = 50


class SileroVADPolicy:
    """Arm A baseline policy conforming to the contract ``BargeInPolicy``."""

    def __init__(self, vad: SileroVAD | None = None, stop_threshold_ms: float = 200.0) -> None:
        self._vad = vad if vad is not None else SileroVAD.load()
        self._heuristic = BargeInHeuristic(stop_threshold_ms=stop_threshold_ms)
        self._step_index = 0
        # Per-step diagnostics for decision logs (not part of the frozen probs).
        self.last_speech_prob = 0.0
        self.last_speech_ms = 0.0

    @property
    def stop_threshold_ms(self) -> float:
        return self._heuristic.stop_threshold_ms

    @property
    def latched(self) -> bool:
        return self._heuristic.latched

    def reset(self) -> None:
        self._vad.reset()
        self._heuristic.reset()
        self._step_index = 0
        self.last_speech_prob = 0.0
        self.last_speech_ms = 0.0

    def step(
        self,
        user_pcm_frame: np.ndarray,  # 800 int16 samples
        dmel_token_ids: np.ndarray | None,  # ignored: arm A is audio-only
        *,
        agent_speaking: bool,
        agent_pcm_frame: np.ndarray | None = None,  # unused: arms D/E plumbing
        speaker_similarity: float | None = None,  # unused: arms D/E plumbing
    ) -> dict:
        audio = _to_f32(user_pcm_frame)
        t_ms = self._step_index * FRAME_MS
        # The wrapper buffers internally; 800-sample steps only sometimes align
        # with its 512-sample windows. Timestamps only feed its event labels,
        # which arm A does not consume.
        self._vad.push(audio, float(t_ms), float(t_ms))
        self.last_speech_prob = float(self._vad.last_prob)
        self.last_speech_ms = float(self._vad.speech_ms)
        action = self._heuristic.decide(agent_speaking=agent_speaking, speech_ms=self.last_speech_ms)
        # Frozen v1 probs schema: p_stop is arm A's calibrated heuristic score —
        # the duration ratio, saturated at 1.0 exactly when the action is
        # STOP_TTS. A p_stop cutoff sweep in [0, 1) reproduces duration-threshold
        # sweeps, which is what the eval's validation-split selection needs.
        # A forced KEEP (agent silent — e.g. the truncated-world drop) scores 0:
        # no stop is possible while nothing is playing.
        if action == STOP_TTS:
            p_stop = 1.0
        elif not agent_speaking:
            p_stop = 0.0
        else:
            p_stop = min(self.last_speech_ms / self._heuristic.stop_threshold_ms, 1.0)
        self._step_index += 1
        return {"action": action, "probs": {"p_stop": p_stop}, "t_ms": t_ms}


def _to_f32(frame: np.ndarray) -> np.ndarray:
    """PCM int16 → float32 in [-1, 1); float input is assumed already scaled."""
    samples = np.asarray(frame).reshape(-1)
    if samples.dtype.kind in "iub":
        return samples.astype(np.float32) / 32768.0
    return samples.astype(np.float32)


# Contract name for harnesses that instantiate policies by interface.
BargeInPolicy = SileroVADPolicy

__all__ = ["BargeInPolicy", "KEEP", "STOP_TTS", "SileroVADPolicy"]
