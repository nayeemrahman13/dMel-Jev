"""Barge-in decision logic for arm A (VAD + duration threshold).

Pure state machine — no audio or ONNX dependencies — so the decision rules are
unit-testable in isolation. Per the dMel contract:

    agent_speaking AND speech AND continuous speech duration > threshold
    => STOP_TTS, latched until agent_speaking goes low.

The continuous-speech duration is measured by the VAD wrapper (``speech_ms``);
this module only decides. The duration threshold is constructor-injectable so
eval sweeps can vary it.
"""

from __future__ import annotations

KEEP = "KEEP"
STOP_TTS = "STOP_TTS"


class BargeInHeuristic:
    """Latch-style barge-in decision from per-step VAD signals.

    Hysteresis contract: once ``STOP_TTS`` fires, the decision stays latched
    while the agent keeps speaking — even if VAD speech drops (a pause
    mid-barge-in must not un-stop playback) — and clears as soon as
    ``agent_speaking`` goes low. While the agent is not speaking the action is
    always ``KEEP`` (V1 ignores turn-end detection).
    """

    def __init__(self, stop_threshold_ms: float = 200.0) -> None:
        if stop_threshold_ms <= 0:
            raise ValueError(f"stop_threshold_ms must be positive, got {stop_threshold_ms}")
        self.stop_threshold_ms = stop_threshold_ms
        self.latched = False

    def reset(self) -> None:
        self.latched = False

    def decide(self, *, agent_speaking: bool, speech_ms: float) -> str:
        """Return the action for one 50 ms step.

        ``speech_ms`` is the VAD's continuous above-threshold speech duration
        in milliseconds for the current stream position.
        """
        if not agent_speaking:
            self.latched = False
        elif not self.latched and speech_ms > self.stop_threshold_ms:
            self.latched = True
        return STOP_TTS if (agent_speaking and self.latched) else KEEP
