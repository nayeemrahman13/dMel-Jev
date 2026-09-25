"""Scripted-PCM tests for arm A against the vendored Silero ONNX.

Uses committed speech assets (``inject_speech.wav`` for a genuine barge-in,
``bench_ack_uh_huh.wav`` for a backchannel-like burst) so the cases are
deterministic and independent of the dMel data PR. Bounds are calibrated to
the wrapper's window mechanics: Silero scores 512-sample windows (32 ms
quanta), so the 200 ms duration gate fires ~224–256 ms after onset
(measured 250 ms on inject_speech).
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("onnxruntime")

from dmel.baselines.policy import FRAME_SAMPLES, BargeInPolicy
from dmel.runtime.audio import Pcm
from dmel.runtime.config import RealtimeConfig

LEAD_FRAMES = 10  # 500 ms of agent speech before the user comes in
ONSET_MS = LEAD_FRAMES * 50
STOP_LATENCY_BUDGET_MS = 300.0


def _to_frames(pcm: np.ndarray, *, pad_last: bool = True) -> list[np.ndarray]:
    frames = [pcm[i : i + FRAME_SAMPLES] for i in range(0, pcm.size, FRAME_SAMPLES)]
    if pad_last and frames and frames[-1].size < FRAME_SAMPLES:
        frames[-1] = np.pad(frames[-1], (0, FRAME_SAMPLES - frames[-1].size))
    return frames


def _silence_frames(count: int) -> list[np.ndarray]:
    return [np.zeros((FRAME_SAMPLES,), dtype=np.int16) for _ in range(count)]


@pytest.fixture(scope="module")
def inject_frames() -> list[np.ndarray]:
    return _to_frames(Pcm.load_wav_i16(RealtimeConfig.INJECT_WAV))


def test_clear_barge_in_stops_within_latency_budget(inject_frames: list[np.ndarray]) -> None:
    policy = BargeInPolicy()
    stream = _silence_frames(LEAD_FRAMES) + inject_frames
    actions = [
        policy.step(frame, None, agent_speaking=True)["action"] for frame in stream
    ]
    pre_onset = actions[:LEAD_FRAMES]
    assert all(a == "KEEP" for a in pre_onset), pre_onset
    stops = [i * 50 for i, a in enumerate(actions) if a == "STOP_TTS"]
    assert stops, "no STOP_TTS emitted for a sustained barge-in"
    latency = stops[0] - ONSET_MS
    assert 0 < latency <= STOP_LATENCY_BUDGET_MS, f"first STOP at t_ms={stops[0]}"


def test_short_backchannel_burst_does_not_stop() -> None:
    policy = BargeInPolicy()
    ack = Pcm.load_wav_i16(RealtimeConfig.ASSETS_DIR / "bench_ack_uh_huh.wav")
    burst = np.zeros((3 * FRAME_SAMPLES,), dtype=np.int16)
    burst[: min(ack.size, 150 * 16)] = ack[: min(ack.size, 150 * 16)]
    stream = _silence_frames(LEAD_FRAMES) + _to_frames(burst) + _silence_frames(20)
    actions = [
        policy.step(frame, None, agent_speaking=True)["action"] for frame in stream
    ]
    assert "STOP_TTS" not in actions, [
        (i * 50, a) for i, a in enumerate(actions) if a == "STOP_TTS"
    ]


def test_silence_during_agent_speech_always_keeps() -> None:
    policy = BargeInPolicy()
    for frame in _silence_frames(30):
        result = policy.step(frame, None, agent_speaking=True)
        assert result["action"] == "KEEP"


def test_stop_latches_until_agent_speaking_goes_low(
    inject_frames: list[np.ndarray],
) -> None:
    policy = BargeInPolicy()
    for frame in _silence_frames(LEAD_FRAMES) + inject_frames[:8]:
        policy.step(frame, None, agent_speaking=True)
    silence = np.zeros((FRAME_SAMPLES,), dtype=np.int16)
    latched = policy.step(silence, None, agent_speaking=True)
    assert latched["action"] == "STOP_TTS", (
        "latch must hold through VAD silence while the agent speaks"
    )
    cleared = policy.step(silence, None, agent_speaking=False)
    assert cleared["action"] == "KEEP"


def test_reset_restores_initial_state(inject_frames: list[np.ndarray]) -> None:
    policy = BargeInPolicy()
    for frame in _silence_frames(LEAD_FRAMES) + inject_frames[:8]:
        policy.step(frame, None, agent_speaking=True)
    policy.reset()
    result = policy.step(np.zeros((FRAME_SAMPLES,), dtype=np.int16), None, agent_speaking=True)
    assert result["action"] == "KEEP"
    assert result["t_ms"] == 0
    assert policy.latched is False


def test_step_result_contract(inject_frames: list[np.ndarray]) -> None:
    policy = BargeInPolicy()
    result = policy.step(inject_frames[0], None, agent_speaking=True)
    assert set(result) == {"action", "probs", "t_ms"}
    assert result["action"] in {"KEEP", "STOP_TTS"}
    assert isinstance(result["t_ms"], int)
    # v1 freezes arm A's probs schema to exactly one key.
    assert set(result["probs"]) == {"p_stop"}
    assert 0.0 <= result["probs"]["p_stop"] <= 1.0


def test_p_stop_saturates_exactly_at_stop(inject_frames: list[np.ndarray]) -> None:
    policy = BargeInPolicy()
    keep_probs, stop_probs = [], []
    for frame in _silence_frames(LEAD_FRAMES) + inject_frames:
        result = policy.step(frame, None, agent_speaking=True)
        (stop_probs if result["action"] == "STOP_TTS" else keep_probs).append(
            result["probs"]["p_stop"]
        )
    assert stop_probs and all(p == 1.0 for p in stop_probs), stop_probs
    assert keep_probs and all(p < 1.0 for p in keep_probs), keep_probs


def test_truncated_world_agent_drop_clears_stop(inject_frames: list[np.ndarray]) -> None:
    """Truncated world: agent_speaking goes low shortly after the stop point —
    the latch must clear and the policy must return KEEP (nothing left to
    cancel)."""
    policy = BargeInPolicy()
    stream = _silence_frames(LEAD_FRAMES) + inject_frames[:8]
    actions = [policy.step(frame, None, agent_speaking=True)["action"] for frame in stream]
    assert "STOP_TTS" in actions, actions
    # The truncation transition: playback state drops right after the stop.
    truncated = policy.step(
        np.zeros((FRAME_SAMPLES,), dtype=np.int16), None, agent_speaking=False
    )
    assert truncated["action"] == "KEEP"
    assert truncated["probs"]["p_stop"] == 0.0
    assert policy.latched is False
