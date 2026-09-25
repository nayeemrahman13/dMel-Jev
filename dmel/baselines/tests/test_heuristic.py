"""Unit tests for the pure barge-in decision state machine."""

from __future__ import annotations

import pytest

from dmel.baselines.heuristic import KEEP, STOP_TTS, BargeInHeuristic


def test_keep_when_agent_not_speaking_even_with_long_speech() -> None:
    h = BargeInHeuristic()
    assert h.decide(agent_speaking=False, speech_ms=10_000.0) == KEEP
    # Agent went quiet: any earlier latch must be cleared.
    assert h.latched is False


def test_stop_fires_only_after_duration_exceeds_threshold() -> None:
    h = BargeInHeuristic(stop_threshold_ms=200.0)
    assert h.decide(agent_speaking=True, speech_ms=199.9) == KEEP
    assert h.decide(agent_speaking=True, speech_ms=200.1) == STOP_TTS
    assert h.latched is True


def test_latch_holds_through_silence_until_agent_goes_quiet() -> None:
    h = BargeInHeuristic(stop_threshold_ms=200.0)
    h.decide(agent_speaking=True, speech_ms=256.0)
    # User pauses mid-barge-in; decision must not un-stop while the agent speaks.
    assert h.decide(agent_speaking=True, speech_ms=0.0) == STOP_TTS
    assert h.decide(agent_speaking=True, speech_ms=32.0) == STOP_TTS
    # Agent playback state goes low: latch clears and the action is KEEP.
    assert h.decide(agent_speaking=False, speech_ms=0.0) == KEEP
    assert h.latched is False


def test_rearmed_latch_requires_fresh_speech_duration() -> None:
    h = BargeInHeuristic(stop_threshold_ms=200.0)
    h.decide(agent_speaking=True, speech_ms=256.0)
    h.decide(agent_speaking=False, speech_ms=0.0)
    assert h.decide(agent_speaking=True, speech_ms=160.0) == KEEP
    assert h.decide(agent_speaking=True, speech_ms=224.0) == STOP_TTS


def test_threshold_is_sweepable() -> None:
    low = BargeInHeuristic(stop_threshold_ms=100.0)
    high = BargeInHeuristic(stop_threshold_ms=500.0)
    assert low.decide(agent_speaking=True, speech_ms=150.0) == STOP_TTS
    assert high.decide(agent_speaking=True, speech_ms=150.0) == KEEP


def test_reset_clears_latch() -> None:
    h = BargeInHeuristic()
    h.decide(agent_speaking=True, speech_ms=300.0)
    assert h.latched is True
    h.reset()
    assert h.latched is False
    assert h.decide(agent_speaking=True, speech_ms=0.0) == KEEP


def test_rejects_non_positive_threshold() -> None:
    with pytest.raises(ValueError):
        BargeInHeuristic(stop_threshold_ms=0.0)
    with pytest.raises(ValueError):
        BargeInHeuristic(stop_threshold_ms=-50.0)
