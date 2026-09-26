"""Streaming-frontend integrity tests: causality, geometry, offline equivalence.

The demo's entire premise rests on the streaming featurizer producing the
SAME token ids the offline path produced at training and eval time. These
tests pin that three ways:

1. stream a whole clip frame by frame and compare against ``tokenize()``;
2. stream the committed fixture MIX stems and compare against the committed
   ``.dmel.npy`` caches (the exact artifacts training and eval consumed);
3. assert causality structurally — a step's tokens depend only on samples
   through the documented 1040-sample window, and nothing is emitted from
   samples that have not arrived.
"""

from __future__ import annotations

import numpy as np
import pytest

from dmel.demo._testutil import clip_with_burst, fixture_cache, fixture_mix, fixture_flags
from dmel.demo.frontend import FRAME_SAMPLES, StreamingFeaturizer
from dmel.demo.session import ACTION_KEEP, ACTION_STOP, DecisionSession
from dmel.features.config import DmelConfig
from dmel.features.tokenizer import tokenize


def _frames(pcm: np.ndarray):
    """Yield the clip as 50 ms frames, zero-padding the ragged tail."""
    for start in range(0, pcm.shape[0], FRAME_SAMPLES):
        frame = pcm[start : start + FRAME_SAMPLES]
        if frame.shape[0] < FRAME_SAMPLES:
            frame = np.pad(frame, (0, FRAME_SAMPLES - frame.shape[0]))
        yield frame


def test_streaming_matches_offline_tokenize() -> None:
    pcm = clip_with_burst()
    offline = tokenize(pcm, DmelConfig())

    featurizer = StreamingFeaturizer()
    streamed: list[np.ndarray] = []
    for frame in _frames(pcm):
        tokens = featurizer.push_frame(frame)
        if tokens is not None:
            streamed.append(tokens)
    streamed.extend(featurizer.flush())

    assert len(streamed) == offline.shape[0]
    for step_tokens, offline_tokens in zip(streamed, offline):
        assert np.array_equal(step_tokens, offline_tokens)


def test_streaming_matches_committed_fixture_caches(fixtures_dir) -> None:
    for sample_id in ("fixtures-interruption-0000", "fixtures-normal_turn-0000"):
        cache = fixture_cache(fixtures_dir, sample_id)
        pcm = fixture_mix(fixtures_dir, sample_id)

        featurizer = StreamingFeaturizer()
        streamed: list[np.ndarray] = []
        for frame in _frames(pcm):
            tokens = featurizer.push_frame(frame)
            if tokens is not None:
                streamed.append(tokens)
        streamed.extend(featurizer.flush())

        assert np.array_equal(np.stack(streamed), cache)


def test_frontend_is_causal_no_lookahead() -> None:
    """Nothing may be emitted before frame t+1 (the 1040-sample window)."""
    pcm = clip_with_burst()
    featurizer = StreamingFeaturizer()
    emitted_on: dict[int, int] = {}
    for index, frame in enumerate(_frames(pcm)):
        tokens = featurizer.push_frame(frame)
        if tokens is not None:
            emitted_on[featurizer.next_step - 1] = index
    for step, push_index in emitted_on.items():
        assert push_index >= 1  # step 0 emits on push of frame 1 at the earliest
        assert push_index == step + 1  # and each step emits exactly one frame later


def test_stream_decision_skew_is_zero(fixtures_dir) -> None:
    """THE pipeline-integrity check: live decisions == offline decisions.

    Same mic signal (the fixture mix stem — what a real microphone hears),
    same policy, same agent_speaking flags. The only difference is where the
    token ids come from: the offline arm reads the committed cache, the live
    arm runs the streaming frontend frame by frame. The learned policy runs
    over both token sets, so any streaming skew would flip a decision here.
    """
    from dmel.demo.policies import build_policy

    pytest.importorskip("torch")
    sample_id = "fixtures-interruption-0000"
    pcm = fixture_mix(fixtures_dir, sample_id)
    cache = fixture_cache(fixtures_dir, sample_id)
    flags = fixture_flags(fixtures_dir, sample_id)
    n_steps = len(flags)

    # Offline arm: cache tokens, straight contract calls (the eval pattern).
    offline_policy = build_policy("c")
    offline_actions: list[str] = []
    for i in range(n_steps):
        result = offline_policy.step(
            pcm[i * FRAME_SAMPLES : (i + 1) * FRAME_SAMPLES],
            cache[i],
            agent_speaking=flags[i],
        )
        offline_actions.append(str(result["action"]).upper())

    # Live arm: the demo's own session (streaming frontend + queue pairing).
    live_session = DecisionSession(build_policy("c"), arm_label="c")
    live_actions: list[str] = []
    for i in range(n_steps):
        for decision in live_session.push(
            pcm[i * FRAME_SAMPLES : (i + 1) * FRAME_SAMPLES],
            agent_speaking=flags[i],
        ):
            live_actions.append(decision.action)
    for decision in live_session.flush():
        live_actions.append(decision.action)

    assert live_actions == offline_actions
    assert set(live_actions) <= {ACTION_KEEP, ACTION_STOP}
