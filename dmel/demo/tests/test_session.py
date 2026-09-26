"""DecisionSession, recorder, and arm-resolution behavior tests.

These use STUB policies (pure functions of the pushed frame) so session
semantics are tested without torch or onnxruntime. The real arms are
exercised separately: the loopback test covers the learned checkpoint, and
``test_baseline_contract_through_session`` covers arm A through the session.
"""

from __future__ import annotations

import numpy as np
import pytest

from dmel.demo._testutil import clip_with_burst
from dmel.demo.policies import (
    ARM_CHOICES,
    EXPECTED_PARAMS,
    build_policy,
    describe_arm,
    resolve_arm,
)
from dmel.demo.recorder import load_session, save_session
from dmel.demo.session import ACTION_KEEP, ACTION_STOP, DecisionSession
from dmel.features.tests._synth import silence


class StubPolicy:
    """Contract-shaped stub: STOPs twice, then KEEPs (latched-sustain style)."""

    def __init__(self) -> None:
        self.reset()
        self.calls: list[dict] = []

    def reset(self) -> None:
        self.calls = []

    def step(self, user_pcm_frame, dmel_token_ids, *, agent_speaking, agent_pcm_frame=None, speaker_similarity=None):
        self.calls.append({"frame": user_pcm_frame.copy(), "tokens": dmel_token_ids, "flag": agent_speaking})
        action = ACTION_STOP if len(self.calls) <= 2 else ACTION_KEEP
        return {
            "action": action,
            "probs": {"p_stop": 0.9 if action == ACTION_STOP else 0.1},
            "t_ms": (len(self.calls) - 1) * 50,
        }


def zeros() -> np.ndarray:
    return silence(50.0)


def test_first_push_queues_no_decision() -> None:
    session = DecisionSession(StubPolicy(), arm_label="a")
    assert session.push(zeros(), agent_speaking=True) == []
    assert session.steps_decided == 0


def test_frame_and_flag_pairing_through_frontend_lag() -> None:
    policy = StubPolicy()
    session = DecisionSession(policy, arm_label="a")
    # First push only queues; second push emits step 0's decision, which must
    # carry step 0's FRAME and step 0's FLAG (not the second push's).
    session.push(zeros(), agent_speaking=True)
    session.push(np.full(800, 5, dtype=np.int16), agent_speaking=False)
    assert session.steps_decided == 1
    decision = session.decisions[0]
    assert decision.step_index == 0
    assert decision.agent_speaking is True  # the FIRST frame's flag
    assert np.array_equal(policy.calls[0]["frame"], zeros())
    assert policy.calls[0]["flag"] is True


def test_stop_callback_fires_on_transition_only() -> None:
    fired: list[int] = []
    session = DecisionSession(
        StubPolicy(), arm_label="a", on_stop=lambda d: fired.append(d.step_index)
    )
    session.push(zeros(), agent_speaking=True)  # queue
    session.push(zeros(), agent_speaking=True)  # decision 0: STOP (fires)
    session.push(zeros(), agent_speaking=True)  # decision 1: STOP (sustain, no fire)
    session.push(zeros(), agent_speaking=True)  # decision 2: KEEP
    assert fired == [0]
    assert [d.stop_fired for d in session.decisions] == [True, False, False]
    assert session.decisions[2].action == ACTION_KEEP


def test_flag_only_passes_none_tokens() -> None:
    policy = StubPolicy()
    session = DecisionSession(policy, arm_label="c", flag_only=True)
    # Flag-only mode has no frontend and therefore no emission lag: every
    # push decides immediately, with dmel_token_ids=None for the policy's
    # documented flag-only fallback rule.
    session.push(zeros(), agent_speaking=True)
    assert session.steps_decided == 1
    assert session.decisions[0].tokens_provided is False
    assert policy.calls[0]["tokens"] is None


def test_reset_swaps_policy_and_clears_state() -> None:
    session = DecisionSession(StubPolicy(), arm_label="a")
    session.push(zeros(), agent_speaking=True)
    fresh = StubPolicy()
    session.reset(fresh, arm_label="c")
    assert session.policy is fresh
    assert session.arm_label == "c"
    assert session.steps_decided == 0
    assert fresh.calls == []  # the new policy was reset, never stepped


def test_non_contract_policy_results_are_rejected() -> None:
    class BadPolicy(StubPolicy):
        def step(self, *args, **kwargs):
            return {"action": "MAYBE", "probs": {"p_stop": 0.5}, "t_ms": 0}

    session = DecisionSession(BadPolicy(), arm_label="a")
    session.push(zeros(), agent_speaking=True)
    with pytest.raises(ValueError, match="non-contract action"):
        session.push(zeros(), agent_speaking=True)


def test_recorder_roundtrip(tmp_path) -> None:
    session = DecisionSession(StubPolicy(), arm_label="c", keep_frames=True)
    session.push(clip_with_burst()[:800], agent_speaking=True)
    session.push(clip_with_burst()[800:1600], agent_speaking=True)
    path = save_session(
        tmp_path / "session.npz",
        frames=session.frames,
        decisions=session.decisions,
        meta={"arms": ["c"], "voice": "en-us"},
    )
    frames, decisions, meta = load_session(path)
    assert frames.shape == (2, 800)
    assert [d["action"] for d in decisions] == [d.action for d in session.decisions]
    assert [d["t_ms"] for d in decisions] == [d.t_ms for d in session.decisions]
    assert meta["arms"] == ["c"]
    assert meta["schema"] == "dmel-demo-session-v1"


def test_resolve_arm_aliases() -> None:
    assert resolve_arm("c") == "c"
    assert resolve_arm("learned") == "c"
    assert resolve_arm("A") == "a"
    assert resolve_arm("baseline") == "a"
    with pytest.raises(ValueError, match="unknown arm"):
        resolve_arm("b")
    assert ARM_CHOICES == ("a", "c")
    assert "transformer" in describe_arm("c")
    assert "Silero" in describe_arm("a")


def test_learned_checkpoint_identity(checkpoint_path) -> None:
    pytest.importorskip("torch")
    policy = build_policy("c", checkpoint_path)
    assert policy.model.num_parameters() == EXPECTED_PARAMS
    assert policy.config.model.arm == "transformer"
    assert policy.config.model.d_model == 256
    # Contract flag-only fallback: a fresh policy must accept tokens=None.
    result = policy.step(zeros(), None, agent_speaking=True)
    assert result["action"] in (ACTION_KEEP, ACTION_STOP)
    assert "p_stop" in result["probs"]


def test_missing_checkpoint_error_names_recovery(checkpoint_path, tmp_path) -> None:
    with pytest.raises(FileNotFoundError, match="recover_checkpoint"):
        build_policy("c", tmp_path / "does-not-exist.pt")


def test_baseline_contract_through_session() -> None:
    pytest.importorskip("onnxruntime")
    policy = build_policy("a")
    session = DecisionSession(policy, arm_label="a")
    decisions: list = []
    for start in range(0, 1600, 800):  # two frames → first decision
        decisions.extend(
            session.push(clip_with_burst()[start : start + 800], agent_speaking=True)
        )
    assert session.steps_decided == 1
    decision = decisions[0]
    assert decision.action in (ACTION_KEEP, ACTION_STOP)
    assert decision.tokens_provided is True  # the session always provides tokens; arm A ignores them
