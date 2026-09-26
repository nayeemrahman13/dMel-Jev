"""The demo's decision loop — the ONE "live path" implementation.

``DecisionSession`` pairs a streaming featurizer with a contract
``BargeInPolicy`` and turns per-frame pushes into per-step decisions. The
same class runs the live mic demo, the session recorder, and the
corpus-loopback test, so "what the test verified" and "what the demo runs"
are the same code by construction.

Frame/token pairing: the frontend's emission lags one frame (its final
subframe window overlaps 240 samples into the next step — see
``dmel.demo.frontend``), so each pushed frame is queued alongside the
``agent_speaking`` flag it was pushed with, and a queued (frame, flag) pair
is consumed exactly when the featurizer emits that step's tokens. The flag is
bound to the frame it accompanies (label-row semantics, matching the eval
harness), not to the moment the decision happens to run.

Recording: with ``keep_frames`` the session retains every mic frame plus the
per-step decision log; ``dmel.demo.recorder.save_session`` dumps them for
offline scoring.
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

from dmel.demo.frontend import StreamingFeaturizer, validate_frame

ACTION_KEEP = "KEEP"
ACTION_STOP = "STOP_TTS"

# Called once per step that FIRES a stop (transition into a STOP latch) —
# the live demo aborts playback here. Receives the firing decision.
OnStopCallback = Callable[["StepDecision"], None]


@dataclass(frozen=True)
class StepDecision:
    """One policy decision, plus what the demo needs to log and score it."""

    step_index: int
    t_ms: int
    action: str  # contract: "KEEP" | "STOP_TTS"
    probs: dict[str, float]  # frozen contract schema (arm-dependent key set)
    agent_speaking: bool
    tokens_provided: bool  # False = the contract flag-only fallback fired
    stop_fired: bool  # True exactly on the step that transitioned into STOP_TTS
    dec_ms: float  # wall time the policy+frontend spent on this decision
    wall_s: float  # seconds since session start (live pacing context)

    @property
    def p_stop(self) -> float:
        return float(self.probs["p_stop"])


@dataclass
class _QueuedStep:
    """A mic frame awaiting its step's tokens, with the flag bound to it."""

    frame: np.ndarray
    agent_speaking: bool
    agent_pcm_frame: np.ndarray | None
    speaker_similarity: float | None


class DecisionSession:
    """Stream mic frames through a policy, one contract decision per 50 ms step.

    ``policy`` is any contract ``BargeInPolicy`` (learned arm C or baseline
    arm A). ``on_stop`` fires on the STOP transition (not while the latch
    sustains). ``flag_only`` runs the contract v1.1 fallback path — the
    policy receives ``dmel_token_ids=None`` every step and must degrade to
    its documented flag-only rule (no new stops; latch sustain only).
    """

    def __init__(
        self,
        policy: Any,
        *,
        arm_label: str,
        on_stop: OnStopCallback | None = None,
        flag_only: bool = False,
        keep_frames: bool = False,
    ) -> None:
        self.policy = policy
        self.arm_label = arm_label
        self.on_stop = on_stop
        self.flag_only = flag_only
        self._featurizer = StreamingFeaturizer()
        self._queue: deque[_QueuedStep] = deque()
        self._last_action = ACTION_KEEP
        self._started = time.perf_counter()
        self.decisions: list[StepDecision] = []
        self.frames: list[np.ndarray] | None = [] if keep_frames else None

    @property
    def steps_decided(self) -> int:
        return len(self.decisions)

    def reset(self, policy: Any | None = None, *, arm_label: str | None = None) -> None:
        """Fresh run: reset (or swap) the policy and drop all stream state.

        Used when the user toggles arms mid-session — the new brain starts
        with an empty context window and a fresh frontend, never the previous
        arm's stale state.
        """
        if policy is not None:
            self.policy = policy
        if arm_label is not None:
            self.arm_label = arm_label
        self.policy.reset()
        self._featurizer = StreamingFeaturizer()
        self._queue.clear()
        self._last_action = ACTION_KEEP
        self.decisions = []

    def push(
        self,
        mic_frame: np.ndarray,
        *,
        agent_speaking: bool,
        agent_pcm_frame: np.ndarray | None = None,
        speaker_similarity: float | None = None,
    ) -> list[StepDecision]:
        """Push one 800-sample mic frame; return the decisions it produced (0 or 1)."""
        frame = validate_frame(mic_frame)
        self._queue.append(
            _QueuedStep(
                frame=np.array(frame, dtype=np.int16, copy=True),
                agent_speaking=bool(agent_speaking),
                agent_pcm_frame=None if agent_pcm_frame is None else np.asarray(agent_pcm_frame),
                speaker_similarity=speaker_similarity,
            )
        )
        if self.frames is not None:
            self.frames.append(np.array(frame, dtype=np.int16, copy=True))
        tokens = None if self.flag_only else self._featurizer.push_frame(frame)
        if tokens is None and not self.flag_only:
            # Frontend emission lag: no decision this push. (In flag-only
            # mode there is no frontend, so every push decides immediately —
            # the contract's fallback rule must still produce decisions.)
            return []
        return [self._decide(tokens)]

    def flush(self) -> list[StepDecision]:
        """End of stream: decide the remaining steps (front-end zero-pad tail).

        Every remaining token step pairs with its queued frame. Frames left
        over (streams that do not end on a whole 50 ms step) decide via the
        contract flag-only fallback — the offline step count defines how many
        token steps exist, and the frontend never invents audio for them.
        """
        out: list[StepDecision] = []
        for tokens in [] if self.flag_only else self._featurizer.flush():
            if not self._queue:  # more tokens than frames: cannot happen for whole-step streams
                raise RuntimeError(
                    "featurizer emitted a token step with no queued mic frame — "
                    "the stream must be fed whole 800-sample frames"
                )
            out.append(self._decide_with(tokens, self._queue.popleft()))
        while self._queue:  # tail frames with no token step: flag-only fallback
            out.append(self._decide_with(None, self._queue.popleft()))
        return out

    def _decide(self, tokens: np.ndarray | None) -> StepDecision:
        return self._decide_with(tokens, self._queue.popleft())

    def _decide_with(self, tokens: np.ndarray | None, queued: _QueuedStep) -> StepDecision:
        started = time.perf_counter()
        result = self.policy.step(
            queued.frame,
            tokens,
            agent_speaking=queued.agent_speaking,
            agent_pcm_frame=queued.agent_pcm_frame,
            speaker_similarity=queued.speaker_similarity,
        )
        dec_ms = (time.perf_counter() - started) * 1000.0

        action = str(result["action"]).upper()
        if action not in (ACTION_KEEP, ACTION_STOP):
            raise ValueError(f"policy returned non-contract action {result['action']!r}")
        probs = dict(result["probs"])
        if "p_stop" not in probs:
            raise ValueError(f"policy probs missing frozen 'p_stop' key: {sorted(probs)}")
        stop_fired = action == ACTION_STOP and self._last_action != ACTION_STOP
        self._last_action = action
        decision = StepDecision(
            step_index=len(self.decisions),
            t_ms=int(result["t_ms"]),
            action=action,
            probs=probs,
            agent_speaking=queued.agent_speaking,
            tokens_provided=tokens is not None,
            stop_fired=stop_fired,
            dec_ms=dec_ms,
            wall_s=time.perf_counter() - self._started,
        )
        self.decisions.append(decision)
        if stop_fired and self.on_stop is not None:
            self.on_stop(decision)
        return decision
