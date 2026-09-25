"""Per-frame label annotation from a planned :class:`SampleScript`.

One row per 50 ms frame (contract v1 label schema):

``frame_index, t_ms, agent_speaking, speech_present, primary_user,
backchannel, interrupt_intent, action ("KEEP"|"STOP_TTS"),
earliest_reasonable_stop_ms, overlap_onset_ms, scenario``

Contract v1 rules implemented here:

- ``agent_speaking``: run-level playback flag from the agent (TTS) layout —
  true for whole runs including intra-run inter-phrase gaps, false between
  runs. In truncated worlds the run's audible end is the cut, so the flag goes
  low shortly after the stop point and ``action`` latches back to KEEP.
- ``speech_present``: acoustic speech activity on the mix from ANY speaker,
  including the background speaker (1 there); noise events stay 0.
  Attribution lives in ``primary_user``.
- ``primary_user``: primary-user speech frames only.
- ``backchannel``: backchannel-type user utterances during agent speech.
- ``interrupt_intent``: genuine barge-in frames from utterance onset until
  ``earliest_reasonable_stop_ms`` (causal-decision labels).
- ``action``: STOP_TTS from the earliest-reasonable-stop timestamp through the
  end of the interrupted run in the sample's world; KEEP elsewhere (always
  KEEP when the agent is not speaking).
- ``earliest_reasonable_stop_ms`` / ``overlap_onset_ms``: event values on
  frames from the event onset through the end of the interrupted run, -1
  elsewhere. ``overlap_onset_ms`` is the physical anchor for latency eval;
  ``earliest_reasonable_stop_ms`` = onset + jittered evidence window.

The deployed ``agent_speaking`` input flag is legitimately available to models
but is not frame-exact in production; ``agent_flag_input_view`` derives the
jittered input stream the eval harness should feed policies (labels stay
stem-exact in this parquet).
"""

from __future__ import annotations

import numpy as np

from dmel.data.constants import FRAME_MS, FRAME_SAMPLES, MS_TO_SAMPLES
from dmel.data.scenarios import ROLE_AGENT, ROLE_USER, SampleScript

LABEL_COLUMNS = (
    "frame_index",
    "t_ms",
    "agent_speaking",
    "speech_present",
    "primary_user",
    "backchannel",
    "interrupt_intent",
    "action",
    "earliest_reasonable_stop_ms",
    "overlap_onset_ms",
    "scenario",
)


def n_frames_for(n_samples: int) -> int:
    return n_samples // FRAME_SAMPLES


def annotate(script: SampleScript, n_samples: int) -> dict[str, np.ndarray]:
    """Label arrays for a sample of ``n_samples`` (frame-aligned) audio."""
    n_frames = n_frames_for(n_samples)
    frame_starts = np.arange(n_frames, dtype=np.int64) * FRAME_SAMPLES
    frame_ends = frame_starts + FRAME_SAMPLES

    agent_speaking = np.zeros(n_frames, dtype=np.int8)
    speech_present = np.zeros(n_frames, dtype=np.int8)
    primary_user = np.zeros(n_frames, dtype=np.int8)
    backchannel = np.zeros(n_frames, dtype=np.int8)
    interrupt_intent = np.zeros(n_frames, dtype=np.int8)
    stop = np.zeros(n_frames, dtype=bool)
    stop_ms = np.full(n_frames, -1, dtype=np.int32)
    overlap_ms = np.full(n_frames, -1, dtype=np.int32)

    def spans_frames(start: int, end: int) -> np.ndarray:
        """Frames whose [start, end) window intersects the span."""
        return (frame_starts < end) & (frame_ends > start)

    for u in script.utterances:
        frames = spans_frames(u.start, u.end)
        if u.role == ROLE_AGENT:
            agent_speaking[frames] = 1
            speech_present[frames] = 1  # agent playback is audible on the mix
        elif u.role == ROLE_USER:
            speech_present[frames] = 1
            primary_user[frames] = 1
            if u.is_backchannel:
                backchannel[frames] = 1
        else:  # background speaker: acoustically speech, never primary user
            speech_present[frames] = 1

    for event in script.interruptions:
        stop_from = ((event.earliest_stop + FRAME_SAMPLES - 1) // FRAME_SAMPLES) * FRAME_SAMPLES
        run_frames = spans_frames(event.run_start, event.run_end) & (frame_starts >= stop_from)
        stop |= run_frames
        intent_frames = spans_frames(event.onset, event.earliest_stop)
        interrupt_intent[intent_frames] = 1
        marked = spans_frames(event.onset, event.run_end)
        stop_ms[marked] = event.earliest_stop // MS_TO_SAMPLES
        overlap_ms[marked] = event.onset // MS_TO_SAMPLES

    action = np.where(stop, "STOP_TTS", "KEEP")
    return {
        "frame_index": np.arange(n_frames, dtype=np.int32),
        "t_ms": (np.arange(n_frames, dtype=np.int32) * FRAME_MS),
        "agent_speaking": agent_speaking,
        "speech_present": speech_present,
        "primary_user": primary_user,
        "backchannel": backchannel,
        "interrupt_intent": interrupt_intent,
        "action": action,
        "earliest_reasonable_stop_ms": stop_ms,
        "overlap_onset_ms": overlap_ms,
        "scenario": np.full(n_frames, script.scenario, dtype=object),
    }


def agent_flag_input_view(
    agent_speaking_labels: np.ndarray,
    jitter_ms: int,
) -> np.ndarray:
    """Deployed ``agent_speaking`` input stream from the stem-exact labels.

    The protocol reports playback position at ~5 Hz with up to 450 ms of
    client buffering, so the deployed flag's transitions are imprecise. We
    model that by shifting every transition of the label column by
    ``jitter_ms`` (sampled per sample, recorded in the manifest). Positive
    jitter = the deployed flag rises late and falls late; negative = early.
    """
    jitter_frames = round(jitter_ms / FRAME_MS)
    if jitter_frames == 0:
        return agent_speaking_labels.copy()
    out = np.zeros_like(agent_speaking_labels)
    n = out.size
    changes = np.flatnonzero(agent_speaking_labels[1:] != agent_speaking_labels[:-1]) + 1
    bounds = [0, *(int(c) for c in changes), n]
    for i in range(len(bounds) - 1):
        lo, hi = bounds[i], bounds[i + 1]
        value = agent_speaking_labels[lo]
        new_lo, new_hi = lo + jitter_frames, hi + jitter_frames
        if value == 1:
            out[max(new_lo, 0): min(new_hi, n)] = 1
    return out
