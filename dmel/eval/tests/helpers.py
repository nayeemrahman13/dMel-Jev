"""Shared builders for the dmel/eval tests: label frames, shard fixtures, policies.

Everything here is hand-rolled — no other dmel area's code is imported.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def seq(n: int, changes: dict[int, object] | None, default: object) -> list:
    """Per-frame values: ``changes`` maps a start frame to a value held until the next change."""
    values = [default] * n
    if not changes:
        return values
    starts = sorted(changes)
    for i, start in enumerate(starts):
        end = starts[i + 1] if i + 1 < len(starts) else n
        for frame in range(start, end):
            values[frame] = changes[start]
    return values


def labels_df(
    n_steps: int,
    *,
    agent_speaking: dict[int, int] | None = None,
    scenario: dict[int, str] | None = None,
    interrupt_intent: dict[int, int] | None = None,
    earliest_reasonable_stop_ms: dict[int, int] | None = None,
    overlap_onset_ms: dict[int, int] | None = None,
    action: dict[int, str] | None = None,
    speech_present: dict[int, int] | None = None,
    primary_user: dict[int, int] | None = None,
    backchannel: dict[int, int] | None = None,
) -> pd.DataFrame:
    """Contract label columns for ``n_steps`` 50 ms frames; per-column change maps."""
    intent = seq(n_steps, interrupt_intent, 0)
    return pd.DataFrame(
        {
            "frame_index": list(range(n_steps)),
            "t_ms": [i * 50 for i in range(n_steps)],
            "agent_speaking": seq(n_steps, agent_speaking, 1),
            "speech_present": seq(n_steps, speech_present, 0),
            "primary_user": seq(n_steps, primary_user, 0),
            "backchannel": seq(n_steps, backchannel, 0),
            "interrupt_intent": intent,
            "action": seq(n_steps, action, "KEEP"),
            "earliest_reasonable_stop_ms": seq(n_steps, earliest_reasonable_stop_ms, -1),
            "overlap_onset_ms": seq(n_steps, overlap_onset_ms, -1),
            "scenario": seq(n_steps, scenario, "normal_turn"),
        }
    )


def write_sample(
    shard_dir: Path,
    sample_id: str,
    labels: pd.DataFrame,
    *,
    with_user_pcm: bool = True,
    with_agent_pcm: bool = True,
    with_dmel_cache: bool = False,
) -> None:
    """Write one sample's files in the contract shard layout."""
    shard_dir.mkdir(parents=True, exist_ok=True)
    labels.to_parquet(shard_dir / f"{sample_id}.labels.parquet", index=False)
    n_samples = len(labels) * 800
    if with_user_pcm:
        (shard_dir / f"{sample_id}.user.pcm").write_bytes(np.zeros(n_samples, dtype=np.int16).tobytes())
    if with_agent_pcm:
        (shard_dir / f"{sample_id}.agent.pcm").write_bytes(np.zeros(n_samples, dtype=np.int16).tobytes())
    if with_dmel_cache:
        np.save(shard_dir / f"{sample_id}.dmel.npy", np.zeros((len(labels), 2), dtype=np.int32))


# The canonical interruption sample used across metric tests (hand-computed in
# the test docstrings): event window [1200, 1900], stop point 1400, anchor 1200.
def interruption_sample(n_steps: int = 60) -> pd.DataFrame:
    return labels_df(
        n_steps,
        agent_speaking={0: 1},
        scenario={0: "normal_turn", 20: "interruption"},
        interrupt_intent={24: 1},
        earliest_reasonable_stop_ms={28: 1400},
        overlap_onset_ms={24: 1200},
        action={28: "STOP_TTS"},
        speech_present={24: 1},
        primary_user={24: 1},
    )


class ScriptedPolicy:
    """Contract-conforming scripted policy: STOP_TTS at/after ``stop_at_ms``.

    Counts steps itself after ``reset()`` (the interface carries no clock) and
    stops every frame from the threshold onward — enough to exercise the
    first-stop rule, recall windows, and the sweep frontier.
    """

    def __init__(self, stop_at_ms: float = 0):
        self.stop_at_ms = stop_at_ms
        self._step = 0

    def reset(self) -> None:
        self._step = 0

    def step(
        self,
        user_pcm_frame: "np.ndarray",
        dmel_token_ids: "np.ndarray | None",
        *,
        agent_speaking: bool,
        agent_pcm_frame: "np.ndarray | None" = None,
        speaker_similarity: "float | None" = None,
    ) -> dict:
        t_ms = self._step * 50
        self._step += 1
        action = "STOP_TTS" if t_ms >= self.stop_at_ms else "KEEP"
        return {"action": action, "probs": {"p_stop": 1.0 if action == "STOP_TTS" else 0.0}, "t_ms": t_ms}


class RecordingPolicy:
    """Records every step's inputs; stops when ``agent_speaking`` held true for the threshold."""

    def __init__(self, threshold_ms: float = 200):
        self.threshold_ms = threshold_ms
        self.run_ms = 0
        self.seen_token_ids: list = []
        self.n_steps = 0

    def reset(self) -> None:
        self.run_ms = 0

    def step(
        self,
        user_pcm_frame: "np.ndarray",
        dmel_token_ids: "np.ndarray | None",
        *,
        agent_speaking: bool,
        agent_pcm_frame: "np.ndarray | None" = None,
        speaker_similarity: "float | None" = None,
    ) -> dict:
        self.n_steps += 1
        self.seen_token_ids.append(dmel_token_ids)
        self.run_ms = self.run_ms + 50 if agent_speaking else 0
        action = "STOP_TTS" if self.run_ms >= self.threshold_ms else "KEEP"
        return {"action": action, "probs": {"p_stop": 1.0 if action == "STOP_TTS" else 0.0}}
