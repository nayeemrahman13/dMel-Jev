"""Deterministic synthetic token/label shards — TEST AND SMOKE SCAFFOLDING ONLY.

The ground-truth corpus is owned by the data engine (dmel/data/**, generated
by `python -m dmel.data.generate`). This module exists so models-area tests,
the CPU smoke, and the Modal GPU smoke can run before that PR merges: it
writes tokens and labels DIRECTLY (no audio, no features) in the contract's
shard layout, with token ids correlated to the labels so overfitting is
learnable.

Token ids are drawn from label-dependent ranges (plus noise), so a model can
genuinely lower its loss by reading the tokens — a decreasing smoke loss then
demonstrates the training loop learns, not that it memorizes noise.
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from dmel.models.config import FeaturesConfig

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
    "scenario",
)


def _synthetic_split(index: int) -> str:
    """Deterministic split assignment so loaders see manifest-driven splits."""
    position = index % 10
    if position < 7:
        return "train"
    if position < 9:
        return "val"
    return "test"


def _one_hot_spans(length: int, rng: np.random.Generator, count: int, min_len: int, max_len: int) -> list[tuple[int, int]]:
    spans = []
    for _ in range(count):
        start = int(rng.integers(0, max(1, length - min_len)))
        end = min(length, start + int(rng.integers(min_len, max_len)))
        spans.append((start, end))
    return spans


def synthesize_sample_labels(
    scenario: str,
    n_steps: int,
    rng: np.random.Generator,
) -> dict[str, np.ndarray]:
    """Label arrays for one synthetic sample, per the contract's semantics."""
    agent_speaking = np.zeros(n_steps, dtype=bool)
    speech_present = np.zeros(n_steps, dtype=bool)
    primary_user = np.zeros(n_steps, dtype=bool)
    backchannel = np.zeros(n_steps, dtype=bool)
    interrupt_intent = np.zeros(n_steps, dtype=bool)
    action = np.zeros(n_steps, dtype=np.int64)  # 0=KEEP, 1=STOP_TTS
    earliest = np.full(n_steps, -1, dtype=np.int64)

    agent_run = _one_hot_spans(n_steps, rng, 2, n_steps // 4, n_steps // 2)
    for start, end in agent_run:
        agent_speaking[start:end] = True
        if scenario == "backchannel":
            for _ in range(2):
                span = int(rng.integers(1, 3))
                pos = int(rng.integers(start, max(start + 1, end - span)))
                backchannel[pos : pos + span] = True
                speech_present[pos : pos + span] = True
        elif scenario == "interruption":
            onset = int(rng.integers(start + 2, max(start + 3, end)))
            evidence = int(rng.integers(3, 7))  # 150-300 ms jittered window
            stop_at = min(onset + evidence, end)
            # v1: ~50% truncated world — agent audio ends shortly after the
            # stop point, agent_speaking goes low, action latches KEEP again.
            run_end = end
            if rng.integers(0, 2) == 0:
                run_end = min(n_steps, stop_at + int(rng.integers(1, 3)))
                agent_speaking[start:end] = False
                agent_speaking[start:run_end] = True
            span = min(max(2, run_end - onset), n_steps - onset)
            interrupt_intent[onset : onset + span] = True
            speech_present[onset : onset + span] = True
            primary_user[onset : onset + span] = True
            earliest[onset:stop_at] = stop_at * 50
            action[stop_at:run_end] = 1  # through the run end, KEEP after
        elif scenario == "hesitation":
            pos = int(rng.integers(start, end))
            speech_present[pos : pos + 2] = True
            primary_user[pos : pos + 2] = True
        elif scenario == "normal_turn":
            pass
    if scenario == "normal_turn":
        start, end = agent_run[0]
        tail = min(end + int(rng.integers(2, 6)), n_steps)
        speech_present[end:tail] = True
        primary_user[end:tail] = True
    if scenario == "noise":
        for _ in range(3):
            pos = int(rng.integers(0, n_steps - 1))
            speech_present[pos : pos + 2] = False  # non-speech vocal event
    if scenario == "background_speaker":
        # Background speech: acoustic energy, but not the primary user.
        for _ in range(2):
            start = int(rng.integers(0, n_steps - 4))
            speech_present[start : start + 4] = False
            primary_user[:] = False

    return {
        "frame_index": np.arange(n_steps, dtype=np.int64),
        "t_ms": np.arange(n_steps, dtype=np.int64) * 50,
        "agent_speaking": agent_speaking.astype(np.int64),
        "speech_present": speech_present.astype(np.int64),
        "primary_user": primary_user.astype(np.int64),
        "backchannel": backchannel.astype(np.int64),
        "interrupt_intent": interrupt_intent.astype(np.int64),
        "action": action,
        "earliest_reasonable_stop_ms": earliest,
        "scenario": np.full(n_steps, scenario, dtype=object),
    }


def synthesize_sample_tokens(
    labels: dict[str, np.ndarray],
    tokens_per_step: int,
    vocab_size: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Token ids correlated with the labels so smoke overfitting can succeed."""
    n_steps = labels["action"].shape[0]
    noise = rng.integers(0, 4, size=(n_steps, tokens_per_step))
    base = np.where(
        (labels["action"][:, None] == 1) & (labels["interrupt_intent"][:, None] == 1),
        vocab_size - 1,
        np.where(
            labels["primary_user"][:, None] == 1,
            vocab_size // 2,
            np.where(labels["agent_speaking"][:, None] == 1, vocab_size // 4, 0),
        ),
    )
    span = max(1, vocab_size // 4)
    token_ids = (base + noise * (span // 4 + 1)) % vocab_size
    return token_ids.astype(np.int64)


# Scaffold token geometry — deliberately tiny; the smoke derives its model
# config from these when no fixture data is present. Real contract features
# use the FeaturesConfig defaults (vocab 320, 100 tokens/step).
SYNTHETIC_VOCAB_SIZE = 48
SYNTHETIC_TOKENS_PER_STEP = 4


def scaffold_features(features: "FeaturesConfig") -> "FeaturesConfig":
    """Model feature geometry matching write_synthetic_shard's default output.

    A model consuming synthetic shards must be built with the scaffold's
    token geometry or its input projection shape-checks against the wrong
    vocab/token count (the contract defaults are 320 x 100/step).
    """
    return replace(
        features,
        token_vocab_size=SYNTHETIC_VOCAB_SIZE,
        tokens_per_step=SYNTHETIC_TOKENS_PER_STEP,
    )


def write_synthetic_shard(
    root: Path,
    *,
    shard_name: str = "shard-synth",
    n_samples: int = 8,
    n_steps: int = 60,
    tokens_per_step: int = SYNTHETIC_TOKENS_PER_STEP,
    vocab_size: int = SYNTHETIC_VOCAB_SIZE,
    seed: int = 7,
    scenarios: tuple[str, ...] = (
        "interruption",
        "backchannel",
        "normal_turn",
        "noise",
        "background_speaker",
        "hesitation",
    ),
) -> Path:
    """Write a contract-layout shard of synthetic token/label samples."""
    rng = np.random.default_rng(seed)
    shard_dir = root / shard_name
    shard_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "seed": seed,
        "synthetic": True,
        "samples": n_samples,
        "steps_per_sample": n_steps,
    }
    for index in range(n_samples):
        scenario = scenarios[index % len(scenarios)]
        sample_rng = np.random.default_rng(seed + 1000 + index)
        labels = synthesize_sample_labels(scenario, n_steps, sample_rng)
        tokens = synthesize_sample_tokens(labels, tokens_per_step, vocab_size, sample_rng)
        sample_id = f"synth-{index:04d}"
        frame_data = {col: labels[col] for col in LABEL_COLUMNS}
        # Contract parquet stores the action as the "KEEP"/"STOP_TTS" string;
        # the int array is internal (token synthesis reads it).
        frame_data["action"] = np.where(
            labels["action"] == 1, "STOP_TTS", "KEEP"
        ).astype(object)
        frame = pd.DataFrame(frame_data)
        frame.to_parquet(shard_dir / f"{sample_id}.labels.parquet", index=False)
        np.save(shard_dir / f"{sample_id}.dmel.npy", tokens)
        manifest[sample_id] = {
            "split": _synthetic_split(index),
            "scenario": scenario,
        }
    (shard_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return shard_dir
