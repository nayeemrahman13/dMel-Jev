"""Shared audio, timing, voice, and augmentation constants for dmel.data.

All ranges are contract v1 values (docs/dmel_data_contract.md).
"""

from __future__ import annotations

import hashlib

SAMPLE_RATE = 16000
FRAME_MS = 50
FRAME_SAMPLES = 800  # 50 ms at 16 kHz
MS_TO_SAMPLES = SAMPLE_RATE // 1000  # 16

# ---------------------------------------------------------------------------
# Voice pools. Distinct espeak-ng 1.52 voices (verified by output hash;
# en-au/en-in/en-ie alias en-gb and are excluded). Voice pairs are the
# split cells' speaker dimension: user x agent with distinct timbres.
# ---------------------------------------------------------------------------

AGENT_VOICES = ("en-us", "en-gb", "en-gb-scotland", "en-gb-x-rp")
USER_VOICES = ("en-gb", "en-gb-scotland", "en-gb-x-rp", "en-us")
BACKGROUND_VOICES = ("en-gb-x-gbclan", "en-gb-x-gbcwmd")

VOICE_PAIRS = tuple(
    (user, agent)
    for user in USER_VOICES
    for agent in AGENT_VOICES
    if user != agent
)

# Fraction of voice pairs / template slots held out for val+test (~30%).
EVAL_ONLY_FRACTION = 0.30
# Fraction of eval samples that get the eval-only codec degradation family.
CODEC_EVAL_RATE = 0.5
# Fraction of samples whose agent_speaking INPUT flag is jittered.
FLAG_JITTER_RATE = 0.30
FLAG_JITTER_MS_RANGE = (-100, 100)  # signed; magnitude kept in [50, 100] when drawn
FLAG_JITTER_MIN_MS = 50
# Interruption/backchannel onsets placed at agent inter-phrase gaps.
GAP_ONSET_RATE = 0.30
# Interruption (and hesitation-commit) events generated in the truncated world.
TRUNCATED_WORLD_RATE = 0.5
TRUNC_GRACE_MS_RANGE = (200, 400)  # agent audio survives this long past the stop

# Discrete speaking-rate choices (espeak -s, words per minute). Discrete so
# the render cache stays hot without hurting variety.
RATE_CHOICES = (150, 155, 160, 165, 170, 175)

# Augmentation ranges. Background and noise levels are dB relative to the
# agent (playback) level.
SNR_DB_RANGE = (-6.0, 6.0)
BACKGROUND_DB_RANGE = (-18.0, -10.0)
NOISE_DB_RANGE = (-20.0, -12.0)
START_JITTER_MS_RANGE = (0, 150)
EVIDENCE_MS_RANGE = (150, 300)  # evidence window after user onset

SPLITS = ("train", "val", "test")
TRAIN_SEED_RANGE = (0, 1_000_000)  # [0, 1e6)
EVAL_SEED_RANGE = (1_000_000, 2_000_000)  # [1e6, 2e6)

# RNG substream tags (seed = [seed, tag] or [preset.seed, index, tag]).
TAG_STRUCTURE = 0
TAG_LEVELS = 1
TAG_NOISE = 2
TAG_CELL = 3


def stable_hash(text: str) -> int:
    """64-bit stable hash (sha1-truncated) — unlike builtin hash(), stable
    across processes, runs, and platforms. Used for cell/split assignment."""
    digest = hashlib.sha1(text.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little")


def augmentation_key(sample_id: str) -> int:
    """Augmentation RNG is keyed on sample id only (contract v1): the same
    sample id always draws the same augmentation stream, so no augmentation
    stream crosses the split."""
    return stable_hash(f"aug|{sample_id}")
