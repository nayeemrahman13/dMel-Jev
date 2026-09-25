"""Scenario planning for dmel.data: cells, splits, and sample layouts.

Planning happens against measured utterance audio: planners receive a
``render(text, voice, rate) -> np.ndarray`` callable (the espeak adapter) and
query it lazily, so placement uses true synthesized durations rather than
estimates. All timing is integer samples (16 samples = 1 ms at 16 kHz), which
keeps every derived label exact.

Contract v1 structures implemented here:

- **Cells & splits.** A cell is ``(voice_pair, template_slot)``. ~30% of voice
  pairs and ~30% of template slots within each semantic class are held out for
  val+test; cells containing any held-out component are hash-assigned to
  val/test (50/50), all other cells go to train. The assignment is a pure
  function of (salt, scenario, vp, slot) — same cell, same split, always.
  Realized split masses are reported in each shard manifest.
- **Seeds.** Train samples draw structural randomness from [0, 1e6), eval
  samples from [1e6, 2e6) (contract "Splits").
- **Agent runs.** ``agent_speaking`` is a run-level playback flag: true for the
  whole run including intra-run inter-phrase gaps (playback state has not
  changed), false between runs.
- **Truncated world.** ~50% of interruption (and hesitation-commit) events cut
  the agent audio shortly after ``earliest_reasonable_stop_ms`` (a successful
  stop); the rest play the full run (a failed stop).
- **Inter-phrase-gap onsets.** ~30% of interruption and backchannel onsets sit
  in an intra-run gap; the flag stays true because playback has not ended.
- **Censored hesitation pairs.** Two samples sharing byte-identical audio up
  to the censoring point: the hesitation variant never commits, the
  ``hesitation_commit`` partner flows into real overlap and is labeled as an
  interruption from the commit point.

RNG discipline: structure and levels use named substreams (``[seed, tag]``);
draw order within a planner is fixed, so every output is reproducible.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Callable

import numpy as np

from dmel.data import text_banks as tb
from dmel.data.constants import (
    BACKGROUND_DB_RANGE,
    BACKGROUND_VOICES,
    EVIDENCE_MS_RANGE,
    EVAL_ONLY_FRACTION,
    FLAG_JITTER_MIN_MS,
    FLAG_JITTER_MS_RANGE,
    FLAG_JITTER_RATE,
    GAP_ONSET_RATE,
    MS_TO_SAMPLES,
    NOISE_DB_RANGE,
    RATE_CHOICES,
    SNR_DB_RANGE,
    START_JITTER_MS_RANGE,
    TRAIN_SEED_RANGE,
    TRUNCATED_WORLD_RATE,
    TRUNC_GRACE_MS_RANGE,
    VOICE_PAIRS,
    augmentation_key,
    stable_hash,
)

RenderFn = Callable[[str, str, int], np.ndarray]

ROLE_AGENT = "agent"
ROLE_USER = "primary_user"
ROLE_BACKGROUND = "background"


@dataclass
class UtteranceSpan:
    role: str
    text: str
    start: int  # samples, inclusive
    end: int  # samples, exclusive (clamped by truncation)
    is_backchannel: bool = False
    run_index: int = -1  # agent utterances carry their run id


@dataclass
class NoiseSpan:
    kind: str  # "cough" | "breath" | "burst"
    start: int
    end: int


@dataclass
class RunSpan:
    start: int  # audible run start
    end: int  # audible run end (clamped by truncation)
    run_index: int


@dataclass
class InterruptionEvent:
    onset: int  # first sample of clear overlapping user speech
    earliest_stop: int  # onset + jittered evidence window
    run_start: int
    run_end: int  # end of the interrupted run in THIS world
    evidence_ms: int
    world: str  # "truncated" | "untruncated"


@dataclass(frozen=True)
class Cell:
    vp_index: int
    slot_index: int
    scenario: str
    split: str


@dataclass
class SampleScript:
    sample_id: str
    scenario: str
    split: str
    utterances: list[UtteranceSpan] = field(default_factory=list)
    noises: list[NoiseSpan] = field(default_factory=list)
    runs: list[RunSpan] = field(default_factory=list)
    interruptions: list[InterruptionEvent] = field(default_factory=list)
    snr_db: float = 0.0
    background_db: float = 0.0
    noise_db: float = 0.0
    evidence_ms: int = -1
    start_jitter_ms: int = 0
    rates: dict[str, int] = field(default_factory=dict)
    voices: dict[str, str] = field(default_factory=dict)
    world: str | None = None
    agent_flag_jitter_ms: int = 0
    codec: dict | None = None
    shared_prefix_ms: int | None = None
    pair_id: str | None = None


# ---------------------------------------------------------------------------
# Split machinery (contract "Splits").
# ---------------------------------------------------------------------------


def bank_pools(scale: str) -> dict[str, tuple[str, ...]]:
    """Template pools per semantic class at the given scale."""
    if scale == "short":
        return {
            "agent": tb.AGENT_SHORT,
            "user_interrupt": tb.USER_INTERRUPT_SHORT,
            "user_turn": tb.USER_TURN_SHORT,
            "backchannel": tb.BACKCHANNELS,
            "hesitation": tb.HESITATION_SHORT,
            "background": tb.BACKGROUND_SHORT,
        }
    return {
        "agent": tb.AGENT_TEMPLATES,
        "user_interrupt": tb.USER_INTERRUPT_TEMPLATES,
        "user_turn": tb.USER_TURN_TEMPLATES,
        "backchannel": tb.BACKCHANNELS,
        "hesitation": tb.HESITATION_TEMPLATES,
        "background": tb.BACKGROUND_TEMPLATES,
    }


def _slot_class(scenario: str) -> str:
    """The semantic class whose template slot indexes this scenario's cell."""
    return {
        "interruption": "user_interrupt",
        "backchannel": "backchannel",
        "normal_turn": "user_turn",
        "noise": "agent",
        "background_speaker": "agent",
        "hesitation": "hesitation",
        "hesitation_commit": "hesitation",
    }[scenario]


def _eval_only_subset(n: int, salt_key: str) -> frozenset[int]:
    """Deterministic hash-chosen ~30% subset of ``range(n)`` (>=1 when n >= 3)."""
    if n < 3:
        return frozenset()  # tiny pools stay usable in every split
    count = max(1, round(n * EVAL_ONLY_FRACTION))
    order = sorted(range(n), key=lambda i: stable_hash(f"{salt_key}|{i}"))
    return frozenset(order[:count])


@dataclass(frozen=True)
class SplitPools:
    salt: str
    scale: str
    eval_pairs: frozenset[int]
    eval_slots: dict[str, frozenset[int]]

    def summary(self) -> dict:
        return {
            "method": "cell-hash",
            "salt": self.salt,
            "cell": "(voice_pair, template_slot)",
            "eval_only_voice_pairs": sorted(self.eval_pairs),
            "eval_only_template_slots": {k: sorted(v) for k, v in sorted(self.eval_slots.items())},
            "eval_only_voice_pair_names": [
                f"{VOICE_PAIRS[i][0]}+{VOICE_PAIRS[i][1]}" for i in sorted(self.eval_pairs)
            ],
            "seed_ranges": {
                "train": list(TRAIN_SEED_RANGE),
                "eval": [TRAIN_SEED_RANGE[1], TRAIN_SEED_RANGE[1] * 2],
            },
            "augmentation_rng": "keyed on sample id (sha1) — never crosses the split",
            "codec_family": "eval-only degradation, applied to a drawn subset of eval samples",
            "note": (
                "cells containing any held-out voice pair or template slot are hash-assigned "
                "to val/test (50/50); other cells go to train. Realized split masses are "
                "reported in split_counts. Contract note: the ~30% pair holdout and ~30% "
                "per-class slot holdout force >=51% of cells into val+test, so realized "
                "masses are ~33/33/33 rather than 70/15/15 - the leakage-prevention "
                "holdouts take precedence and the masses are reported, not assumed."
            ),
        }


def build_pools(salt: str, scale: str) -> SplitPools:
    pools = bank_pools(scale)
    return SplitPools(
        salt=salt,
        scale=scale,
        eval_pairs=_eval_only_subset(len(VOICE_PAIRS), f"{salt}|vp"),
        eval_slots={
            cls: _eval_only_subset(len(pool), f"{salt}|slot|{cls}")
            for cls, pool in pools.items()
        },
    )


def split_for_cell(scenario: str, vp_index: int, slot_index: int, pools: SplitPools) -> str:
    if vp_index in pools.eval_pairs or slot_index in pools.eval_slots[_slot_class(scenario)]:
        h = stable_hash(f"{pools.salt}|eval|{scenario}|{vp_index}|{slot_index}")
        return "val" if h % 2 == 0 else "test"
    h = stable_hash(f"{pools.salt}|train|{scenario}|{vp_index}|{slot_index}")
    r = h % 100
    if r < 70:
        return "train"
    return "val" if r < 85 else "test"


def draw_cell(scenario: str, n_slots: int, rng_cell: np.random.Generator, pools: SplitPools) -> Cell:
    vp_index = int(rng_cell.integers(0, len(VOICE_PAIRS)))
    slot_index = int(rng_cell.integers(0, n_slots))
    return Cell(vp_index, slot_index, scenario, split_for_cell(scenario, vp_index, slot_index, pools))


def sample_seed(preset_seed: int, index: int, split: str) -> int:
    """Structural seed in the contract's split range: train [0, 1e6),
    eval [1e6, 2e6)."""
    base = (preset_seed * 1_000_003 + index * 7_919) % TRAIN_SEED_RANGE[1]
    return base if split == "train" else base + TRAIN_SEED_RANGE[1]


def voices_for(cell: Cell) -> dict[str, str]:
    user, agent = VOICE_PAIRS[cell.vp_index]
    return {
        ROLE_AGENT: agent,
        ROLE_USER: user,
        ROLE_BACKGROUND: BACKGROUND_VOICES[cell.vp_index % len(BACKGROUND_VOICES)],
    }


# ---------------------------------------------------------------------------
# Placement helpers.
# ---------------------------------------------------------------------------


def _u(rng: np.random.Generator, lo: int, hi: int) -> int:
    """Uniform integer ms draw, inclusive of both ends."""
    return int(rng.integers(lo, hi + 1))


def _ms_align(samples: int) -> int:
    return round(samples / MS_TO_SAMPLES) * MS_TO_SAMPLES


def _place_run(
    start: int,
    rendered: list[tuple[str, np.ndarray]],
    gap_ms_range: tuple[int, int],
    rng: np.random.Generator,
    run_index: int,
) -> tuple[list[UtteranceSpan], int, list[tuple[int, int]]]:
    """Place one agent run: consecutive utterances with intra-run gaps.

    Returns (utterance spans, run end, gap spans). Gaps belong to the run:
    ``agent_speaking`` stays true through them.
    """
    spans: list[UtteranceSpan] = []
    gaps: list[tuple[int, int]] = []
    cursor = start
    for i, (text, pcm) in enumerate(rendered):
        if i:
            gap = _u(rng, *gap_ms_range) * MS_TO_SAMPLES
            gaps.append((cursor, cursor + gap))
            cursor += gap
        spans.append(UtteranceSpan(ROLE_AGENT, text, cursor, cursor + pcm.size, run_index=run_index))
        cursor += pcm.size
    return spans, cursor, gaps


def _jitter(rng: np.random.Generator) -> int:
    return _u(rng, *START_JITTER_MS_RANGE)


def _params_and_rates(rng_levels: np.random.Generator, rng: np.random.Generator) -> dict:
    return {
        "snr_db": float(rng_levels.uniform(*SNR_DB_RANGE)),
        "background_db": float(rng_levels.uniform(*BACKGROUND_DB_RANGE)),
        "noise_db": float(rng_levels.uniform(*NOISE_DB_RANGE)),
        "rates": {
            role: int(RATE_CHOICES[int(rng.integers(0, len(RATE_CHOICES)))])
            for role in (ROLE_AGENT, ROLE_USER, ROLE_BACKGROUND)
        },
    }


def _apply_world(
    script: SampleScript,
    event: InterruptionEvent,
    run_index: int,
    rng: np.random.Generator,
    forced_world: str | None = None,
) -> str:
    """Draw the world variant and clamp the run audio for truncated worlds."""
    world = forced_world if forced_world is not None else (
        "truncated" if rng.random() < TRUNCATED_WORLD_RATE else "untruncated"
    )
    run_end_full = max(u.end for u in script.utterances if u.run_index == run_index)
    if world == "truncated":
        grace = _u(rng, *TRUNC_GRACE_MS_RANGE) * MS_TO_SAMPLES
        cut = event.earliest_stop + grace
        if cut < run_end_full - 50 * MS_TO_SAMPLES:
            kept: list[UtteranceSpan] = []
            for u in script.utterances:
                if u.run_index == run_index:
                    if u.start >= cut:
                        continue  # spans at/after the cut drop out entirely
                    if u.end > cut:
                        u.end = cut  # clamp the span crossing the cut
                kept.append(u)
            script.utterances = kept
            script.runs = [replace(r, end=cut) if r.run_index == run_index else r for r in script.runs]
            event.run_end = cut
        else:
            world = "untruncated"  # run too short for a meaningful truncation
    event.world = world
    script.world = world
    return world


def _flag_jitter_for(sample_id: str) -> int:
    """Signed agent_speaking INPUT-flag jitter (ms) for ~30% of samples.

    Keyed on sample id via the augmentation stream; labels stay stem-exact,
    downstream derives the deployed flag by shifting label boundaries.
    """
    rng = np.random.default_rng([augmentation_key(sample_id), 7])
    if rng.random() >= FLAG_JITTER_RATE:
        return 0
    magnitude = int(rng.integers(FLAG_JITTER_MIN_MS, abs(FLAG_JITTER_MS_RANGE[1]) + 1))
    return magnitude if rng.random() < 0.5 else -magnitude


# ---------------------------------------------------------------------------
# Per-scenario planners.
# ---------------------------------------------------------------------------


def _plan_interruption(
    script: SampleScript, scale: str, cell: Cell, rng: np.random.Generator,
    render: RenderFn, voices: dict[str, str], forced_world: str | None = None,
) -> None:
    short = scale == "short"
    pools = bank_pools(scale)
    voice_a, voice_u = voices[ROLE_AGENT], voices[ROLE_USER]
    rate_a, rate_u = script.rates[ROLE_AGENT], script.rates[ROLE_USER]

    lead = _u(rng, 200, 400) if short else _u(rng, 300, 900)
    # The run must be long enough to host onset (up to 60% of the run, plus
    # start jitter), the evidence window, and truncation grace with margin;
    # size it from constants (slots can be short at any render speed).
    utter: list[tuple[str, np.ndarray]] = []
    headroom_ms = (
        EVIDENCE_MS_RANGE[1] + START_JITTER_MS_RANGE[1] + TRUNC_GRACE_MS_RANGE[1] + 200
    )
    min_run_samples = int(2.5 * headroom_ms * MS_TO_SAMPLES)
    while sum(p.size for _, p in utter) < min_run_samples and len(utter) < 6:
        t = tb.fill(tb.pick(rng, pools["agent"]), rng)
        utter.append((t, render(t, voice_a, rate_a)))
    run_start = lead * MS_TO_SAMPLES
    run_spans, run_end, gaps = _place_run(
        run_start, utter,
        (0, 0) if short else (250, 500), rng, run_index=0,
    )
    script.utterances.extend(run_spans)
    script.runs.append(RunSpan(run_start, run_end, 0))

    # Evidence window; the stop must land before the run ends.
    evidence_ms = _u(rng, *EVIDENCE_MS_RANGE)
    script.evidence_ms = evidence_ms
    latest_onset = run_end - (evidence_ms + 300) * MS_TO_SAMPLES

    use_gap = rng.random() < GAP_ONSET_RATE
    usable = [g for g in gaps if g[0] + evidence_ms * MS_TO_SAMPLES <= run_end and g[0] <= latest_onset]
    if use_gap and usable:
        g0, g1 = usable[int(rng.integers(0, len(usable)))]  # inter-phrase gap onset
        onset = _ms_align(g0 + int(float(rng.uniform(0.1, 0.6)) * (g1 - g0)))
    else:
        frac = float(rng.uniform(0.40, 0.60)) if short else float(rng.uniform(0.35, 0.70))
        onset = _ms_align(run_start + int(frac * (run_end - run_start)))
    onset = min(onset, _ms_align(latest_onset))
    onset = max(onset, run_start + 100 * MS_TO_SAMPLES)

    jitter = _jitter(rng)
    script.start_jitter_ms = jitter
    user_start = onset + jitter * MS_TO_SAMPLES
    user_pool = pools["user_interrupt"]
    user_text = tb.fill(user_pool[cell.slot_index % len(user_pool)], rng)
    user_pcm = render(user_text, voice_u, rate_u)
    script.utterances.append(UtteranceSpan(ROLE_USER, user_text, user_start, user_start + user_pcm.size))
    event = InterruptionEvent(
        onset=user_start,
        earliest_stop=user_start + evidence_ms * MS_TO_SAMPLES,
        run_start=run_start,
        run_end=run_end,
        evidence_ms=evidence_ms,
        world="untruncated",
    )
    script.interruptions.append(event)
    _apply_world(script, event, 0, rng, forced_world)

    run_end_now = max(u.end for u in run_spans)
    if not short:
        if rng.random() < 0.6:  # agent resumes after the interruption
            resume = _u(rng, 400, 900) * MS_TO_SAMPLES
            resume_texts = [tb.fill(tb.pick(rng, pools["agent"]), rng)]
            resume_spans, resume_end, _ = _place_run(
                run_end_now + resume, [(t, render(t, voice_a, rate_a)) for t in resume_texts],
                (250, 500), rng, run_index=1,
            )
            script.utterances.extend(resume_spans)
            script.runs.append(RunSpan(resume_spans[0].start, resume_end, 1))
            cursor = resume_end
        else:
            cursor = run_end_now
        if rng.random() < 0.5:  # user acknowledges once the agent is done
            gap = _u(rng, 400, 900) * MS_TO_SAMPLES
            ack = tb.fill(tb.pick(rng, pools["user_turn"]), rng)
            ack_pcm = render(ack, voice_u, rate_u)
            script.utterances.append(UtteranceSpan(ROLE_USER, ack, cursor + gap, cursor + gap + ack_pcm.size))


def _plan_backchannel(
    script: SampleScript, scale: str, cell: Cell, rng: np.random.Generator,
    render: RenderFn, voices: dict[str, str], forced_world: str | None = None,
) -> None:
    short = scale == "short"
    pools = bank_pools(scale)
    voice_a, voice_u = voices[ROLE_AGENT], voices[ROLE_USER]
    rate_a, rate_u = script.rates[ROLE_AGENT], script.rates[ROLE_USER]

    lead = _u(rng, 200, 400) if short else _u(rng, 300, 900)
    if short:
        run_texts = [tb.fill(tb.pick(rng, pools["agent"]), rng)]
    else:
        n_sent = int(rng.integers(2, 4))
        run_texts = [tb.fill(tb.pick(rng, pools["agent"]), rng) for _ in range(n_sent)]
    run_spans, run_end, gaps = _place_run(
        lead * MS_TO_SAMPLES, [(t, render(t, voice_a, rate_a)) for t in run_texts],
        (300, 600), rng, run_index=0,
    )
    script.utterances.extend(run_spans)
    script.runs.append(RunSpan(run_spans[0].start, run_end, 0))
    run_start = run_spans[0].start

    n_bc = 1 if short else int(rng.integers(1, 3))
    for i in range(n_bc):
        bc_pool = pools["backchannel"]
        text = bc_pool[cell.slot_index % len(bc_pool)] if i == 0 else tb.pick(rng, bc_pool)
        pcm = render(text, voice_u, rate_u)
        if len(gaps) > 0 and rng.random() < GAP_ONSET_RATE:
            g0, g1 = gaps[int(rng.integers(0, len(gaps)))]  # inter-phrase gap onset
            start = _ms_align(g0 + int(float(rng.uniform(0.1, 0.6)) * (g1 - g0)))
        else:
            earliest = run_start + 200 * MS_TO_SAMPLES
            latest = run_end - pcm.size - 100 * MS_TO_SAMPLES
            start = int(rng.uniform(earliest, latest)) if latest > earliest else max(
                run_start + 50 * MS_TO_SAMPLES, run_end - pcm.size
            )
        start += _jitter(rng) * MS_TO_SAMPLES
        start = min(max(start, run_start), run_end - pcm.size)
        script.utterances.append(UtteranceSpan(ROLE_USER, text, start, start + pcm.size, is_backchannel=True))

    if not short and rng.random() < 0.7:
        gap = _u(rng, 500, 900) * MS_TO_SAMPLES
        text = tb.fill(tb.pick(rng, pools["user_turn"]), rng)
        pcm = render(text, voice_u, rate_u)
        script.utterances.append(UtteranceSpan(ROLE_USER, text, run_end + gap, run_end + gap + pcm.size))


def _plan_normal_turn(
    script: SampleScript, scale: str, cell: Cell, rng: np.random.Generator,
    render: RenderFn, voices: dict[str, str], forced_world: str | None = None,
) -> None:
    short = scale == "short"
    pools = bank_pools(scale)
    cursor = _u(rng, 200, 400) * MS_TO_SAMPLES if short else _u(rng, 300, 900) * MS_TO_SAMPLES
    user_pool = pools["user_turn"]

    n_exchanges = 1 if short else int(rng.integers(2, 3))
    run_index = 0
    for i in range(n_exchanges):
        for role, pool_name in ((ROLE_AGENT, "agent"), (ROLE_USER, "user_turn")):
            if role == ROLE_USER and i == 0:
                text = tb.fill(user_pool[cell.slot_index % len(user_pool)], rng)
            else:
                text = tb.fill(tb.pick(rng, pools[pool_name]), rng)
            pcm = render(text, voices[role], script.rates[role])
            if role == ROLE_AGENT:
                script.utterances.append(UtteranceSpan(role, text, cursor, cursor + pcm.size, run_index=run_index))
                script.runs.append(RunSpan(cursor, cursor + pcm.size, run_index))
                run_index += 1
            else:
                script.utterances.append(UtteranceSpan(role, text, cursor, cursor + pcm.size))
            cursor += pcm.size
            gap = _u(rng, 200, 400) if short else _u(rng, 500, 1000)
            cursor += gap * MS_TO_SAMPLES
            if role == ROLE_USER and i == n_exchanges - 1 and (short or rng.random() < 0.6):
                cursor -= gap * MS_TO_SAMPLES  # no dangling gap after the final turn


def _plan_noise(
    script: SampleScript, scale: str, cell: Cell, rng: np.random.Generator,
    render: RenderFn, voices: dict[str, str], forced_world: str | None = None,
) -> None:
    """Non-speech vocal events only: speech_present stays 0 throughout."""
    short = scale == "short"
    pools = bank_pools(scale)
    voice_a, rate_a = voices[ROLE_AGENT], script.rates[ROLE_AGENT]

    lead = _u(rng, 200, 400) if short else _u(rng, 300, 900)
    if short:
        run_texts = [tb.fill(pools["agent"][cell.slot_index % len(pools["agent"])], rng)]
    else:
        n_sent = int(rng.integers(2, 3))
        run_texts = [
            tb.fill(pools["agent"][cell.slot_index % len(pools["agent"])], rng) if k == 0
            else tb.fill(tb.pick(rng, pools["agent"]), rng)
            for k in range(n_sent)
        ]
    run_spans, run_end, _ = _place_run(
        lead * MS_TO_SAMPLES, [(t, render(t, voice_a, rate_a)) for t in run_texts], (300, 600), rng, run_index=0,
    )
    script.utterances.extend(run_spans)
    script.runs.append(RunSpan(run_spans[0].start, run_end, 0))
    run_start = run_spans[0].start

    n_events = 1 if short else int(rng.integers(1, 4))
    for _ in range(n_events):
        kind = tb.pick(rng, ("cough", "breath", "burst"))
        dur_ms = {"cough": _u(rng, 180, 420), "breath": _u(rng, 350, 900), "burst": _u(rng, 30, 120)}[kind]
        if rng.random() < 0.5:
            # During agent playback — the harder, masked case.
            latest = max(run_start, run_end - dur_ms * MS_TO_SAMPLES)
            start = int(rng.uniform(run_start, latest)) if latest > run_start else run_start
        else:
            # In agent silence after the run — the unmasked case, where a
            # spurious STOP on a cough is most dangerous.
            start = run_end + _u(rng, 100, 1200) * MS_TO_SAMPLES
        start += _jitter(rng) * MS_TO_SAMPLES
        script.noises.append(NoiseSpan(kind, start, start + dur_ms * MS_TO_SAMPLES))

    if not short and rng.random() < 0.4:  # second agent run: cross-run hazards
        gap = _u(rng, 500, 900) * MS_TO_SAMPLES
        resume_texts = [tb.fill(tb.pick(rng, pools["agent"]), rng)]
        resume_spans, resume_end, _ = _place_run(
            run_end + gap, [(t, render(t, voice_a, rate_a)) for t in resume_texts], (250, 500), rng, run_index=1,
        )
        script.utterances.extend(resume_spans)
        script.runs.append(RunSpan(resume_spans[0].start, resume_end, 1))


def _plan_background_speaker(
    script: SampleScript, scale: str, cell: Cell, rng: np.random.Generator,
    render: RenderFn, voices: dict[str, str], forced_world: str | None = None,
) -> None:
    short = scale == "short"
    pools = bank_pools(scale)
    voice_a, rate_a = voices[ROLE_AGENT], script.rates[ROLE_AGENT]

    lead = _u(rng, 200, 400) if short else _u(rng, 300, 900)
    if short:
        run_texts = [tb.fill(pools["agent"][cell.slot_index % len(pools["agent"])], rng)]
    else:
        n_sent = int(rng.integers(2, 3))
        run_texts = [
            tb.fill(pools["agent"][cell.slot_index % len(pools["agent"])], rng) if k == 0
            else tb.fill(tb.pick(rng, pools["agent"]), rng)
            for k in range(n_sent)
        ]
    run_spans, run_end, _ = _place_run(
        lead * MS_TO_SAMPLES, [(t, render(t, voice_a, rate_a)) for t in run_texts], (300, 600), rng, run_index=0,
    )
    script.utterances.extend(run_spans)
    script.runs.append(RunSpan(run_spans[0].start, run_end, 0))
    run_start = run_spans[0].start

    n_bg = 1 if short else int(rng.integers(1, 3))
    for _ in range(n_bg):
        text = tb.fill(tb.pick(rng, pools["background"]), rng)
        pcm = render(text, voices[ROLE_BACKGROUND], script.rates[ROLE_BACKGROUND])
        earliest = max(0, run_start - 200 * MS_TO_SAMPLES)
        latest = max(earliest, run_end - min(pcm.size // 2, 500 * MS_TO_SAMPLES))
        start = int(rng.uniform(earliest, latest)) + _jitter(rng) * MS_TO_SAMPLES
        script.utterances.append(UtteranceSpan(ROLE_BACKGROUND, text, start, start + pcm.size))

    if not short and rng.random() < 0.3:
        gap = _u(rng, 500, 900) * MS_TO_SAMPLES
        text = tb.fill(tb.pick(rng, pools["user_turn"]), rng)
        pcm = render(text, voices[ROLE_USER], script.rates[ROLE_USER])
        script.utterances.append(UtteranceSpan(ROLE_USER, text, run_end + gap, run_end + gap + pcm.size))
    if not short and rng.random() < 0.3:  # second agent run
        gap = _u(rng, 500, 900) * MS_TO_SAMPLES
        resume_texts = [tb.fill(tb.pick(rng, pools["agent"]), rng)]
        resume_spans, resume_end, _ = _place_run(
            run_end + gap, [(t, render(t, voice_a, rate_a)) for t in resume_texts], (250, 500), rng, run_index=1,
        )
        script.utterances.extend(resume_spans)
        script.runs.append(RunSpan(resume_spans[0].start, resume_end, 1))


def _plan_hesitation_pure(
    script: SampleScript, scale: str, cell: Cell, rng: np.random.Generator,
    render: RenderFn, voices: dict[str, str], forced_world: str | None = None,
) -> None:
    short = scale == "short"
    pools = bank_pools(scale)
    voice_a, voice_u = voices[ROLE_AGENT], voices[ROLE_USER]
    rate_a, rate_u = script.rates[ROLE_AGENT], script.rates[ROLE_USER]

    lead = _u(rng, 200, 400) if short else _u(rng, 300, 900)
    if short:
        run_texts = [tb.fill(tb.pick(rng, pools["agent"]), rng)]
    else:
        n_sent = int(rng.integers(2, 3))
        run_texts = [tb.fill(tb.pick(rng, pools["agent"]), rng) for _ in range(n_sent)]
    run_spans, run_end, _ = _place_run(
        lead * MS_TO_SAMPLES, [(t, render(t, voice_a, rate_a)) for t in run_texts], (300, 600), rng, run_index=0,
    )
    script.utterances.extend(run_spans)
    script.runs.append(RunSpan(run_spans[0].start, run_end, 0))
    run_start = run_spans[0].start

    n_hes = 1 if short else int(rng.integers(1, 3))
    for i in range(n_hes):
        pool = pools["hesitation"]
        text = tb.fill(pool[cell.slot_index % len(pool)] if i == 0 else tb.pick(rng, pool), rng)
        pcm = render(text, voice_u, rate_u)
        frac = float(rng.uniform(0.30, 0.60))
        start = run_start + int(frac * (run_end - run_start)) + _jitter(rng) * MS_TO_SAMPLES
        start = min(max(start, run_start), max(run_start, run_end - pcm.size // 2))
        script.utterances.append(UtteranceSpan(ROLE_USER, text, start, start + pcm.size))

    if not short and rng.random() < 0.5:
        gap = _u(rng, 400, 900) * MS_TO_SAMPLES
        resume_texts = [tb.fill(tb.pick(rng, pools["agent"]), rng)]
        resume_spans, resume_end, _ = _place_run(
            run_end + gap, [(t, render(t, voice_a, rate_a)) for t in resume_texts], (250, 500), rng, run_index=1,
        )
        script.utterances.extend(resume_spans)
        script.runs.append(RunSpan(resume_spans[0].start, resume_end, 1))


_PLANNERS = {
    "interruption": _plan_interruption,
    "backchannel": _plan_backchannel,
    "normal_turn": _plan_normal_turn,
    "noise": _plan_noise,
    "background_speaker": _plan_background_speaker,
    "hesitation": _plan_hesitation_pure,
}


def plan_sample(
    scenario: str,
    sample_id: str,
    scale: str,
    cell: Cell,
    rng_struct: np.random.Generator,
    rng_levels: np.random.Generator,
    render: RenderFn,
    forced_world: str | None = None,
) -> SampleScript:
    """Build one sample's script. ``render`` synthesizes on demand."""
    voices = voices_for(cell)
    base = _params_and_rates(rng_levels, rng_struct)
    script = SampleScript(
        sample_id=sample_id,
        scenario=scenario,
        split=cell.split,
        snr_db=base["snr_db"],
        background_db=base["background_db"],
        noise_db=base["noise_db"],
        rates=base["rates"],
        voices=voices,
    )
    _PLANNERS[scenario](script, scale, cell, rng_struct, render, voices, forced_world)
    script.agent_flag_jitter_ms = _flag_jitter_for(sample_id)
    return script


def plan_hesitation_pair(
    pair_id: str,
    id_a: str,
    id_b: str,
    scale: str,
    cell: Cell,
    rng_struct: np.random.Generator,
    rng_levels: np.random.Generator,
    render: RenderFn,
    forced_world: str | None = None,
) -> tuple[SampleScript, SampleScript]:
    """Generate a censored hesitation pair: byte-identical audio up to the
    censoring point (the end of the hesitation span), then sample B commits
    into real overlap and is labeled as an interruption from the commit point.

    One cell, one seed, one set of augmentation parameters — the pair shares
    everything (voices, rates, levels, placement) so the prefix is truly
    identical, including mixing gains and the codec decision (keyed on the
    pair id, not the sample id).
    """
    short = scale == "short"
    pools = bank_pools(scale)
    voices = voices_for(cell)
    voice_a, voice_u = voices[ROLE_AGENT], voices[ROLE_USER]
    base = _params_and_rates(rng_levels, rng_struct)
    rate_a, rate_u = base["rates"][ROLE_AGENT], base["rates"][ROLE_USER]

    lead = _u(rng_struct, 200, 400) if short else _u(rng_struct, 300, 900)
    pool = pools["hesitation"]
    hes_text = tb.fill(pool[cell.slot_index % len(pool)], rng_struct)
    hes_pcm = render(hes_text, voice_u, rate_u)
    # The run must host the hesitation (starting at up to 60% into the run),
    # the commit gap, the evidence window, and truncation grace.
    min_run_samples = int(2.5 * (hes_pcm.size + 600 * MS_TO_SAMPLES))
    utter: list[tuple[str, np.ndarray]] = []
    while sum(p.size for _, p in utter) < min_run_samples and len(utter) < 8:
        t = tb.fill(tb.pick(rng_struct, pools["agent"]), rng_struct)
        utter.append((t, render(t, voice_a, rate_a)))
    run_start = lead * MS_TO_SAMPLES
    run_spans, run_end, _ = _place_run(
        run_start, utter, (300, 600), rng_struct, run_index=0,
    )
    run_span = RunSpan(run_start, run_end, 0)

    hes_start = run_start + int(float(rng_struct.uniform(0.30, 0.60)) * (run_end - run_start))
    hes_jitter = _jitter(rng_struct)
    hes_start = _ms_align(hes_start + hes_jitter * MS_TO_SAMPLES)
    hes_start = min(max(hes_start, run_start), max(run_start, run_end - hes_pcm.size // 2))
    hes_end = hes_start + hes_pcm.size
    shared_prefix_ms = (hes_end - hes_start) // MS_TO_SAMPLES

    script_a = SampleScript(
        sample_id=id_a, scenario="hesitation", split=cell.split,
        utterances=[replace(u) for u in run_spans] + [UtteranceSpan(ROLE_USER, hes_text, hes_start, hes_end)],
        runs=[replace(run_span)],
        noises=[],
        interruptions=[],
        snr_db=base["snr_db"], background_db=base["background_db"], noise_db=base["noise_db"],
        rates=base["rates"], voices=voices, pair_id=pair_id, shared_prefix_ms=shared_prefix_ms,
        start_jitter_ms=hes_jitter,
    )
    script_a.agent_flag_jitter_ms = _flag_jitter_for(id_a)

    # --- commit partner: identical prefix, then a real interruption ---
    commit_gap = _u(rng_struct, 0, 100) * MS_TO_SAMPLES
    commit_onset = _ms_align(hes_end + commit_gap)
    commit_text = tb.fill(tb.pick(rng_struct, pools["user_interrupt"]), rng_struct)
    commit_pcm = render(commit_text, voice_u, rate_u)
    evidence_ms = _u(rng_struct, *EVIDENCE_MS_RANGE)

    script_b = SampleScript(
        sample_id=id_b, scenario="hesitation_commit", split=cell.split,
        utterances=[replace(u) for u in run_spans] + [
            UtteranceSpan(ROLE_USER, hes_text, hes_start, hes_end),
            UtteranceSpan(ROLE_USER, commit_text, commit_onset, commit_onset + commit_pcm.size),
        ],
        runs=[replace(run_span)],
        noises=[],
        interruptions=[],
        snr_db=base["snr_db"], background_db=base["background_db"], noise_db=base["noise_db"],
        rates=base["rates"], voices=voices, pair_id=pair_id, shared_prefix_ms=shared_prefix_ms,
        start_jitter_ms=hes_jitter, evidence_ms=evidence_ms,
    )
    event = InterruptionEvent(
        onset=commit_onset,
        earliest_stop=commit_onset + evidence_ms * MS_TO_SAMPLES,
        run_start=run_start,
        run_end=run_end,
        evidence_ms=evidence_ms,
        world="untruncated",
    )
    script_b.interruptions.append(event)
    _apply_world(script_b, event, 0, rng_struct, forced_world)
    script_b.agent_flag_jitter_ms = _flag_jitter_for(id_b)
    return script_a, script_b
