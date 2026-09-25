"""Label invariant tests: annotate() against planned scripts (fake render)."""

import numpy as np
import pytest

from dmel.data.annotation import LABEL_COLUMNS, agent_flag_input_view, annotate, n_frames_for
from dmel.data.constants import FRAME_MS, FRAME_SAMPLES, MS_TO_SAMPLES
from dmel.data.scenarios import (
    ROLE_AGENT,
    ROLE_USER,
    UtteranceSpan,
    build_pools,
    draw_cell,
    plan_sample,
    sample_seed,
)


def fake_render(text: str, voice: str, rate: int) -> np.ndarray:
    # ~50 ms per character (800 samples) — speech-like durations so planned
    # runs are long enough to host evidence windows.
    n = int(len(text) * 800 * (170 / rate))
    return np.full(max(n, 160), 1000, dtype=np.int16)


SALT = "test|1"
POOLS = build_pools(salt=SALT, scale="short")


def make_script(scenario: str, seed: int = 0, forced_world: str | None = None):
    rng_cell = np.random.default_rng([seed, 3])
    cell = draw_cell(scenario, 2, rng_cell, POOLS)
    rng_struct = np.random.default_rng([sample_seed(1, seed, cell.split), 0])
    rng_levels = np.random.default_rng([sample_seed(1, seed, cell.split), 1])
    script = plan_sample(
        scenario, f"t-{scenario}-{seed}", "short", cell, rng_struct, rng_levels,
        fake_render, forced_world,
    )
    # Same length rule as production (generate.py annotates the mix length,
    # which includes noise-span ends): tail-placed noise stays inside labels.
    total = max(u.end for u in script.utterances) + FRAME_SAMPLES * 4
    if script.noises:
        total = max(total, max(n.end for n in script.noises) + FRAME_SAMPLES * 4)
    return script, annotate(script, total), total


def frames(labels, flag):
    return labels[flag].astype(bool)


class TestSchema:
    def test_columns_and_lengths(self):
        script, labels, total = make_script("interruption", seed=0)
        assert tuple(labels.keys()) == LABEL_COLUMNS
        n = n_frames_for(total)
        for col in LABEL_COLUMNS:  # scenario is a per-frame constant column
            assert len(labels[col]) == n, col
        assert set(labels["scenario"].tolist()) <= {script.scenario}

    def test_frame_alignment(self):
        script, labels, total = make_script("backchannel", seed=1)
        n = total // FRAME_SAMPLES
        assert labels["t_ms"][0] == 0
        assert labels["t_ms"][n - 1] == (n - 1) * FRAME_MS
        assert list(labels["frame_index"]) == list(range(n))


class TestAgentSpeaking:
    def test_run_level_flag_spans_gaps(self):
        # agent_speaking is the run-level playback flag: stays 1 through
        # intra-run inter-phrase gaps (multi-sentence runs). Frames are flagged
        # when the run intersects them, so the last flagged frame contains the
        # run end.
        script, labels, _ = make_script("interruption", seed=2)
        run = script.runs[0]
        first = run.start // FRAME_SAMPLES
        last = (run.end - 1) // FRAME_SAMPLES
        assert last > first
        assert labels["agent_speaking"][first : last + 1].all()
        assert not labels["agent_speaking"][:first].any()
        assert not labels["agent_speaking"][last + 1 :].any()

    def test_truncated_world_flag_drops_after_cut(self):
        script, labels, _ = make_script("interruption", seed=3, forced_world="truncated")
        event = script.interruptions[0]
        first_zero = ((event.run_end - 1) // FRAME_SAMPLES) + 1
        assert not labels["agent_speaking"][first_zero:].any()


class TestSpeechAttribution:
    def test_background_speaker_speech(self):
        # background_speaker scenario: speech_present=1 on background speech
        # while primary_user stays 0 (attribution lives in primary_user).
        script, labels, _ = make_script("background_speaker", seed=4)
        bg = [u for u in script.utterances if u.role == "background"]
        assert bg, "background utterance planned"
        f0 = bg[0].start // FRAME_SAMPLES
        f1 = bg[0].end // FRAME_SAMPLES
        assert labels["speech_present"][f0:f1].all()
        assert not labels["primary_user"][f0:f1].any()

    def test_noise_events_are_not_speech(self):
        # Noise events themselves are not speech — but a noise event landing
        # during agent playback shares the frame with agent speech, which the
        # mix genuinely contains. Only agent-silent noise frames must stay 0.
        # The planner places ~half of events in agent silence; seeds are fixed,
        # so the sweep below deterministically covers both placements.
        silent_frames_total = 0
        for seed in range(10):
            script, labels, _ = make_script("noise", seed=seed)
            for span in script.noises:
                f0 = span.start // FRAME_SAMPLES
                f1 = (span.end - 1) // FRAME_SAMPLES + 1
                silent = labels["agent_speaking"][f0:f1] == 0
                assert not labels["speech_present"][f0:f1][silent].any()
                silent_frames_total += int(silent.sum())
        assert silent_frames_total > 0  # at least one agent-silent noise frame exists


class TestActionDerivation:
    def test_stop_window_and_event_columns(self):
        script, labels, _ = make_script("interruption", seed=6)
        event = script.interruptions[0]
        # STOP frames: frame starts at/after the stop point (rounded up to the
        # frame boundary) while the frame intersects the run.
        stop_first = -(-event.earliest_stop // FRAME_SAMPLES)  # ceil to a frame
        stop_last_exclusive = ((event.run_end - 1) // FRAME_SAMPLES) + 1
        stop = labels["action"] == "STOP_TTS"  # string column — no bool cast
        assert stop[stop_first:stop_last_exclusive].all()
        assert not stop[:stop_first].any()
        assert not stop[stop_last_exclusive:].any()
        # Event columns populated from onset through run end, -1 elsewhere.
        onset_frame = event.onset // FRAME_SAMPLES
        for col, expected_ms in (
            ("earliest_reasonable_stop_ms", event.earliest_stop // MS_TO_SAMPLES),
            ("overlap_onset_ms", event.onset // MS_TO_SAMPLES),
        ):
            col_arr = labels[col]
            assert (col_arr[:onset_frame] == -1).all()
            assert (col_arr[onset_frame:stop_last_exclusive] == expected_ms).all()
            assert (col_arr[stop_last_exclusive:] == -1).all()

    def test_intent_labeling_range(self):
        script, labels, _ = make_script("interruption", seed=7)
        event = script.interruptions[0]
        onset_frame = event.onset // FRAME_SAMPLES
        stop_frame = event.earliest_stop // FRAME_SAMPLES
        intent = labels["interrupt_intent"].astype(bool)
        assert intent[onset_frame:stop_frame].all()
        # Intent ends at the stop point (causal-decision labeling).
        user_after = [u for u in script.utterances if u.role == ROLE_USER and u.end > event.earliest_stop]
        if user_after:  # user keeps talking — intent must NOT extend
            assert not intent[stop_frame:].all()

    def test_hesitation_keeps_action(self):
        script, labels, _ = make_script("hesitation", seed=8)
        assert script.interruptions == []
        assert (labels["action"] == "KEEP").all()
        assert not labels["interrupt_intent"].any()

    def test_keep_when_agent_silent(self):
        script, labels, _ = make_script("normal_turn", seed=9)
        silent = ~labels["agent_speaking"].astype(bool)
        assert (labels["action"][silent] == "KEEP").all()


class TestFlagJitterView:
    def test_jittered_input_view_differs_on_jittered_samples(self):
        script, labels, _ = make_script("interruption", seed=10)
        exact = labels["agent_speaking"]
        input_view = agent_flag_input_view(exact, script.agent_flag_jitter_ms)
        assert len(input_view) == len(exact)
        if script.agent_flag_jitter_ms == 0:
            assert (input_view == exact).all()
        else:
            # Boundary shift only — same count within tolerance, differs somewhere.
            assert not (input_view == exact).all()
            assert abs(int(input_view.sum()) - int(exact.sum())) <= 4  # ≤200 ms shift

    def test_zero_jitter_is_identity(self):
        script, labels, _ = make_script("normal_turn", seed=11)
        assert (agent_flag_input_view(labels["agent_speaking"], 0) == labels["agent_speaking"]).all()
