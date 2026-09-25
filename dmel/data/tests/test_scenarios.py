"""Scenario planner tests against a deterministic fake renderer (no espeak)."""

import numpy as np
import pytest

from dmel.data.scenarios import (
    ROLE_AGENT,
    ROLE_USER,
    Cell,
    build_pools,
    draw_cell,
    plan_hesitation_pair,
    plan_sample,
    sample_seed,
    voices_for,
)


def fake_render(text: str, voice: str, rate: int) -> np.ndarray:
    # ~50 ms per character (800 samples) — speech-like durations so planned
    # runs are long enough to host evidence windows.
    n = int(len(text) * 800 * (170 / rate))
    return np.full(max(n, 160), 1000, dtype=np.int16)


SALT = "test|1"
POOLS = build_pools(salt=SALT, scale="short")


def make_cell(scenario: str = "interruption", seed: int = 0) -> Cell:
    rng = np.random.default_rng([seed, 3])
    return draw_cell(scenario, 2, rng, POOLS)


def make_script(scenario: str, seed: int = 0, forced_world: str | None = None):
    cell = make_cell(scenario, seed)
    rng_struct = np.random.default_rng([sample_seed(1, seed, cell.split), 0])
    rng_levels = np.random.default_rng([sample_seed(1, seed, cell.split), 1])
    return plan_sample(
        scenario, f"t-{scenario}-{seed}", "short", cell, rng_struct, rng_levels,
        fake_render, forced_world,
    )


class TestInterruption:
    def test_untruncated_run_covers_stop_window(self):
        script = make_script("interruption", forced_world="untruncated")
        event = script.interruptions[0]
        assert event.world == "untruncated"
        # The flagged run extends past the stop point (full agent run plays).
        run = script.runs[0]
        assert run.end > event.earliest_stop

    def test_truncated_world_clamps_run(self):
        script = make_script("interruption", seed=1, forced_world="truncated")
        event = script.interruptions[0]
        assert event.world == "truncated"
        # Flagged territory ends shortly after the stop point (≤400 ms grace).
        assert event.run_end <= event.earliest_stop + 400 * 16 + 1
        # No agent utterance extends past the cut.
        agent_ends = [u.end for u in script.utterances if u.role == ROLE_AGENT]
        assert max(agent_ends) <= event.run_end

    def test_evidence_window_in_contract_range(self):
        for seed in range(5):
            script = make_script("interruption", seed=seed)
            event = script.interruptions[0]
            assert 150 <= event.evidence_ms <= 300
            assert event.earliest_stop == event.onset + event.evidence_ms * 16

    def test_user_overlaps_agent_run(self):
        script = make_script("interruption", seed=2)
        event = script.interruptions[0]
        user = [u for u in script.utterances if u.role == ROLE_USER]
        assert user, "user utterance planned"
        assert any(u.start < event.run_end and u.end > event.onset for u in user)


class TestRuns:
    def test_backchannel_is_single_run(self):
        script = make_script("backchannel", seed=3)
        assert len(script.runs) == 1

    def test_onset_inside_flagged_run(self):
        # The interruption onset must fall inside the flagged run (mid-phrase
        # or in a gap — both belong to the run's flag territory).
        script = make_script("interruption", seed=4)
        event = script.interruptions[0]
        run = script.runs[0]
        assert run.start <= event.onset < run.end


class TestHesitationPair:
    def test_pair_prefix_planned_identical(self):
        cell = make_cell("hesitation", seed=5)
        rng_a = np.random.default_rng([777, 0])
        rng_b = np.random.default_rng([777, 1])
        a, b = plan_hesitation_pair("p", "pa", "pb", "short", cell, rng_a, rng_b, fake_render)
        assert a.shared_prefix_ms is not None and a.shared_prefix_ms > 0
        hes_a = [u for u in a.utterances if u.role == ROLE_USER][-1]
        prefix_end = hes_a.end  # the censoring point, in samples
        # Everything fully inside the shared prefix is identical (same texts,
        # same placement). The commit partner's truncated world may legitimately
        # drop agent spans that start after the stop decision, so only the
        # prefix is compared span-for-span.
        pre_a = [u for u in a.utterances if u.end <= prefix_end]
        pre_b = [u for u in b.utterances if u.end <= prefix_end]
        assert pre_a and len(pre_a) == len(pre_b)
        for span_a, span_b in zip(pre_a, pre_b):
            assert (span_a.role, span_a.text, span_a.start, span_a.end) == (
                span_b.role, span_b.text, span_b.start, span_b.end,
            )
        # The hesitation sample has NO interruption event; the partner does.
        assert a.interruptions == []
        assert len(b.interruptions) == 1
        event = b.interruptions[0]
        # Commit onset is at or after the censoring point (hesitation end).
        assert event.onset >= prefix_end - 16

    def test_pair_shares_levels_and_rates(self):
        cell = make_cell("hesitation", seed=6)
        a, b = plan_hesitation_pair(
            "p", "pa", "pb", "short", cell,
            np.random.default_rng([1, 0]), np.random.default_rng([2, 1]), fake_render,
        )
        assert a.rates == b.rates
        assert a.snr_db == b.snr_db
        assert a.voices == b.voices

    def test_commit_partner_truncation_applies(self):
        cell = make_cell("hesitation", seed=7)
        a, b = plan_hesitation_pair(
            "p", "pa", "pb", "short", cell,
            np.random.default_rng([3, 0]), np.random.default_rng([4, 1]), fake_render,
            forced_world="truncated",
        )
        assert b.world == "truncated"
        assert a.world is None  # pure hesitation has no world


class TestCells:
    def test_cell_draw_uses_pools(self):
        pools = build_pools(salt=SALT, scale="short")
        rng = np.random.default_rng([0, 3])
        cell = draw_cell("interruption", 2, rng, pools)
        assert cell.split in ("train", "val", "test")
        assert voices_for(cell)[ROLE_AGENT] != voices_for(cell)[ROLE_USER]

    def test_seed_ranges_follow_split(self):
        assert sample_seed(5, 0, "train") < 1_000_000
        assert sample_seed(5, 0, "val") >= 1_000_000
        assert sample_seed(5, 0, "test") >= 1_000_000

    @pytest.mark.parametrize("scenario", [
        "interruption", "backchannel", "normal_turn", "noise",
        "background_speaker", "hesitation",
    ])
    def test_every_scenario_plans(self, scenario):
        script = make_script(scenario, seed=8)
        assert script.scenario == scenario
        assert script.utterances  # something was planned
