"""Hand-computed metric-math tests on synthetic decision logs.

Every expected number below is computed by hand from the contract definitions:
event active = [user onset, earliest_reasonable_stop_ms + 500 ms grace], the
first STOP per event is the scored one, decision latency anchors to the
overlap_onset_ms physical anchor (not the jittered evidence reference), and
never-stopped events are censored at infinity. Frame i covers t = [50i, 50i+50).
"""

from __future__ import annotations

import numpy as np
import pytest

from dmel.eval.metrics import (
    InterruptionEvent,
    MS_PER_STEP,
    _average_ranks,
    _bootstrap_ci,
    _censored_latency_stats,
    _fbeta,
    _runs_of,
    compute_metrics,
    first_stop_in_window,
)
from dmel.eval.sweep import select_operating_point
from dmel.eval.tests.helpers import interruption_sample, labels_df


def int_labels(erts_ms: int, intent_frame: int, overlap_ms: int, action_frame: int, n: int = 60):
    """Canonical single-event interruption sample with movable anchor/evidence point."""
    return labels_df(
        n,
        agent_speaking={0: 1},
        scenario={0: "normal_turn", 20: "interruption"},
        interrupt_intent={intent_frame: 1},
        overlap_onset_ms={intent_frame: overlap_ms},
        earliest_reasonable_stop_ms={action_frame: erts_ms},
        action={action_frame: "STOP_TTS"},
        speech_present={intent_frame: 1},
        primary_user={intent_frame: 1},
    )


def test_first_stop_in_window_and_alignment():
    stops = [1250, 1400, 1450]
    assert first_stop_in_window(stops, 1200, 1900) == 1250
    assert first_stop_in_window([1901], 1200, 1900) is None
    assert first_stop_in_window([], 0, 100) is None
    assert first_stop_in_window([100, 50], 0, 1000) == 50  # temporally first in-window stop is the scored one
    assert MS_PER_STEP == 50  # the contract frame hop this whole file assumes


def test_perfect_system():
    """Stop exactly at the evidence point: recall 1, latency 200, zero false stops."""
    m = compute_metrics({"int60": interruption_sample()}, {"int60": [1400]})
    it = m["interruptions"]
    assert it["total_events"] == 1
    assert it["handled"] == 1
    assert it["recall"] == 1.0
    assert it["never_stopped_rate"] == 0.0
    assert it["premature_during_overlap"] == 0
    assert it["premature_during_non_overlap"] == 0
    lat = it["decision_latency_ms"]
    assert lat["median"] == pytest.approx(200.0)  # 1400 - anchor 1200
    assert lat["p95"] == pytest.approx(200.0)
    assert lat["n_handled"] == 1 and lat["n_censored"] == 0
    assert lat["median_censored"] is False and lat["p95_censored"] is False
    sd = m["stop_decision"]
    assert sd["tp"] == 1 and sd["fp"] == 0 and sd["fn"] == 31
    assert sd["precision"] == 1.0
    assert sd["recall"] == pytest.approx(1 / 32)
    assert all(row["false_stop_frames"] == 0 for row in m["false_stops_by_scenario"].values())
    # One event -> degenerate (exact) bootstrap CI.
    assert m["uncertainty"]["recall_ci95"] == [1.0, 1.0]
    assert it["recall_by_event_scenario"]["interruption"]["events"] == 1


def test_never_stop_system():
    """No stops at all: recall 0, latency fully censored, F-beta undefined."""
    m = compute_metrics({"int60": interruption_sample()}, {"int60": []})
    it = m["interruptions"]
    assert it["total_events"] == 1
    assert it["handled"] == 0
    assert it["recall"] == 0.0
    assert it["never_stopped_rate"] == 1.0
    lat = it["decision_latency_ms"]
    assert lat["median"] is None and lat["p95"] is None
    assert lat["median_censored"] is True and lat["p95_censored"] is True
    assert lat["n_censored"] == 1 and lat["n_handled"] == 0
    sd = m["stop_decision"]
    assert sd["tp"] == 0 and sd["fp"] == 0 and sd["fn"] == 32
    assert sd["precision"] is None  # 0/0
    assert sd["recall"] == 0.0
    assert sd["fbeta"] is None


def test_first_stop_scored_and_premature_during_overlap():
    """Early stop inside overlap: scored once, premature-during-overlap; later stop is not a false stop."""
    m = compute_metrics({"int60": interruption_sample()}, {"int60": [1250, 1400]})
    it = m["interruptions"]
    assert it["handled"] == 1 and it["total_events"] == 1
    assert it["decision_latency_ms"]["median"] == pytest.approx(50.0)  # 1250 - 1200
    assert it["premature_during_overlap"] == 1
    assert it["premature_during_non_overlap"] == 0
    assert it["never_stopped_rate"] == 0.0
    # 1400 lies inside the active window -> not scored again, not a false stop.
    assert m["totals"]["false_stops_total"] == 0


def test_premature_during_non_overlap():
    """Stop before the physical overlap anchor: spurious-premature, negative decision latency."""
    labels = labels_df(
        60,
        agent_speaking={0: 1},
        scenario={0: "normal_turn", 20: "interruption"},
        interrupt_intent={24: 1},
        overlap_onset_ms={28: 1400},
        earliest_reasonable_stop_ms={32: 1600},
        action={32: "STOP_TTS"},
        speech_present={24: 1},
        primary_user={24: 1},
    )
    m = compute_metrics({"int60": labels}, {"int60": [1250]})
    it = m["interruptions"]
    assert it["handled"] == 1  # window [1200, 2100]
    assert it["premature_during_non_overlap"] == 1  # 1250 < 1600-100, before anchor 1400
    assert it["premature_during_overlap"] == 0
    assert it["decision_latency_ms"]["median"] == pytest.approx(-150.0)  # 1250 - 1400


def test_false_stop_on_backchannel_and_silence():
    """Stop during a backchannel (agent speaking) and during agent silence."""
    bc = labels_df(
        40,
        agent_speaking={0: 0, 5: 1},  # frames 0-4: agent silent
        scenario={10: "backchannel"},
        speech_present={10: 1},
    )
    m = compute_metrics({"bc40": bc}, {"bc40": [0, 700]})
    assert m["interruptions"]["total_events"] == 0
    assert m["interruptions"]["recall"] is None  # recall undefined with zero events
    assert m["totals"]["false_stops_total"] == 1
    fs = m["false_stops_by_scenario"]["backchannel"]
    assert fs["segments"] == 1 and fs["segments_with_false_stop"] == 1
    assert fs["false_stop_rate"] == 1.0
    assert fs["false_stop_frames"] == 1
    assert fs["agent_speech_frames"] == 30  # backchannel span (frames 10-39), agent audible
    hazards = m["deployment_hazards"]
    assert hazards["stops_while_agent_silent"]["count"] == 1
    assert hazards["stops_while_agent_silent"]["rate"] == pytest.approx(1 / 40)
    assert hazards["cross_run_false_stops"]["count"] == 0


def test_hesitation_commit_event_starts_at_commit():
    """hesitation_commit trap: pre-commit hesitation stop is a false stop; the event anchors at the commit point."""
    hc = labels_df(
        50,
        agent_speaking={5: 1},
        scenario={0: "hesitation_commit"},
        interrupt_intent={30: 1},  # commit at t=1500
        overlap_onset_ms={30: 1500},
        earliest_reasonable_stop_ms={34: 1700},
        action={34: "STOP_TTS"},
        speech_present={20: 1},  # hesitation speech before commit
        primary_user={20: 1},
    )
    m = compute_metrics({"hc50": hc}, {"hc50": [1000, 1700]})
    it = m["interruptions"]
    assert it["total_events"] == 1  # one event, user onset AT the commit point
    assert it["handled"] == 1
    assert it["decision_latency_ms"]["median"] == pytest.approx(200.0)  # 1700 - 1500
    assert it["recall"] == 1.0
    assert it["premature_during_overlap"] == 0
    fs = m["false_stops_by_scenario"]["hesitation_commit"]
    assert fs["segments"] == 1 and fs["segments_with_false_stop"] == 1
    assert fs["false_stop_rate"] == 1.0 and fs["false_stop_frames"] == 1
    sd = m["stop_decision"]
    assert sd["tp"] == 1 and sd["fp"] == 1 and sd["fn"] == 15
    assert it["recall_by_event_scenario"]["hesitation_commit"]["handled"] == 1


def test_cross_run_false_stop():
    """STOP leaking into the agent's second utterance after a handled event."""
    tr = labels_df(
        60,
        agent_speaking={0: 1, 25: 0, 30: 1},  # run 1: 0-24, gap 25-29, run 2: 30-59
        scenario={10: "interruption"},
        interrupt_intent={14: 1},
        overlap_onset_ms={14: 700},
        earliest_reasonable_stop_ms={18: 900},
        action={18: "STOP_TTS", 25: "KEEP"},  # GT: stop only for run 1
        speech_present={14: 1},
        primary_user={14: 1},
    )
    m = compute_metrics({"tr60": tr}, {"tr60": [900, 2000]})
    assert m["interruptions"]["handled"] == 1
    assert m["interruptions"]["decision_latency_ms"]["median"] == pytest.approx(200.0)
    cross = m["deployment_hazards"]["cross_run_false_stops"]
    assert cross["count"] == 1
    assert cross["samples_with_multiple_runs"] == 1
    assert cross["rate_per_sample"] == 1.0
    assert m["totals"]["false_stops_total"] == 1  # t=2000 during run 2
    assert m["false_stops_by_scenario"]["interruption"]["false_stop_frames"] == 1  # frame 40 sits in the interruption scenario span


def test_truncated_world_stop_after_agent_end():
    """Truncated world: event window fixed by labels; a late stop lands after agent_speaking drops."""
    tr = labels_df(
        35,
        agent_speaking={0: 1, 23: 0},  # agent cut at t=1150, just past the stop point
        scenario={10: "interruption"},
        interrupt_intent={10: 1},
        overlap_onset_ms={10: 500},
        earliest_reasonable_stop_ms={20: 1000},
        action={20: "STOP_TTS", 23: "KEEP"},
        speech_present={10: 1},
        primary_user={10: 1},
    )
    m = compute_metrics({"tr35": tr}, {"tr35": [1450]})
    it = m["interruptions"]
    assert it["total_events"] == 1 and it["handled"] == 1  # window [500, 1500]
    assert it["decision_latency_ms"]["median"] == pytest.approx(950.0)
    # The stop landed on a silent frame: deployment hazard, but still the scored event stop.
    assert m["deployment_hazards"]["stops_while_agent_silent"]["count"] == 1
    sd = m["stop_decision"]
    assert sd["tp"] == 0 and sd["fp"] == 1 and sd["fn"] == 3
    assert sd["precision"] == 0.0
    assert sd["fbeta"] is None  # p + r == 0
    assert m["totals"]["false_stops_total"] == 0


def test_stop_beyond_grace_is_never_stopped_and_false():
    m = compute_metrics({"int60": interruption_sample()}, {"int60": [1950]})
    it = m["interruptions"]
    assert it["handled"] == 0
    assert it["never_stopped_rate"] == 1.0
    assert m["totals"]["false_stops_total"] == 1  # frame 39: agent still speaking, past grace
    assert m["false_stops_by_scenario"]["interruption"]["false_stop_frames"] == 1


def test_censored_latency_percentiles():
    """Two handled + one never-stopped: median finite, P95 hits the imputed infinity."""
    labels = {
        "a": int_labels(erts_ms=1400, intent_frame=24, overlap_ms=1200, action_frame=28),
        "b": int_labels(erts_ms=1600, intent_frame=28, overlap_ms=1400, action_frame=32),
        "c": int_labels(erts_ms=1400, intent_frame=24, overlap_ms=1200, action_frame=28),
    }
    stops = {"a": [1400], "b": [1400], "c": []}
    m = compute_metrics(labels, stops)
    it = m["interruptions"]
    lat = it["decision_latency_ms"]
    assert lat["n_handled"] == 2 and lat["n_censored"] == 1
    assert lat["median"] == pytest.approx(200.0)  # sorted [0, 200, inf]; median = 200
    assert lat["median_censored"] is False
    assert lat["p95"] is None and lat["p95_censored"] is True
    assert it["recall"] == pytest.approx(2 / 3)
    assert it["never_stopped_rate"] == pytest.approx(1 / 3)

    direct = _censored_latency_stats(np.array([200.0, 400.0, np.inf]))
    assert direct["median"] == pytest.approx(400.0) and direct["median_censored"] is False
    assert direct["p95"] is None and direct["p95_censored"] is True
    assert direct["n_handled"] == 2 and direct["n_censored"] == 1
    empty = _censored_latency_stats(np.array([]))
    assert empty["median"] is None and empty["p95"] is None and empty["n_handled"] == 0


def test_fbeta_math():
    assert _fbeta(0.5, 0.5, 2.0) == pytest.approx(0.5)
    assert _fbeta(1.0, 0.0, 2.0) == 0.0
    assert _fbeta(0.0, 0.0, 2.0) is None
    assert _fbeta(1.0, 1.0, 2.0) == pytest.approx(1.0)


def test_operating_point_selection_tiebreak():
    def row(threshold, fbeta):
        return {"threshold_ms": threshold, "metrics": {"stop_decision": {"fbeta": fbeta}}}

    rows = [row(100, 0.4), row(200, 0.6), row(300, None)]
    assert select_operating_point(rows)["threshold_ms"] == 200
    tied = [row(200, 0.5), row(100, 0.5)]
    assert select_operating_point(tied)["threshold_ms"] == 100  # tie -> lowest threshold
    assert select_operating_point([row(100, None)]) is None


def test_bootstrap_ci_deterministic_and_degrenate():
    values = np.array([1.0, 1.0, 1.0, 0.0])
    first = _bootstrap_ci(values, np.mean, 1000, 0.95, seed=7)
    second = _bootstrap_ci(values, np.mean, 1000, 0.95, seed=7)
    assert first == second  # same seed -> identical CI
    assert 0.0 <= first["lo"] <= first["hi"] <= 1.0 and first["n"] == 4
    single = _bootstrap_ci(np.array([0.42]), np.mean, 1000, 0.95, seed=7)
    assert single == {"lo": 0.42, "hi": 0.42, "n": 1}
    empty = _bootstrap_ci(np.array([]), np.mean, 1000, 0.95, seed=7)
    assert empty == {"lo": None, "hi": None, "n": 0}


def test_e2e_latency_uses_supplied_cancel_constants():
    m = compute_metrics(
        {"int60": interruption_sample()},
        {"int60": [1400]},
        cancel_path_constants={"vad_ms": 10.0, "cancel_send_ms": 5.0, "audio_stop_ms": 35.0},
        cancel_path_source="upstream harness events.py",
        cancel_path_retrieved="2026-09-25",
    )
    e2e = m["interruptions"]["e2e_latency_ms"]
    assert e2e["n_handled"] == 1 and e2e["n_censored"] == 0
    assert e2e["cancel_path_total_ms"] == pytest.approx(50.0)
    assert e2e["median"] == pytest.approx(250.0)  # decision latency 200 + 50 ms cancel path
    assert e2e["p95"] == pytest.approx(250.0)
    assert e2e["source"] == "upstream harness events.py"
    assert e2e["retrieved"] == "2026-09-25"


def test_runs_of_exclusive_end():
    assert _runs_of(np.array([False, True, True, False, True])) == [(1, 3), (4, 5)]
    assert _runs_of(np.array([], dtype=bool)) == []


def test_average_ranks_handles_ties():
    assert list(_average_ranks(np.array([3.0, 1.0, 2.0]))) == pytest.approx([3.0, 1.0, 2.0])
    assert list(_average_ranks(np.array([1.0, 1.0, 3.0]))) == pytest.approx([1.5, 1.5, 3.0])


def test_pure_hesitation_and_never_committing_pair_are_not_events():
    """Pure hesitation: no event; its false stops land in the hesitation row."""
    pure = labels_df(50, agent_speaking={5: 1}, scenario={20: "hesitation"}, speech_present={20: 1}, primary_user={20: 1})
    m = compute_metrics({"pure50": pure}, {"pure50": [1000]})
    assert m["interruptions"]["total_events"] == 0
    assert m["false_stops_by_scenario"]["hesitation"]["false_stop_rate"] == pytest.approx(1.0)

    # A hesitation_commit region that never commits and carries no stop evidence is not an event.
    nc = labels_df(50, agent_speaking={5: 1}, scenario={0: "hesitation_commit"}, speech_present={20: 1}, primary_user={20: 1})
    m = compute_metrics({"nc50": nc}, {"nc50": [1000]})
    assert m["interruptions"]["total_events"] == 0
    assert m["false_stops_by_scenario"]["hesitation_commit"]["false_stop_rate"] == pytest.approx(1.0)


def test_event_extraction_fallbacks_are_flagged():
    """Labels without the overlap column: anchor falls back to the stop point and is flagged."""
    labels = labels_df(
        60,
        agent_speaking={0: 1},
        scenario={0: "normal_turn", 20: "interruption"},
        interrupt_intent={24: 1},
        earliest_reasonable_stop_ms={28: 1400},
        action={28: "STOP_TTS"},
    )
    m = compute_metrics({"s": labels}, {"s": [1400]})
    it = m["interruptions"]
    assert it["diagnostics"]["events_with_erts_fallback_anchor"] == 1
    assert it["decision_latency_ms"]["median"] == pytest.approx(0.0)  # anchor fell back to erts


def test_jitter_latency_diagnostic():
    """A jitter-tracking policy correlates 1.0 with the sampled jitter; a
    constant-latency policy has zero latency variance -> correlation undefined."""
    tracked_labels, tracked_stops = {}, {}
    constant_labels, constant_stops = {}, {}
    for i, jitter in enumerate((100, 200, 300)):
        erts_frame = 14 + jitter // 50
        labels = labels_df(
            40,
            agent_speaking={0: 1},
            scenario={10: "interruption"},
            interrupt_intent={14: 1},
            overlap_onset_ms={14: 700},
            earliest_reasonable_stop_ms={erts_frame: 700 + jitter},
            action={erts_frame: "STOP_TTS"},
        )
        tracked_labels[f"s{i}"] = labels
        tracked_stops[f"s{i}"] = [700 + jitter]  # latency == jitter -> perfect tracking
        constant_labels[f"s{i}"] = labels
        constant_stops[f"s{i}"] = [900]  # latency 200 regardless of jitter
    tracked = compute_metrics(tracked_labels, tracked_stops)["interruptions"]["jitter_latency_diagnostic"]
    assert tracked["n"] == 3
    assert tracked["pearson"] == pytest.approx(1.0)
    assert tracked["spearman"] == pytest.approx(1.0)
    constant = compute_metrics(constant_labels, constant_stops)["interruptions"]["jitter_latency_diagnostic"]
    assert constant["pearson"] is None  # zero variance in latency
    assert constant["n"] == 3


def test_event_dataclass_to_dict_roundtrip():
    event = InterruptionEvent(
        sample_id="s",
        index=0,
        user_onset_ms=100,
        end_ms=200,
        stop_point_ms=150,
        overlap_onset_ms=100,
        scenario="interruption",
        user_onset_is_fallback=False,
        stop_point_is_fallback=False,
        anchor_is_fallback=False,
    )
    assert event.to_dict()["stop_point_ms"] == 150
