"""Metric engine for the dMel barge-in POC evaluation (contract v1).

Definitions (contract v1, art_owEFcMrX):

- An interruption event exists where ``scenario`` is ``interruption`` or
  ``hesitation_commit`` (or ``interrupt_intent == 1``). Its user onset is the
  first ``interrupt_intent == 1`` frame — for hesitation-then-commit samples
  that is the commit point, so interrupts are scored from the commit point.
- Event active window: ``[user onset, earliest_reasonable_stop_ms + 500 ms]``.
- Recall: first STOP_TTS emitted inside the active window / total events.
- Premature stops (first stop >100 ms before the stop point) split into
  during-overlap (early but evidence-responsive) vs during-non-overlap
  (spurious), anchored at ``overlap_onset_ms``.
- Decision latency = ``t_pred_stop - overlap_onset_ms`` — the physical anchor,
  NOT the jittered stop-point reference a prior-matching policy could game.
  Median + P95 over handled events with never-stopped events censored at
  infinity; the never-stopped rate prints in the same cell as latency.
- Standing diagnostic: per-event latency correlated with the sampled jitter
  (``erts - overlap_onset``). Evidence-following policies are independent of
  the jitter draw; prior-matchers track it.
- False stops: STOP_TTS during agent speech with no active interruption event,
  broken out per scenario (backchannel, hesitation, hesitation_commit, noise,
  background_speaker), plus deployment hazards: cross-run false stops on the
  >=2-agent-runs subset and STOP during agent silence.
- Stop-decision confusion (frame-level vs the ground-truth ``action`` column)
  with F-beta (beta=2) for the pre-registered operating-point selection:
  full frontier on validation, one point per arm on test.
- Uncertainty: event-level bootstrap CIs (95%, 10k resamples) on headline
  metrics; learned arms additionally report mean +- sd across seeds upstream.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

STOP_TTS = "STOP_TTS"
KEEP = "KEEP"
ACTIONS = (KEEP, STOP_TTS)
MS_PER_STEP = 50
EVENT_GRACE_MS = 500
PREMATURE_LEAD_MS = 100
DEFAULT_STOP_DECISION_BETA = 2.0

INTERRUPTION_SCENARIOS = ("interruption", "hesitation_commit")
SCENARIOS = (
    "interruption",
    "backchannel",
    "normal_turn",
    "noise",
    "background_speaker",
    "hesitation",
    "hesitation_commit",
)
FALSE_STOP_SCENARIOS = (
    "backchannel",
    "hesitation",
    "hesitation_commit",
    "noise",
    "background_speaker",
)

JITTER_NOTE = (
    "per-event decision latency vs sampled jitter (erts - overlap onset): an "
    "evidence-following policy is independent of the jitter draw (correlation ~0); "
    "a prior-matcher tracks the sampled window (positive correlation)"
)


@dataclass
class InterruptionEvent:
    """One interruption event extracted from a sample's labels."""

    sample_id: str
    index: int
    user_onset_ms: int
    end_ms: int
    stop_point_ms: int  # earliest_reasonable_stop_ms; -1 if unlabeled
    overlap_onset_ms: int | None  # physical anchor; None if unlabeled
    scenario: str
    user_onset_is_fallback: bool
    stop_point_is_fallback: bool
    anchor_is_fallback: bool

    def to_dict(self) -> dict:
        return {
            "sample_id": self.sample_id,
            "index": self.index,
            "user_onset_ms": self.user_onset_ms,
            "end_ms": self.end_ms,
            "stop_point_ms": self.stop_point_ms,
            "overlap_onset_ms": self.overlap_onset_ms,
            "scenario": self.scenario,
            "user_onset_is_fallback": self.user_onset_is_fallback,
            "stop_point_is_fallback": self.stop_point_is_fallback,
            "anchor_is_fallback": self.anchor_is_fallback,
        }


def _runs_of(mask: np.ndarray) -> list[tuple[int, int]]:
    """Maximal True runs of a boolean array as (start, end) index pairs."""
    if len(mask) == 0:
        return []
    padded = np.concatenate([[False], mask.astype(bool), [False]])
    diffs = np.diff(padded.astype(np.int8))
    starts = np.flatnonzero(diffs == 1)
    ends = np.flatnonzero(diffs == -1)
    return list(zip(starts.tolist(), ends.tolist()))


def _col(labels: pd.DataFrame, name: str, default: np.ndarray | None) -> np.ndarray | None:
    if name in labels.columns:
        return labels[name].to_numpy()
    return default


def _t_ms_col(labels: pd.DataFrame) -> np.ndarray:
    if "t_ms" in labels.columns:
        return labels["t_ms"].to_numpy(dtype=float)
    return np.arange(len(labels), dtype=float) * MS_PER_STEP


def extract_interruption_events(sample_id: str, labels: pd.DataFrame) -> list[InterruptionEvent]:
    """Extract interruption events from one sample's label frame.

    Robust to two hesitation_commit layouts: the scenario tag covering the
    whole sample (event starts at the interrupt_intent onset = commit point)
    or only the post-commit frames.
    """
    n = len(labels)
    t_ms = _t_ms_col(labels)
    scenario = _col(labels, "scenario", None)
    intent = _col(labels, "interrupt_intent", np.zeros(n))
    erts_col = _col(labels, "earliest_reasonable_stop_ms", np.full(n, -1.0))
    overlap_col = _col(labels, "overlap_onset_ms", np.full(n, -1.0))
    action = _col(labels, "action", None)
    gt_stop = (action == STOP_TTS) if action is not None else np.zeros(n, dtype=bool)

    if scenario is not None:
        event_mask = np.isin(scenario, INTERRUPTION_SCENARIOS) | (intent == 1)
    else:
        event_mask = intent == 1

    events: list[InterruptionEvent] = []
    for run_start, run_end in _runs_of(event_mask):
        region = slice(run_start, run_end)
        intent_frames = np.flatnonzero(intent[region] == 1) + run_start
        has_stop_signal = bool((erts_col[region] >= 0).any()) or bool(gt_stop[region].any())
        if len(intent_frames) == 0 and not has_stop_signal:
            # A hesitation_commit region that never commits and carries no
            # stop evidence is effectively pure hesitation — not an event.
            continue
        if len(intent_frames) > 0:
            onset_idx = int(intent_frames[0])
            onset_fallback = False
        else:
            onset_idx = run_start
            onset_fallback = True
        user_onset_ms = int(t_ms[onset_idx])

        after_onset = slice(onset_idx, run_end)
        stop_point = -1
        stop_fallback = False
        erts_frames = np.flatnonzero(erts_col[after_onset] >= 0) + onset_idx
        if len(erts_frames) > 0:
            stop_point = int(erts_col[erts_frames[0]])
        else:
            gt_frames = np.flatnonzero(gt_stop[after_onset]) + onset_idx
            if len(gt_frames) > 0:
                stop_point = int(t_ms[gt_frames[0]])
                stop_fallback = True

        anchor: int | None = None
        anchor_fallback = False
        ov_frames = np.flatnonzero(overlap_col[after_onset] >= 0) + onset_idx
        if len(ov_frames) > 0:
            anchor = int(overlap_col[ov_frames[0]])
        else:
            # Fall back to the stop-point reference (jittered) when the
            # physical anchor column is absent — flagged so readers know the
            # latency numbers inherit the prior-matching weakness.
            if stop_point >= 0:
                anchor = stop_point
                anchor_fallback = True

        end_ms = stop_point + EVENT_GRACE_MS if stop_point >= 0 else int(t_ms[run_end - 1])
        ev_scenario = str(scenario[onset_idx]) if scenario is not None else "unknown"
        events.append(
            InterruptionEvent(
                sample_id=sample_id,
                index=len(events),
                user_onset_ms=user_onset_ms,
                end_ms=end_ms,
                stop_point_ms=stop_point,
                overlap_onset_ms=anchor,
                scenario=ev_scenario,
                user_onset_is_fallback=onset_fallback,
                stop_point_is_fallback=stop_fallback,
                anchor_is_fallback=anchor_fallback,
            )
        )
    return events


def first_stop_in_window(stops_ms: list[int], start_ms: int, end_ms: int) -> int | None:
    """First predicted stop inside the active window, or None."""
    for t in sorted(stops_ms):
        if start_ms <= t <= end_ms:
            return t
    return None


def _fbeta(precision: float, recall: float, beta: float) -> float | None:
    if precision + recall == 0:
        return None
    beta_sq = beta * beta
    return (1 + beta_sq) * precision * recall / (beta_sq * precision + recall)


def _average_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    sorted_vals = values[order]
    ranks = np.empty(len(values), dtype=float)
    i = 0
    while i < len(values):
        j = i
        while j + 1 < len(values) and sorted_vals[j + 1] == sorted_vals[i]:
            j += 1
        ranks[order[i : j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    return ranks


def _pearson(x: np.ndarray, y: np.ndarray) -> float | None:
    if len(x) < 3 or np.std(x) == 0 or np.std(y) == 0:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def _bootstrap_ci(
    values: np.ndarray,
    statistic: callable,  # type: ignore[valid-type]
    n_resamples: int,
    confidence: float,
    seed: int,
) -> dict:
    """Percentile bootstrap CI for ``statistic`` over event-level resamples."""
    values = np.asarray(values, dtype=float)
    n = len(values)
    if n == 0:
        return {"lo": None, "hi": None, "n": 0}
    if n == 1:
        point = float(statistic(values))
        return {"lo": point, "hi": point, "n": 1}
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_resamples, n))
    stats = statistic(values[idx], axis=1)
    alpha = (1 - confidence) / 2
    lo, hi = np.percentile(stats, [100 * alpha, 100 * (1 - alpha)])
    return {"lo": float(lo), "hi": float(hi), "n": n}


def _censored_latency_stats(latencies_ms_with_inf: np.ndarray) -> dict:
    """Median/P95 treating never-stopped events as censored at infinity.

    A quantile that lands on imputed infinity is reported as None with a
    censored flag rather than a finite number.
    """
    values = np.asarray(latencies_ms_with_inf, dtype=float)
    n_total = int(values.size)
    n_censored = int(np.isinf(values).sum())
    stats: dict = {
        "n_handled": n_total - n_censored,
        "n_censored": n_censored,
        "median_censored": False,
        "p95_censored": False,
    }
    if n_total == 0:
        stats.update(median=None, p95=None)
        return stats
    ordered = np.sort(values)

    def quantile(q: float) -> tuple[float | None, bool]:
        # np.percentile's lerp returns nan when a neighbor is inf (inf * 0), so
        # resolve the quantile by hand: a quantile anchored at (or interpolating
        # into) the imputed-infinity tail is censored, not finite.
        pos = q * (n_total - 1)
        lo = int(np.floor(pos))
        frac = pos - lo
        a = ordered[lo]
        if np.isinf(a):
            return None, True
        b = ordered[min(lo + 1, n_total - 1)]
        if np.isinf(b):
            return (float(a), False) if frac == 0 else (None, True)
        if frac == 0:
            return float(a), False
        return float(a + (b - a) * frac), False

    median, med_censored = quantile(0.5)
    p95, p95_censored = quantile(0.95)
    stats.update(
        median=median,
        median_censored=med_censored,
        p95=p95,
        p95_censored=p95_censored,
    )
    return stats


def _scenario_segment_stats(
    labels: pd.DataFrame,
    scenario: str,
    false_stop_frames: set[int],
) -> dict:
    """Segments (maximal runs) of one scenario and their false-stop exposure."""
    if "scenario" not in labels.columns:
        return {"segments": 0, "segments_with_false_stop": 0, "false_stop_rate": None, "false_stop_frames": 0, "agent_speech_frames": 0}
    scenario_mask = labels["scenario"].to_numpy() == scenario
    agent_speaking = labels["agent_speaking"].to_numpy().astype(bool)
    segments = _runs_of(scenario_mask)
    seg_with_fs = sum(1 for start, end in segments if any(f in false_stop_frames for f in range(start, end)))
    return {
        "segments": len(segments),
        "segments_with_false_stop": seg_with_fs,
        "false_stop_rate": (seg_with_fs / len(segments)) if segments else None,
        "false_stop_frames": len(false_stop_frames),
        "agent_speech_frames": int((scenario_mask & agent_speaking).sum()),
    }


def compute_metrics(
    labels_by_sample: dict[str, pd.DataFrame],
    stops_by_sample: dict[str, list[int]],
    *,
    stop_decision_beta: float = DEFAULT_STOP_DECISION_BETA,
    cancel_path_constants: dict[str, float] | None = None,
    cancel_path_source: str | None = None,
    cancel_path_retrieved: str | None = None,
    bootstrap: bool = True,
    n_resamples: int = 10_000,
    ci_seed: int = 0,
    meta: dict | None = None,
) -> dict:
    """Compute all contract-v1 metrics for one policy over labeled samples."""
    events: list[InterruptionEvent] = []
    for sample_id in sorted(labels_by_sample):
        events.extend(extract_interruption_events(sample_id, labels_by_sample[sample_id]))

    outcomes: list[float] = []
    latencies: list[float] = []
    jitters: list[float] = []
    latencies_for_diag: list[float] = []
    premature_overlap = 0
    premature_non_overlap = 0
    per_scenario_recall: dict[str, list[float]] = {}
    anchor_fallbacks = 0
    onset_fallbacks = 0
    stop_point_fallbacks = 0
    no_anchor_events = 0

    for event in events:
        stops = stops_by_sample.get(event.sample_id, [])
        t_pred = first_stop_in_window(stops, event.user_onset_ms, event.end_ms)
        handled = t_pred is not None
        outcomes.append(1.0 if handled else 0.0)
        per_scenario_recall.setdefault(event.scenario, []).append(1.0 if handled else 0.0)
        if event.user_onset_is_fallback:
            onset_fallbacks += 1
        if event.stop_point_is_fallback:
            stop_point_fallbacks += 1
        if event.anchor_is_fallback:
            anchor_fallbacks += 1
        if not handled:
            continue
        if event.overlap_onset_ms is None:
            no_anchor_events += 1
            continue  # latency undefined without any anchor
        latency = float(t_pred - event.overlap_onset_ms)
        latencies.append(latency)
        if event.stop_point_ms >= 0 and not event.anchor_is_fallback:
            jitters.append(float(event.stop_point_ms - event.overlap_onset_ms))
            latencies_for_diag.append(latency)
        if t_pred < event.stop_point_ms - PREMATURE_LEAD_MS:
            anchor = event.overlap_onset_ms
            if anchor is not None and t_pred < anchor:
                premature_non_overlap += 1
            else:
                premature_overlap += 1

    total_events = len(events)
    handled = int(sum(outcomes))
    never_stopped = total_events - handled

    lat_with_inf = np.concatenate([np.asarray(latencies, dtype=float), np.full(never_stopped, np.inf)])
    decision_latency = _censored_latency_stats(lat_with_inf)

    diag: dict = {"pearson": None, "spearman": None, "n": len(latencies_for_diag), "note": JITTER_NOTE}
    if len(latencies_for_diag) >= 3:
        lat_arr = np.asarray(latencies_for_diag)
        jit_arr = np.asarray(jitters)
        diag["pearson"] = _pearson(lat_arr, jit_arr)
        diag["spearman"] = _pearson(_average_ranks(lat_arr), _average_ranks(jit_arr))

    e2e_block: dict | None = None
    if cancel_path_constants:
        floor = float(sum(cancel_path_constants[k] for k in ("vad_ms", "cancel_send_ms", "audio_stop_ms")))
        e2e_stats = _censored_latency_stats(lat_with_inf + floor)
        e2e_block = {
            **e2e_stats,
            "cancel_path_constants": dict(cancel_path_constants),
            "cancel_path_total_ms": floor,
            "source": cancel_path_source,
            "retrieved": cancel_path_retrieved,
        }

    # Frame-level stop-decision confusion against the ground-truth action column.
    stop_decision: dict | None = None
    tp = fp = fn = 0
    scored_gt_frames = False
    for sample_id, labels in labels_by_sample.items():
        if "action" not in labels.columns:
            continue
        scored_gt_frames = True
        pred_stops = {int(round(t / MS_PER_STEP)) for t in stops_by_sample.get(sample_id, [])}
        gt = labels["action"].to_numpy()
        for i in range(len(labels)):
            pred_stop = i in pred_stops
            gt_stop = str(gt[i]) == STOP_TTS
            if pred_stop and gt_stop:
                tp += 1
            elif pred_stop and not gt_stop:
                fp += 1
            elif not pred_stop and gt_stop:
                fn += 1
    if scored_gt_frames:
        precision = tp / (tp + fp) if (tp + fp) else None
        recall_frame = tp / (tp + fn) if (tp + fn) else None
        stop_decision = {
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "precision": precision,
            "recall": recall_frame,
            "fbeta": _fbeta(precision, recall_frame, stop_decision_beta) if precision is not None and recall_frame is not None else None,
            "beta": stop_decision_beta,
        }

    # False stops: STOP during agent speech with no active interruption event.
    false_stop_frames: list[tuple[str, int, str]] = []  # (sample_id, frame, scenario bucket)
    false_stops_total = 0
    stops_silent = 0
    stops_outside = 0
    cross_run_stops = 0
    samples_with_multiple_runs = 0
    total_frames = 0

    for sample_id, labels in labels_by_sample.items():
        total_frames += len(labels)
        agent_speaking = labels["agent_speaking"].to_numpy().astype(bool)
        t_ms = _t_ms_col(labels)
        scenario = _col(labels, "scenario", None)
        sample_events = [e for e in events if e.sample_id == sample_id]
        runs = _runs_of(agent_speaking)
        if len(runs) >= 2:
            samples_with_multiple_runs += 1
        stops = sorted(stops_by_sample.get(sample_id, []))
        for t in stops:
            frame = int(round(t / MS_PER_STEP))
            if frame < 0 or frame >= len(labels):
                stops_outside += 1
                continue
            if not agent_speaking[frame]:
                stops_silent += 1
                continue
            active = any(e.user_onset_ms <= t <= e.end_ms for e in sample_events)
            if active:
                continue
            false_stops_total += 1
            bucket = str(scenario[frame]) if scenario is not None else "unknown"
            if bucket not in SCENARIOS:
                bucket = "unknown"
            false_stop_frames.append((sample_id, frame, bucket))
            run_index = next((ri for ri, (start, end) in enumerate(runs) if start <= frame < end), None)
            if len(runs) >= 2 and run_index is not None and run_index >= 1:
                cross_run_stops += 1

    fs_by_scenario: dict[str, dict] = {}
    for s in list(SCENARIOS) + ["unknown"]:
        segments_total = 0
        seg_with_fs = 0
        agent_frames = 0
        fs_count = 0
        for sample_id, labels in labels_by_sample.items():
            if "scenario" not in labels.columns:
                continue
            # False-stop frames stay scoped to their own sample — frame indices
            # from different samples must never match across segments.
            frames_here = {f for sid, f, b in false_stop_frames if sid == sample_id and b == s}
            stats = _scenario_segment_stats(labels, s, frames_here)
            segments_total += stats["segments"]
            seg_with_fs += stats["segments_with_false_stop"]
            agent_frames += stats["agent_speech_frames"]
            fs_count += len(frames_here)
        fs_by_scenario[s] = {
            "segments": segments_total,
            "segments_with_false_stop": seg_with_fs,
            "false_stop_rate": (seg_with_fs / segments_total) if segments_total else None,
            "false_stop_frames": fs_count,
            "agent_speech_frames": agent_frames,
        }

    uncertainty: dict | None = None
    if bootstrap and total_events > 0:
        recall_ci = _bootstrap_ci(
            np.asarray(outcomes), np.mean, n_resamples, 0.95, ci_seed
        )
        lat_ci = (
            _bootstrap_ci(np.asarray(latencies), np.median, n_resamples, 0.95, ci_seed + 1)
            if latencies
            else {"lo": None, "hi": None, "n": 0}
        )
        uncertainty = {
            "recall_ci95": [recall_ci["lo"], recall_ci["hi"]],
            "decision_latency_median_ci95": [lat_ci["lo"], lat_ci["hi"]],
            "decision_latency_ci_scope": "conditional on handled events; never-stopped rate reported alongside latency",
            "n_resamples": n_resamples,
            "confidence": 0.95,
            "seed": ci_seed,
        }

    return {
        "totals": {
            "samples": len(labels_by_sample),
            "label_frames": total_frames,
            "predicted_stop_frames": sum(len(v) for v in stops_by_sample.values()),
            "interruption_events": total_events,
            "false_stops_total": false_stops_total,
            "stops_while_agent_silent": stops_silent,
            "stops_outside_labels": stops_outside,
        },
        "interruptions": {
            "total_events": total_events,
            "handled": handled,
            "recall": (handled / total_events) if total_events else None,
            "never_stopped": never_stopped,
            "never_stopped_rate": (never_stopped / total_events) if total_events else None,
            "premature_during_overlap": premature_overlap,
            "premature_during_overlap_rate": (premature_overlap / total_events) if total_events else None,
            "premature_during_non_overlap": premature_non_overlap,
            "premature_during_non_overlap_rate": (premature_non_overlap / total_events) if total_events else None,
            "premature_total": premature_overlap + premature_non_overlap,
            "premature_rate": ((premature_overlap + premature_non_overlap) / total_events) if total_events else None,
            "decision_latency_ms": decision_latency,
            "e2e_latency_ms": e2e_block,
            "jitter_latency_diagnostic": diag,
            "recall_by_event_scenario": {
                s: {
                    "events": len(v),
                    "handled": int(sum(v)),
                    "recall": (sum(v) / len(v)) if v else None,
                }
                for s, v in sorted(per_scenario_recall.items())
            },
            "diagnostics": {
                "events_without_overlap_anchor": no_anchor_events,
                "events_with_erts_fallback_anchor": anchor_fallbacks,
                "events_with_onset_fallback": onset_fallbacks,
                "events_with_stop_point_fallback": stop_point_fallbacks,
            },
        },
        "stop_decision": stop_decision,
        "false_stops_by_scenario": fs_by_scenario,
        "deployment_hazards": {
            "cross_run_false_stops": {
                "count": cross_run_stops,
                "samples_with_multiple_runs": samples_with_multiple_runs,
                "rate_per_sample": (cross_run_stops / samples_with_multiple_runs) if samples_with_multiple_runs else None,
            },
            "stops_while_agent_silent": {
                "count": stops_silent,
                "total_frames": total_frames,
                "rate": (stops_silent / total_frames) if total_frames else None,
            },
        },
        "uncertainty": uncertainty,
        "events_detail": [e.to_dict() for e in events],
        "meta": {
            **(meta or {}),
            "definitions": {
                "event_active_window": "[user onset, earliest_reasonable_stop_ms + 500 ms grace]",
                "premature_lead_ms": PREMATURE_LEAD_MS,
                "decision_latency_anchor": "overlap_onset_ms",
                "stop_decision_beta": stop_decision_beta,
            },
        },
    }
