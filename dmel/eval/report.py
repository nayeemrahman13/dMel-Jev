"""Markdown report generation for the dMel barge-in POC eval (contract v1).

The report opens with the contract's load-bearing assumption (synthetic ground
truth = policy-vs-labeler agreement, not real-world quality), then renders the
A-E arm comparison table and per-arm detail. Arms D/E are placeholders marked
"not trained in V1" unless real metrics are supplied.

Learned arms may pass one metrics file per training seed (``--metrics B=f1.json
--metrics B=f2.json ...``); the comparison table then shows seed means and the
per-arm section summarizes mean ± sd across seeds.
"""

from __future__ import annotations

import statistics

from dmel.eval.metrics import FALSE_STOP_SCENARIOS

ARM_NAMES = {
    "A": "VAD + heuristic (Silero)",
    "B": "dMel + LSTM",
    "C": "dMel + transformer",
    "D": "placeholder (arm D)",
    "E": "placeholder (arm E)",
}
ARM_ORDER = ("A", "B", "C", "D", "E")
NOT_TRAINED_IN_V1 = "not trained in V1"
NOT_PROVIDED = "not provided"
NOT_MEASURED = "not measured"
DASH = "\u2014"
INFINITY = "\u221e"

LOAD_BEARING_ASSUMPTION = (
    "**Load-bearing assumption.** The synthetic generator defines ground truth, so every number in "
    "this report measures policy-vs-labeler agreement, not real-world barge-in quality. A policy that "
    "matches the labeler's own conventions (for example, when overlapping speech counts as \"clear "
    "overlap\") will score well here even if it would behave differently on live calls. Mitigations: "
    "held-out augmentation slices, human spot-check listening on sampled clips, and real-call replay "
    "later."
)

PROTOCOL_TEXT = (
    "- Full threshold frontier on the **validation** split for every arm (see `sweep`).\n"
    "- Test-split numbers reported at **one pre-registered operating point per arm**, selected on "
    "validation by maximizing F-beta (beta = 2, recall-weighted) on stop decisions.\n"
    "- Event-level bootstrap CIs: 95%, 10k resamples, deterministic seed.\n"
    "- Learned arms: mean \u00b1 sd across \u22653 training seeds.\n"
    "- Fixture protocol: both the token path and the `dmel_token_ids=None` flag-only path run on the "
    "fixture set."
)

COMPARISON_HEADERS = (
    "Arm",
    "System",
    "Recall",
    "Premature (overlap)",
    "Premature (non-overlap)",
    "Decision latency med/P95 (ms)",
    "E2E latency med/P95 (ms)",
    "FS backchannel",
    "FS hesitation",
    "FS hes-commit",
    "FS noise",
    "FS bg-spk",
)

SEED_METRIC_PATHS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("recall", ("interruptions", "recall")),
    ("never_stopped_rate", ("interruptions", "never_stopped_rate")),
    ("premature_during_overlap_rate", ("interruptions", "premature_during_overlap_rate")),
    ("premature_during_non_overlap_rate", ("interruptions", "premature_during_non_overlap_rate")),
    ("decision_latency_median_ms", ("interruptions", "decision_latency_ms", "median")),
    ("decision_latency_p95_ms", ("interruptions", "decision_latency_ms", "p95")),
    ("fs_backchannel", ("false_stops_by_scenario", "backchannel", "false_stop_rate")),
    ("fs_hesitation", ("false_stops_by_scenario", "hesitation", "false_stop_rate")),
    ("fs_hesitation_commit", ("false_stops_by_scenario", "hesitation_commit", "false_stop_rate")),
    ("fs_noise", ("false_stops_by_scenario", "noise", "false_stop_rate")),
    ("fs_background_speaker", ("false_stops_by_scenario", "background_speaker", "false_stop_rate")),
)


def _fmt(value: float | int | None, nd: int = 3) -> str:
    return DASH if value is None else f"{value:.{nd}f}"


def _get_path(metrics: dict | None, path: tuple[str, ...]):
    cur: object = metrics
    for key in path:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(key)
    return cur


def _latency_pair(stats: dict | None) -> str:
    if not stats:
        return DASH
    def side(value, censored):
        if value is None:
            return INFINITY if censored else DASH
        return f"{value:.1f}"
    return f"{side(stats.get('median'), stats.get('median_censored', False))} / " \
        f"{side(stats.get('p95'), stats.get('p95_censored', False))}"


def _rate_cell(metrics: dict, key: str) -> str:
    value = metrics.get("interruptions", {}).get(key)
    if value is None:
        return DASH
    if isinstance(value, int):
        return str(value)
    return f"{value:.3f}"


def _latency_cell(metrics: dict) -> str:
    """Decision latency med/P95 with the never-stopped rate in the same cell."""
    stats = metrics.get("interruptions", {}).get("decision_latency_ms") or {}
    never = metrics.get("interruptions", {}).get("never_stopped_rate")
    cell = _latency_pair(stats)
    if never is not None:
        cell += f" \u00b7 never {never * 100:.1f}%"
    return cell


def _e2e_cell(metrics: dict) -> str:
    e2e = metrics.get("interruptions", {}).get("e2e_latency_ms")
    if not e2e:
        return NOT_MEASURED
    return _latency_pair(e2e)


def _fs_cell(metrics: dict, scenario: str) -> str:
    block = metrics.get("false_stops_by_scenario", {}).get(scenario) or {}
    return _fmt(block.get("false_stop_rate"))


def _arm_name(arm: str) -> str:
    return ARM_NAMES.get(arm, arm)


def _comparison_row(arm: str, metrics: dict | None, placeholder: str) -> str:
    if metrics is None:
        cells = " | ".join([placeholder] * 10)
        return f"| {arm} | {_arm_name(arm)} | {cells} |"
    return (
        f"| {arm} | {_arm_name(arm)} "
        f"| {_rate_cell(metrics, 'recall')} "
        f"| {_rate_cell(metrics, 'premature_during_overlap_rate')} "
        f"| {_rate_cell(metrics, 'premature_during_non_overlap_rate')} "
        f"| {_latency_cell(metrics)} "
        f"| {_e2e_cell(metrics)} "
        f"| {_fs_cell(metrics, 'backchannel')} "
        f"| {_fs_cell(metrics, 'hesitation')} "
        f"| {_fs_cell(metrics, 'hesitation_commit')} "
        f"| {_fs_cell(metrics, 'noise')} "
        f"| {_fs_cell(metrics, 'background_speaker')} |"
    )


def normalize_metrics_by_arm(
    metrics_by_arm: dict[str, dict | list[dict]],
) -> tuple[dict[str, dict], dict[str, list[dict]]]:
    """Split arm entries into (table metrics, per-arm seed lists).

    Single-dict arms map to themselves; list arms aggregate to seed means for
    the table and keep the full list for the per-arm seed summary.
    """
    table: dict[str, dict] = {}
    seeds: dict[str, list[dict]] = {}
    for arm, value in metrics_by_arm.items():
        if isinstance(value, list):
            seeds[arm] = value
            table[arm] = aggregate_seed_metrics(value)
        else:
            seeds[arm] = [value]
            table[arm] = value
    return table, seeds


def _aggregate_uncertainty(metrics_list: list[dict]) -> dict | None:
    """Element-wise mean of the per-seed bootstrap CIs (None when no seed has any)."""

    def ci_mean(path: tuple[str, ...]) -> list[float] | None:
        pairs = [
            v
            for m in metrics_list
            if isinstance(v := _get_path(m, path), list) and len(v) == 2 and all(x is not None for x in v)
        ]
        if not pairs:
            return None
        return [statistics.fmean([pair[0] for pair in pairs]), statistics.fmean([pair[1] for pair in pairs])]

    if not any(m.get("uncertainty") for m in metrics_list):
        return None
    n_resamples = next(
        (m["uncertainty"].get("n_resamples") for m in metrics_list if m.get("uncertainty", {}).get("n_resamples") is not None),
        None,
    )
    return {
        "recall_ci95": ci_mean(("uncertainty", "recall_ci95")),
        "decision_latency_median_ci95": ci_mean(("uncertainty", "decision_latency_median_ci95")),
        "n_resamples": n_resamples,
    }


def aggregate_seed_metrics(metrics_list: list[dict]) -> dict:
    """Seed-mean metrics dict containing exactly the fields table cells read."""

    def mean_of(path: tuple[str, ...]) -> float | None:
        values = [v for m in metrics_list if (v := _get_path(m, path)) is not None]
        return statistics.fmean(values) if values else None

    def any_of(path: tuple[str, ...]) -> bool:
        # Censoring flags are booleans: "any seed hit censoring" is the honest aggregate,
        # not their mean (0.5 would read as falsy in the table renderer).
        return any(bool(_get_path(m, path)) for m in metrics_list)

    def block(path_prefix: tuple[str, ...], keys: tuple[str, ...]) -> dict:
        return {k: mean_of((*path_prefix, k)) for k in keys}

    decision_latency = {
        **block(("interruptions", "decision_latency_ms"), ("median", "p95", "n_handled", "n_censored")),
        "median_censored": any_of(("interruptions", "decision_latency_ms", "median_censored")),
        "p95_censored": any_of(("interruptions", "decision_latency_ms", "p95_censored")),
    }
    first = metrics_list[0] if metrics_list else {}
    interruptions = first.get("interruptions", {})
    fs = {
        s: block(("false_stops_by_scenario", s), ("false_stop_rate",))
        for s in FALSE_STOP_SCENARIOS
    }
    aggregate = {
        "interruptions": {
            "total_events": mean_of(("interruptions", "total_events")),
            "handled": mean_of(("interruptions", "handled")),
            "never_stopped": mean_of(("interruptions", "never_stopped")),
            "premature_during_overlap": mean_of(("interruptions", "premature_during_overlap")),
            "premature_during_non_overlap": mean_of(("interruptions", "premature_during_non_overlap")),
            "recall": mean_of(("interruptions", "recall")),
            "never_stopped_rate": mean_of(("interruptions", "never_stopped_rate")),
            "premature_during_overlap_rate": mean_of(("interruptions", "premature_during_overlap_rate")),
            "premature_during_non_overlap_rate": mean_of(("interruptions", "premature_during_non_overlap_rate")),
            "decision_latency_ms": decision_latency,
            "e2e_latency_ms": interruptions.get("e2e_latency_ms"),
        },
        "false_stops_by_scenario": fs,
        "meta": {"seed_files": len(metrics_list)},
    }
    uncertainty = _aggregate_uncertainty(metrics_list)
    if uncertainty is not None:
        aggregate["uncertainty"] = uncertainty
    return aggregate


def seed_summary(metrics_list: list[dict]) -> str:
    """Markdown mean \u00b1 sd table across seed metric files."""
    header = "| Metric | Mean \u00b1 SD |"
    divider = "|---|---|"
    rows = []
    for name, path in SEED_METRIC_PATHS:
        values = [v for m in metrics_list if (v := _get_path(m, path)) is not None]
        if not values:
            rows.append(f"| {name} | {DASH} |")
        elif len(values) == 1:
            rows.append(f"| {name} | {values[0]:.3f} |")
        else:
            mean = statistics.fmean(values)
            sd = statistics.stdev(values) if len(values) > 1 else 0.0
            rows.append(f"| {name} | {mean:.3f} \u00b1 {sd:.3f} |")
    return "\n".join([header, divider, *rows])


def render_comparison_table(metrics_by_arm: dict[str, dict]) -> str:
    """The fixed A-E comparison table; missing arms degrade to placeholders."""
    header = "| " + " | ".join(COMPARISON_HEADERS) + " |"
    divider = "|" + "---|" * len(COMPARISON_HEADERS)
    rows = []
    for arm in ARM_ORDER:
        if arm in metrics_by_arm and metrics_by_arm[arm] is not None:
            rows.append(_comparison_row(arm, metrics_by_arm[arm], placeholder=""))
        elif arm in ("D", "E"):
            rows.append(_comparison_row(arm, None, placeholder=NOT_TRAINED_IN_V1))
        else:
            rows.append(_comparison_row(arm, None, placeholder=NOT_PROVIDED))
    return "\n".join([header, divider, *rows])


def _false_stop_detail_table(metrics: dict) -> str:
    header = "| Scenario | Segments | Segments w/ false stop | False-stop rate | False-stop frames | Agent-speech frames |"
    divider = "|" + "---|" * 6
    rows = []
    for scenario, block in metrics.get("false_stops_by_scenario", {}).items():
        rows.append(
            f"| {scenario} | {block.get('segments', 0)} | {block.get('segments_with_false_stop', 0)} "
            f"| {_fmt(block.get('false_stop_rate'))} | {block.get('false_stop_frames', 0)} "
            f"| {block.get('agent_speech_frames', 0)} |"
        )
    return "\n".join([header, divider, *rows])


def _hazards_table(metrics: dict) -> str:
    hazards = metrics.get("deployment_hazards", {})
    cross = hazards.get("cross_run_false_stops", {})
    silent = hazards.get("stops_while_agent_silent", {})
    rate = cross.get("rate_per_sample")
    silent_rate = silent.get("rate")
    return "\n".join(
        [
            "| Deployment hazard | Count | Rate |",
            "|---|---|---|",
            f"| Cross-run false stop (STOP leaking into a later agent run, \u22652-run samples) "
            f"| {cross.get('count', 0)} | {_fmt(rate)} (per multi-run sample) |",
            f"| STOP during agent silence | {silent.get('count', 0)} "
            f"| {_fmt(silent_rate)} (of all label frames) |",
        ]
    )


def _cancel_path_lines(metrics: dict) -> list[str]:
    e2e = metrics.get("interruptions", {}).get("e2e_latency_ms")
    if not e2e:
        return [
            "Estimated end-to-end latency is **omitted**: no measured cancel-path constants "
            "(`vad_ms`, `cancel_send_ms`, `audio_stop_ms` measured on the "
            "upstream realtime harness cancel path) were supplied. Pass "
            "`--cancel-path-constants` with measured values; the report will cite them with their "
            "retrieval date. Do not substitute estimates for measurements.",
        ]
    constants = e2e.get("cancel_path_constants", {})
    rendered = ", ".join(f"{k}={v}" for k, v in constants.items())
    source = e2e.get("source") or "unspecified source"
    retrieved = e2e.get("retrieved") or "retrieval date not recorded"
    return [
        f"Estimated end-to-end latency adds measured cancel-path constants to decision latency: "
        f"{rendered} (total {e2e.get('cancel_path_total_ms'):.1f} ms; source: {source}; retrieved: {retrieved}).",
    ]


def _arm_section(arm: str, metrics: dict, seed_files: list[dict]) -> str:
    interruptions = metrics.get("interruptions", {})
    totals = metrics.get("totals", {})
    latency = interruptions.get("decision_latency_ms") or {}
    lines = [
        f"### Arm {arm} \u2014 {_arm_name(arm)}",
        "",
        f"- Interruption events: {interruptions.get('total_events', 0)} "
        f"(handled {interruptions.get('handled', 0)}, never stopped {interruptions.get('never_stopped', 0)}, "
        f"premature during overlap {interruptions.get('premature_during_overlap', 0)}, "
        f"premature during non-overlap {interruptions.get('premature_during_non_overlap', 0)})",
        f"- Decision latency (ms, anchor `overlap_onset_ms`): median {_fmt(latency.get('median'), 1)} "
        f"\u00b7 P95 {_fmt(latency.get('p95'), 1)} "
        f"(censored flags: median {latency.get('median_censored', False)}, P95 {latency.get('p95_censored', False)}; "
        f"n handled {latency.get('n_handled', 0)}, censored {latency.get('n_censored', 0)})",
        f"- Total false stops: {totals.get('false_stops_total', 0)} "
        f"(stops outside label timeline: {totals.get('stops_outside_labels', 0)})",
        f"- Samples scored: {totals.get('samples', 0)}",
        "",
        "Deployment hazards:",
        "",
        _hazards_table(metrics),
        "",
        "False stops by scenario:",
        "",
        _false_stop_detail_table(metrics),
        "",
        *_cancel_path_lines(metrics),
    ]
    diag = interruptions.get("jitter_latency_diagnostic") or {}
    if diag.get("n"):
        lines.append(
            f"Jitter-latency diagnostic (standing): Pearson {_fmt(diag.get('pearson'))}, "
            f"Spearman {_fmt(diag.get('spearman'))} over n={diag.get('n')} \u2014 {diag.get('note', '')}"
        )
        lines.append("")
    uncertainty = metrics.get("uncertainty")
    if uncertainty:
        recall_ci = uncertainty.get("recall_ci95") or [None, None]
        lat_ci = uncertainty.get("decision_latency_median_ci95") or [None, None]
        lines.append(
            f"Bootstrap CIs (95%, {uncertainty.get('n_resamples')} resamples): recall "
            f"[{_fmt(recall_ci[0])}, {_fmt(recall_ci[1])}]; decision-latency median "
            f"[{_fmt(lat_ci[0], 1)}, {_fmt(lat_ci[1], 1)}] ms (conditional on handled events; "
            f"never-stopped rate reported alongside latency)."
        )
        lines.append("")
    stop_decision = metrics.get("stop_decision")
    if stop_decision:
        lines.append(
            f"Stop-decision frame metrics: precision {_fmt(stop_decision.get('precision'))}, "
            f"recall {_fmt(stop_decision.get('recall'))}, F-beta(\u03b2={stop_decision.get('beta')}) "
            f"{_fmt(stop_decision.get('fbeta'))} \u2014 the pre-registered operating point maximizes "
            "this on validation."
        )
        lines.append("")
    if len(seed_files) > 1:
        lines.append(f"Seed summary ({len(seed_files)} seed files, mean \u00b1 sd):")
        lines.append("")
        lines.append(seed_summary(seed_files))
        lines.append("")
    meta = metrics.get("meta") or {}
    meta_lines = [f"{key}: {value}" for key, value in meta.items() if key != "seed_files" and value is not None]
    if meta_lines:
        lines.append(f"Run metadata: {'; '.join(meta_lines)}")
        lines.append("")
    return "\n".join(lines)


def render_report(metrics_by_arm: dict[str, dict | list[dict]]) -> str:
    """Full markdown report; the load-bearing assumption opens the body."""
    table_metrics, seeds = normalize_metrics_by_arm(metrics_by_arm)
    parts = [
        "# dMel barge-in POC \u2014 evaluation report (V1)",
        "",
        "## Load-bearing assumption",
        "",
        LOAD_BEARING_ASSUMPTION,
        "",
        "## Arm comparison",
        "",
        render_comparison_table(table_metrics),
        "",
        "FS = false-stop rate (segments with a false stop / segments of that scenario). "
        f"Contract false-stop scenarios: {', '.join(FALSE_STOP_SCENARIOS)}. "
        "Decision-latency cells carry the never-stopped rate (events censored at infinity are "
        "imputed there, never shown as a standalone handled-events column).",
        "",
        "## Protocol",
        "",
        PROTOCOL_TEXT,
        "",
    ]
    provided = sorted((arm, metrics) for arm, metrics in table_metrics.items() if metrics is not None)
    if provided:
        parts.extend(["## Per-arm detail", ""])
        for arm, metrics in provided:
            parts.append(_arm_section(arm, metrics, seeds.get(arm, [metrics])))
    parts.extend(
        [
            "## Metric definitions",
            "",
            "- **Event active window**: `[user onset, earliest_reasonable_stop_ms + 500 ms grace]`; "
            "user onset is the first `interrupt_intent == 1` frame (the commit point for "
            "hesitation-then-commit samples).",
            "- **recall**: STOP_TTS emitted while the event is active / total interruption events; "
            "the first STOP per event is the scored one.",
            "- **premature-stop rate**: first in-window stop >100 ms before `earliest_reasonable_stop_ms`, "
            "split into during-overlap (early but evidence-responsive) vs during-non-overlap (spurious) "
            "by the `overlap_onset_ms` anchor.",
            "- **Decision latency** = `t_pred_stop \u2212 overlap_onset_ms` (physical anchor, NOT the "
            "jittered stop-point reference), median + P95 over handled events with never-stopped events "
            "censored at infinity in the same cell.",
            "- **Jitter-latency diagnostic**: per-event latency correlated with the sampled jitter "
            "(`erts \u2212 overlap onset`) \u2014 an evidence-following policy is independent of the "
            "jitter draw; a prior-matcher tracks it.",
            "- **false stop**: STOP_TTS during agent speech (per labels) with no active interruption "
            "event; per-scenario rate = segments with a false stop / segments of that scenario "
            "(maximal runs).",
            "- **E2E latency** = decision latency + measured cancel-path constants "
            "(`vad_ms + cancel_send_ms + audio_stop_ms`), cited with source and retrieval date.",
            "",
            "## Decision log format",
            "",
            'jsonl, one record per 50 ms step: `{"sample_id", "frame_index", "t_ms", "action"}` with '
            "`action` in {KEEP, STOP_TTS}; common alias field names are accepted on read.",
            "",
        ]
    )
    return "\n".join(parts)
