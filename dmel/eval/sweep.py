"""Threshold-frontier sweep for threshold-parameterized policies.

Contract-v1 protocol: every arm reports a full swept operating frontier on the
validation split; test-split numbers are reported at ONE pre-registered
operating point per arm, selected on validation by maximizing F-beta
(beta = 2, recall-weighted) on stop decisions. The selected point is marked in
the rendered table; eval-set tuning beyond this rule is a protocol violation.
"""

from __future__ import annotations

from typing import Any, Callable

from dmel.eval.metrics import DEFAULT_STOP_DECISION_BETA
from dmel.eval.runner import run_shard

PolicyFactory = Callable[[float], Any]


def sweep_thresholds(
    policy_factory: PolicyFactory,
    shard_root: str,
    thresholds_ms: list[float],
    *,
    provide_agent_stem: bool = True,
    max_samples: int | None = None,
    ignore_dmel_cache: bool = False,
    stop_decision_beta: float = DEFAULT_STOP_DECISION_BETA,
    cancel_path_constants: dict[str, float] | None = None,
    cancel_path_source: str | None = None,
    cancel_path_retrieved: str | None = None,
    bootstrap: bool = True,
    ci_seed: int = 0,
    meta: dict | None = None,
) -> list[dict]:
    """Run the policy at each threshold; one row per frontier point."""
    rows: list[dict] = []
    for threshold_ms in thresholds_ms:
        policy = policy_factory(threshold_ms)
        result = run_shard(
            policy,
            shard_root,
            provide_agent_stem=provide_agent_stem,
            max_samples=max_samples,
            ignore_dmel_cache=ignore_dmel_cache,
            stop_decision_beta=stop_decision_beta,
            cancel_path_constants=cancel_path_constants,
            cancel_path_source=cancel_path_source,
            cancel_path_retrieved=cancel_path_retrieved,
            bootstrap=bootstrap,
            ci_seed=ci_seed,
            meta={**(meta or {}), "threshold_ms": threshold_ms},
        )
        rows.append({"threshold_ms": threshold_ms, "metrics": result.metrics})
    return rows


def select_operating_point(rows: list[dict], beta: float = DEFAULT_STOP_DECISION_BETA) -> dict | None:
    """The pre-registered pick: max F-beta on stop decisions (ties -> lowest threshold).

    Returns None when no row has a computable F-beta (e.g. no ground-truth
    action column or no predicted stops anywhere on the frontier).
    """

    def fbeta(row: dict) -> float:
        sd = row["metrics"].get("stop_decision") or {}
        value = sd.get("fbeta")
        return float("-inf") if value is None else float(value)

    candidates = [row for row in rows if fbeta(row) != float("-inf")]
    if not candidates:
        return None
    return max(candidates, key=lambda row: (fbeta(row), -row["threshold_ms"]))


def render_sweep_table(rows: list[dict], beta: float = DEFAULT_STOP_DECISION_BETA) -> str:
    """Markdown frontier table plus the pre-registered operating point."""
    header = (
        "| Threshold (ms) | Recall | Premature (overlap) | Premature (non-overlap) | Never-stopped "
        "| Decision latency med (ms) | P95 (ms) | Frame precision | Frame recall | F-beta(beta={:.0f}) | False stops |".format(beta)
    )
    divider = "|" + "---|" * 11
    lines = [header, divider]
    for row in rows:
        m = row["metrics"]
        interruptions = m.get("interruptions", {})
        latency = interruptions.get("decision_latency_ms") or {}
        sd = m.get("stop_decision") or {}

        def fmt(value, nd=3):
            return "\u2014" if value is None else f"{value:.{nd}f}"

        lines.append(
            f"| {row['threshold_ms']:g} "
            f"| {fmt(interruptions.get('recall'))} "
            f"| {fmt(interruptions.get('premature_during_overlap_rate'))} "
            f"| {fmt(interruptions.get('premature_during_non_overlap_rate'))} "
            f"| {fmt(interruptions.get('never_stopped_rate'))} "
            f"| {fmt(latency.get('median'), 1)} | {fmt(latency.get('p95'), 1)} "
            f"| {fmt(sd.get('precision'))} | {fmt(sd.get('recall'))} | {fmt(sd.get('fbeta'))} "
            f"| {m.get('totals', {}).get('false_stops_total', 0)} |"
        )
    selected = select_operating_point(rows, beta)
    lines.append("")
    if selected is None:
        lines.append(
            "No operating point selectable: no frontier row has a computable F-beta on stop "
            "decisions (check that the labels carry the ground-truth `action` column and that the "
            "policy emits at least one STOP on the sweep)."
        )
    else:
        lines.append(
            f"Pre-registered operating point (max F-beta beta={beta:g} on stop decisions, validation "
            f"split): threshold = {selected['threshold_ms']:g} ms."
        )
    return "\n".join(lines)
