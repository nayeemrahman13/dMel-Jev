"""Stream shard samples through a contract BargeInPolicy and compute metrics.

The runner knows policies only through the contract interface:

    policy.reset()
    policy.step(user_pcm_frame, dmel_token_ids, *, agent_speaking,
                agent_pcm_frame=None, speaker_similarity=None)
        -> {"action": "KEEP"|"STOP_TTS", "probs": dict, "t_ms": int}

It never imports policy implementations itself — the CLI resolves a
``module:ClassName`` spec at runtime, so arms A-E plug in without this area
depending on any other area's code.
"""

from __future__ import annotations

import importlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from dmel.eval.decisions import decision_row, write_decisions
from dmel.eval.labels import SamplePaths, discover_samples, frame_view, load_dmel_steps, load_labels, read_pcm
from dmel.eval.metrics import DEFAULT_STOP_DECISION_BETA, STOP_TTS, compute_metrics

POLICY_SPEC_ERROR = "--policy must look like 'package.module:ClassName'"


@dataclass
class PolicyStream:
    """Raw outputs of streaming a policy over samples."""

    stops_by_sample: dict[str, list[int]]
    decisions: list[dict]
    skipped_samples: list[dict]
    warnings: list[str]
    ran_sample_ids: set[str]


@dataclass
class RunResult:
    """Everything one policy run produced."""

    metrics: dict
    stream: PolicyStream


def _import_policy_spec(spec: str) -> Any:
    module_name, sep, class_name = spec.partition(":")
    if not sep or not module_name or not class_name:
        raise ValueError(f"{POLICY_SPEC_ERROR}, got {spec!r}")
    return getattr(importlib.import_module(module_name), class_name)


def load_policy(spec: str, kwargs: dict[str, Any] | None = None) -> Any:
    """Instantiate a policy from a ``module:ClassName`` spec."""
    return _import_policy_spec(spec)(**(kwargs or {}))


def load_policy_factory(spec: str) -> Any:
    """Load a policy class for threshold sweeps: called once per threshold."""
    factory = _import_policy_spec(spec)
    if not callable(factory):
        raise ValueError(f"policy factory {spec!r} is not callable")
    return factory


def run_policy(
    policy: Any,
    samples: list[SamplePaths],
    *,
    provide_agent_stem: bool = True,
    max_samples: int | None = None,
    log_probs: bool = False,
    ignore_dmel_cache: bool = False,
) -> PolicyStream:
    """Stream every sample through the policy, 50 ms at a time.

    Samples missing user audio are skipped and reported, never silently dropped.
    The labels' ``t_ms`` is authoritative for the log; the policy's own returned
    ``t_ms`` is informational (the interface carries no clock argument — policies
    count steps themselves after ``reset()``).
    """
    stops_by_sample: dict[str, list[int]] = {}
    decision_rows: list[dict] = []
    skipped: list[dict] = []
    warnings: list[str] = []
    ran_sample_ids: set[str] = set()

    selected = samples if max_samples is None else samples[:max_samples]
    for sample in selected:
        if sample.user_pcm is None:
            skipped.append({"sample_id": sample.sample_id, "reason": "missing <sample_id>.user.pcm"})
            continue
        labels = load_labels(sample.labels)
        n_steps = len(labels)
        user_pcm = read_pcm(sample.user_pcm)
        agent_pcm = read_pcm(sample.agent_pcm) if provide_agent_stem and sample.agent_pcm else None
        if ignore_dmel_cache:
            # Contract flag-only path: policies receive dmel_token_ids=None and
            # must degrade to their documented flag-threshold rule.
            dmel_steps: list[np.ndarray | None] = [None] * n_steps
            dmel_warning = "dmel cache ignored (--ignore-dmel-cache); flag-only path"
        else:
            dmel_steps, dmel_warning = load_dmel_steps(sample.dmel_npy, n_steps)
        if dmel_warning:
            warnings.append(f"{sample.sample_id}: {dmel_warning}")

        policy.reset()
        has_t_ms = "t_ms" in labels.columns
        for i in range(n_steps):
            label = labels.iloc[i]
            result = policy.step(
                frame_view(user_pcm, i),
                dmel_steps[i],
                agent_speaking=bool(label["agent_speaking"]),
                agent_pcm_frame=frame_view(agent_pcm, i) if agent_pcm is not None else None,
                speaker_similarity=None,
            )
            if not isinstance(result, dict) or "action" not in result:
                raise TypeError(
                    f"{type(policy).__name__}.step returned {result!r} at frame {i}; "
                    "contract requires a dict with an 'action' key"
                )
            action = str(result["action"]).upper()
            if action not in ("KEEP", "STOP_TTS"):
                raise ValueError(
                    f"{type(policy).__name__}.step returned action {result['action']!r} at frame {i}; "
                    "contract allows only KEEP or STOP_TTS"
                )
            t_ms = int(label["t_ms"]) if has_t_ms else i * 50
            if action == STOP_TTS:
                stops_by_sample.setdefault(sample.sample_id, []).append(t_ms)
            probs = result.get("probs") if log_probs else None
            decision_rows.append(decision_row(sample.sample_id, i, t_ms, action, probs))
        ran_sample_ids.add(sample.sample_id)
    return PolicyStream(
        stops_by_sample=stops_by_sample,
        decisions=decision_rows,
        skipped_samples=skipped,
        warnings=warnings,
        ran_sample_ids=ran_sample_ids,
    )


def run_shard(
    policy: Any,
    shard_root: str,
    *,
    provide_agent_stem: bool = True,
    max_samples: int | None = None,
    log_probs: bool = False,
    ignore_dmel_cache: bool = False,
    stop_decision_beta: float = DEFAULT_STOP_DECISION_BETA,
    cancel_path_constants: dict[str, float] | None = None,
    cancel_path_source: str | None = None,
    cancel_path_retrieved: str | None = None,
    bootstrap: bool = True,
    ci_seed: int = 0,
    meta: dict | None = None,
) -> RunResult:
    """Discover samples under ``shard_root``, run the policy, compute metrics."""
    samples = discover_samples(shard_root)
    stream = run_policy(
        policy,
        samples,
        provide_agent_stem=provide_agent_stem,
        max_samples=max_samples,
        log_probs=log_probs,
        ignore_dmel_cache=ignore_dmel_cache,
    )
    labels_by_sample = {
        sample.sample_id: load_labels(sample.labels)
        for sample in samples
        if sample.sample_id in stream.ran_sample_ids
    }
    metrics = compute_metrics(
        labels_by_sample,
        stream.stops_by_sample,
        stop_decision_beta=stop_decision_beta,
        cancel_path_constants=cancel_path_constants,
        cancel_path_source=cancel_path_source,
        cancel_path_retrieved=cancel_path_retrieved,
        bootstrap=bootstrap,
        ci_seed=ci_seed,
        meta={
            **(meta or {}),
            "shard_dir": str(shard_root),
            "samples_run": len(labels_by_sample),
            "samples_skipped": len(stream.skipped_samples),
            "token_path": "none (flag-only)" if ignore_dmel_cache else "dmel cache where present",
        },
    )
    return RunResult(metrics=metrics, stream=stream)


def persist_run(result: RunResult, *, out_json: str | None = None, out_decisions: str | None = None) -> None:
    """Write a run's metrics json and/or decision log jsonl."""
    if out_decisions:
        write_decisions(out_decisions, result.stream.decisions)
    if out_json:
        path = Path(out_json)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(result.metrics, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
