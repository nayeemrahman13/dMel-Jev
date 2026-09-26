"""Capacity sweep evaluation + aggregation driver (validation split only).

Per (width, seed) checkpoint from the Modal training runs:
  1. sweep the policy's stop_threshold (a probability, not ms) over the
     validation split through the eval harness (`dmel.eval.sweep`);
  2. select ONE operating point by max F-beta (beta=2) on stop decisions —
     the harness's own `select_operating_point` rule (ties -> lowest);
  3. re-run at the selected point with bootstrap CIs and write the metrics
     json + decision-log jsonl per existing run conventions.

Then `--aggregate` pools the per-run metrics into per-width mean +- sd and
emits the markdown comparison fragment for docs/capacity_sweep.md.

The learned policy is only exposed to the harness through the contract
interface (`dmel.models.policy`), as for any other arm.

Usage (from the repo root, after pulling runs from the dmel-jev-runs volume):

    python -m dmel.training.capacity_sweep \
        --data-root data/pilot --split val \
        --runs-root runs/capacity_sweep \
        --widths 256 --seeds 0 1 2 \
        --thresholds 0.30,0.35,0.40,0.45,0.50,0.55,0.60,0.65,0.70 \
        --out-dir runs/capacity_sweep/eval

    python -m dmel.training.capacity_sweep --aggregate \
        --runs-root runs/capacity_sweep --widths 256 384 512 --seeds 0 1 2 \
        --out-dir runs/capacity_sweep/eval
"""

from __future__ import annotations

import argparse
import json
import shutil
import statistics
from dataclasses import replace
from pathlib import Path

import torch

from dmel.eval.runner import persist_run, run_shard
from dmel.eval.sweep import select_operating_point, sweep_thresholds
from dmel.models.model import load_model_from_checkpoint
from dmel.models.policy import CheckpointedBargeInPolicy

DEFAULT_THRESHOLDS = "0.30,0.35,0.40,0.45,0.50,0.55,0.60,0.65,0.70"


def materialize_split_dir(data_root: Path, split: str, target: Path) -> Path:
    """Symlink one split's samples into ``target`` so the eval harness (which
    discovers samples by directory walk) scores exactly that split.

    The manifest's ``split`` column stays the single source of truth; the
    tree contains only links, never copies, so the shard bytes cannot drift
    from the generated corpus.
    """
    from dmel.training.dataset import scan_samples

    if target.exists():
        shutil.rmtree(target)
    target.mkdir(parents=True)
    counts = {split: 0, "skipped": 0}
    seen: set[Path] = set()
    for sample in scan_samples(data_root):
        if sample.split != split:
            counts["skipped"] += 1
            continue
        shard_dir = sample.path.parent
        dest = target / shard_dir.name
        dest.mkdir(exist_ok=True)
        stem = sample.sample_id
        for suffix in (".labels.parquet", ".user.pcm", ".agent.pcm", ".dmel.npy"):
            source = shard_dir / f"{stem}{suffix}"
            if source.exists() and source not in seen:
                (dest / source.name).symlink_to(source.resolve())
                seen.add(source)
        counts[split] += 1
    if counts[split] == 0:
        raise ValueError(f"{data_root}: no samples assigned to split {split!r}")
    return target


def checkpoint_for_run(runs_root: Path, width: int, seed: int) -> Path:
    """Best checkpoint of one (width, seed) run, per the training layout
    ``<runs_root>/w<W>/seed_<S>/<arm>/seed_<S>/best.pt``."""
    path = runs_root / f"w{width}" / f"seed_{seed}" / "transformer" / f"seed_{seed}" / "best.pt"
    if not path.is_file():
        raise FileNotFoundError(f"checkpoint not found: {path}")
    return path


def evaluate_checkpoint(
    checkpoint: Path,
    labels_dir: str,
    thresholds: list[float],
    out_dir: Path,
    *,
    stop_decision_beta: float = 2.0,
) -> dict:
    """Harness threshold sweep + pre-registered operating-point run."""
    out_dir.mkdir(parents=True, exist_ok=True)

    def factory(threshold: float) -> CheckpointedBargeInPolicy:
        policy = CheckpointedBargeInPolicy.from_checkpoint(checkpoint)
        config = replace(
            policy.config,
            policy=replace(policy.config.policy, stop_threshold=float(threshold)),
        )
        return CheckpointedBargeInPolicy(policy.model, config)

    torch.set_num_threads(1)  # driver instances run in parallel; stay single-core
    rows = sweep_thresholds(
        factory,
        labels_dir,
        thresholds,
        meta={"checkpoint": str(checkpoint)},
        stop_decision_beta=stop_decision_beta,
    )
    (out_dir / "sweep.json").write_text(json.dumps(rows, indent=2, ensure_ascii=False))

    selected = select_operating_point(rows, beta=stop_decision_beta)
    if selected is None:
        degenerate = write_degenerate_metrics(
            rows, out_dir, checkpoint, stop_decision_beta=stop_decision_beta
        )
        if degenerate is None:
            raise ValueError(
                f"{checkpoint}: no frontier row has a computable F-beta, but the "
                "frontier is not a never-stop case — refusing to pick an operating point"
            )
        return degenerate
    threshold = selected["threshold_ms"]  # sweep keys are generic; here: stop probability

    model, _ = load_model_from_checkpoint(checkpoint)
    result = run_shard(
        factory(threshold),
        labels_dir,
        stop_decision_beta=stop_decision_beta,
        meta={
            "checkpoint": str(checkpoint),
            "selected_stop_threshold": threshold,
            "selection": f"max F-beta (beta={stop_decision_beta}) on val stop decisions; ties -> lowest threshold",
            "params": model.num_parameters(),
        },
    )
    persist_run(
        result,
        out_json=str(out_dir / "metrics.json"),
        out_decisions=str(out_dir / "decisions.jsonl"),
    )
    return {
        "checkpoint": str(checkpoint),
        "params": model.num_parameters(),
        "selected_stop_threshold": threshold,
        "metrics": result.metrics,
    }


def write_degenerate_metrics(
    rows: list[dict],
    out_dir: Path,
    checkpoint: Path,
    *,
    stop_decision_beta: float = 2.0,
) -> dict | None:
    """Materialize metrics for a never-stop checkpoint from its sweep rows.

    select_operating_point returns None only when no row has a computable
    F-beta; with zero predicted stops every row is identical, so the row
    metrics ARE the run metrics — copied verbatim, not recomputed. Returns
    None (writes nothing) when the frontier is not a never-stop case.
    """
    if any(r["metrics"]["totals"]["predicted_stop_frames"] != 0 for r in rows):
        return None
    model, _ = load_model_from_checkpoint(checkpoint)
    metrics = dict(rows[0]["metrics"])
    metrics["meta"] = {
        "checkpoint": str(checkpoint),
        "selected_stop_threshold": None,
        "selection": (
            f"no operating point: zero predicted stop frames at every sweep "
            f"threshold; stop-decision F-beta (beta={stop_decision_beta}) undefined"
        ),
        "params": model.num_parameters(),
        "no_operating_point": True,
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False))
    return {
        "checkpoint": str(checkpoint),
        "params": model.num_parameters(),
        "selected_stop_threshold": None,
        "metrics": metrics,
    }


def _mean_sd(values: list[float | None]) -> dict:
    finite = [v for v in values if v is not None]
    if not finite:
        return {"mean": None, "sd": None, "n": 0}
    return {
        "mean": statistics.fmean(finite),
        "sd": statistics.stdev(finite) if len(finite) > 1 else 0.0,
        "n": len(finite),
    }


def _rate(metrics: dict, *path: str) -> float | None:
    node: object = metrics
    for key in path:
        if not isinstance(node, dict) or key not in node:
            return None
        node = node[key]
    return node if isinstance(node, (int, float)) else None


def aggregate(widths: list[int], seeds: list[int], out_dir: Path) -> dict:
    """Per-width mean +- sd over seeds, from the per-run metrics.json files."""
    per_run: dict[int, dict[int, dict]] = {}
    for width in widths:
        per_run[width] = {}
        for seed in seeds:
            path = out_dir / f"w{width}_seed{seed}" / "metrics.json"
            # metrics.json is the flat harness metrics dict; run_shard merges the
            # run meta (checkpoint, params, selected_stop_threshold) into .meta.
            per_run[width][seed] = json.loads(path.read_text())

    widths_summary: dict[int, dict] = {}
    for width in widths:
        runs = per_run[width]
        metrics = list(runs.values())
        params = sorted({run["meta"]["params"] for run in runs.values()})
        if len(params) != 1:
            raise ValueError(f"w{width}: inconsistent param counts across seeds: {params}")
        widths_summary[width] = {
            "params": params[0],
            "seeds": sorted(runs),
            "selected_stop_thresholds": sorted(
                {
                    t
                    for run in runs.values()
                    if (t := run["meta"]["selected_stop_threshold"]) is not None
                }
            ),
            "no_operating_point_seeds": sum(
                1
                for run in runs.values()
                if run["meta"]["selected_stop_threshold"] is None
            ),
            "interruption_recall": _mean_sd(
                [_rate(m, "interruptions", "recall") for m in metrics]
            ),
            "never_stopped_rate": _mean_sd(
                [_rate(m, "interruptions", "never_stopped_rate") for m in metrics]
            ),
            "premature_during_overlap_rate": _mean_sd(
                [_rate(m, "interruptions", "premature_during_overlap_rate") for m in metrics]
            ),
            "premature_during_non_overlap_rate": _mean_sd(
                [_rate(m, "interruptions", "premature_during_non_overlap_rate") for m in metrics]
            ),
            "decision_latency_median_ms": _mean_sd(
                [_rate(m, "interruptions", "decision_latency_ms", "median") for m in metrics]
            ),
            "decision_latency_p95_ms": _mean_sd(
                [_rate(m, "interruptions", "decision_latency_ms", "p95") for m in metrics]
            ),
            # Censoring per the harness: never-stopped events are censored at
            # infinity, and a quantile anchored in that tail is None, not a
            # finite number. Surface the counts so dashes are explainable.
            "decision_latency_n_handled": _mean_sd(
                [_rate(m, "interruptions", "decision_latency_ms", "n_handled") for m in metrics]
            ),
            "decision_latency_n_censored": _mean_sd(
                [_rate(m, "interruptions", "decision_latency_ms", "n_censored") for m in metrics]
            ),
            "decision_latency_median_censored_seeds": sum(
                1
                for m in metrics
                if _rate(m, "interruptions", "decision_latency_ms", "median_censored")
            ),
            "stop_decision_fbeta": _mean_sd(
                [_rate(m, "stop_decision", "fbeta") for m in metrics]
            ),
            "false_stops_total": _mean_sd(
                [_rate(m, "totals", "false_stops_total") for m in metrics]
            ),
            "false_stops_by_scenario": {
                scenario: {
                    key: _mean_sd(
                        [
                            _rate(
                                m,
                                "false_stops_by_scenario",
                                scenario,
                                key,
                            )
                            for m in metrics
                        ]
                    )
                    for key in ("false_stop_rate", "segments", "segments_with_false_stop", "false_stop_frames")
                }
                for scenario in ("backchannel", "hesitation", "hesitation_commit", "noise", "background_speaker")
            },
            "stops_while_agent_silent": _mean_sd(
                [_rate(m, "deployment_hazards", "stops_while_agent_silent", "count") for m in metrics]
            ),
        }

    report = {
        "widths": {str(w): widths_summary[w] for w in widths},
        "per_run": {
            str(w): {
                str(s): {
                    "params": per_run[w][s]["meta"]["params"],
                    "selected_stop_threshold": per_run[w][s]["meta"]["selected_stop_threshold"],
                    "interruption_recall": _rate(per_run[w][s], "interruptions", "recall"),
                    "stop_decision_fbeta": _rate(per_run[w][s], "stop_decision", "fbeta"),
                    "decision_latency_median_ms": _rate(
                        per_run[w][s], "interruptions", "decision_latency_ms", "median"
                    ),
                    "decision_latency_p95_ms": _rate(
                        per_run[w][s], "interruptions", "decision_latency_ms", "p95"
                    ),
                }
                for s in seeds
            }
            for w in widths
        },
    }
    (out_dir / "aggregate.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))

    def cell(stat: dict, nd: int = 3) -> str:
        if stat["mean"] is None:
            return "—"
        return f"{stat['mean']:.{nd}f} ± {stat['sd']:.{nd}f}"

    lines = [
        "| Metric (validation, mean ± sd over seeds) | "
        + " | ".join(f"6×{w}" for w in widths)
        + " |",
        "|" + "---|" * (len(widths) + 1),
        "| Parameters | "
        + " | ".join(f"{widths_summary[w]['params']:,}" for w in widths)
        + " |",
    ]
    rows = [
        ("First in-window STOP recall", "interruption_recall", 3),
        ("Never stopped (censored rate)", "never_stopped_rate", 3),
        ("Premature stop — during overlap", "premature_during_overlap_rate", 3),
        ("Premature stop — during non-overlap", "premature_during_non_overlap_rate", 3),
        ("Hesitation-commit false stops (segment rate)", ("false_stops_by_scenario", "hesitation_commit", "false_stop_rate"), 3),
        ("Noise false stops (segment rate)", ("false_stops_by_scenario", "noise", "false_stop_rate"), 3),
        ("Background-speech false stops (segment rate)", ("false_stops_by_scenario", "background_speaker", "false_stop_rate"), 3),
        ("Backchannel false stops (segment rate)", ("false_stops_by_scenario", "backchannel", "false_stop_rate"), 3),
        ("Total false stops (frames)", "false_stops_total", 1),
        ("Decision latency median (ms)", "decision_latency_median_ms", 1),
        ("Decision latency p95 (ms)", "decision_latency_p95_ms", 1),
        ("Stop-decision F-beta (β=2)", "stop_decision_fbeta", 3),
    ]
    for label, key, nd in rows:
        cells = []
        for width in widths:
            stat = widths_summary[width][key] if not isinstance(key, tuple) else widths_summary[width]
            if isinstance(key, tuple):
                for part in key:
                    stat = stat[part]
            cells.append(cell(stat, nd))
        lines.append(f"| {label} | " + " | ".join(cells) + " |")
    (out_dir / "aggregate_table.md").write_text("\n".join(lines) + "\n")
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-root", default="data/pilot")
    parser.add_argument("--split", default="val")
    parser.add_argument("--runs-root", default="runs/capacity_sweep")
    parser.add_argument("--widths", type=int, nargs="+", default=[256, 384, 512])
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument(
        "--thresholds",
        default=DEFAULT_THRESHOLDS,
        help="stop_threshold probability grid (sweep mode only)",
    )
    parser.add_argument("--out-dir", default="runs/capacity_sweep/eval")
    parser.add_argument("--aggregate", action="store_true", help="pool metrics.json into mean ± sd")
    parser.add_argument(
        "--rows-to-degenerate-metrics",
        type=Path,
        metavar="RUN_DIR",
        help="materialize a never-stop run's metrics.json from its persisted "
        "sweep.json (for runs that failed before the degenerate path existed)",
    )
    parser.add_argument("--keep-split-dir", action="store_true", help="keep the symlinked split tree")
    parser.add_argument(
        "--reuse-split-dir",
        action="store_true",
        help="skip re-materializing the symlink tree when it already exists — "
        "required when several driver processes evaluate concurrently "
        "(a concurrent rebuild would pull links out from under a running eval)",
    )
    args = parser.parse_args(argv)

    out_dir = Path(args.out_dir)
    if args.rows_to_degenerate_metrics is not None:
        run_dir = args.rows_to_degenerate_metrics
        rows = json.loads((run_dir / "sweep.json").read_text())
        checkpoint = Path(rows[0]["metrics"]["meta"]["checkpoint"])
        degenerate = write_degenerate_metrics(rows, run_dir, checkpoint)
        if degenerate is None:
            print(f"{run_dir}: frontier is not a never-stop case; nothing written")
            return 1
        print(
            f"{run_dir}: wrote degenerate metrics (no operating point, "
            f"params {degenerate['params']:,})"
        )
        return 0

    if args.aggregate:
        aggregate(args.widths, args.seeds, out_dir)
        print(f"wrote {out_dir / 'aggregate.json'} and {out_dir / 'aggregate_table.md'}")
        return 0

    data_root = Path(args.data_root)
    split_dir = data_root.parent / f"{data_root.name}-{args.split}"
    if args.reuse_split_dir and split_dir.is_dir():
        print(f"reusing split dir {split_dir}", flush=True)
    else:
        materialize_split_dir(data_root, args.split, split_dir)
    thresholds = [float(t) for t in args.thresholds.split(",") if t.strip()]

    for width in args.widths:
        for seed in args.seeds:
            checkpoint = checkpoint_for_run(Path(args.runs_root), width, seed)
            run_out = out_dir / f"w{width}_seed{seed}"
            print(f"=== w{width} seed {seed}: {checkpoint} -> {run_out}", flush=True)
            summary = evaluate_checkpoint(checkpoint, str(split_dir), thresholds, run_out)
            print(
                f"=== w{width} seed {seed}: threshold {summary['selected_stop_threshold']}, "
                f"recall {_rate(summary['metrics'], 'interruptions', 'recall')}",
                flush=True,
            )
    if not args.keep_split_dir:
        shutil.rmtree(split_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
