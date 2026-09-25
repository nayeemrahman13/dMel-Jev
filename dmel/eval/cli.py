"""CLI for the dMel eval harness: ``python -m dmel.eval <run|sweep|report>``.

run    — stream a contract BargeInPolicy (``module:ClassName``) over a shard dir,
         emit metrics json + optional decision-log jsonl.
sweep  — threshold frontier for a ``module:ClassFactory`` taking the threshold;
         renders the frontier table with the pre-registered F-beta(2) point.
report — combine per-arm metrics json files (one or many seeds per arm) into the
         markdown comparison report. Also re-scores from decision logs when
         paired with --labels-dir (see run's --out-decisions counterpart).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

CANCEL_PATH_KEYS = ("vad_ms", "cancel_send_ms", "audio_stop_ms")


def _parse_cancel_constants(spec: str) -> tuple[dict[str, float], str, str]:
    """Parse ``vad_ms=3.2,cancel_send_ms=1.1,audio_stop_ms=12.0`` or a json file path.

    Returns (constants, source, retrieved); retrieved defaults to today's date
    for CLI-supplied values (the caller citing them measured them now).
    """
    import datetime

    if spec.endswith(".json") and Path(spec).is_file():
        data = json.loads(Path(spec).read_text(encoding="utf-8"))
        source = spec
        retrieved = str(data.pop("retrieved", datetime.date.today().isoformat()))
        source = str(data.pop("source", source))
        data = data.get("cancel_path_constants", data)
    else:
        data = {}
        for part in spec.split(","):
            key, sep, value = part.partition("=")
            if not sep:
                raise SystemExit(f"--cancel-path-constants: expected key=value, got {part!r}")
            data[key.strip()] = float(value)
        source = "supplied via CLI --cancel-path-constants"
        retrieved = datetime.date.today().isoformat()
    missing = [k for k in CANCEL_PATH_KEYS if k not in data]
    if missing:
        raise SystemExit(
            f"--cancel-path-constants: missing {missing}; all of {CANCEL_PATH_KEYS} are required "
            "(measured values only — never estimates)"
        )
    extra = [k for k in data if k not in CANCEL_PATH_KEYS]
    if extra:
        raise SystemExit(f"--cancel-path-constants: unexpected keys {extra}")
    constants = {k: float(data[k]) for k in CANCEL_PATH_KEYS}
    return constants, source, retrieved


def _metrics_from_decision_log(log_path: str, labels_dir: str, **kwargs) -> dict:
    """Recompute metrics from a decision jsonl against a shard/fixture dir."""
    from dmel.eval.decisions import stops_from_log
    from dmel.eval.labels import discover_samples, load_labels
    from dmel.eval.metrics import compute_metrics

    stops = stops_from_log(log_path)
    labels_by_sample = {}
    for sample in discover_samples(labels_dir):
        if sample.sample_id in stops:
            labels_by_sample[sample.sample_id] = load_labels(sample.labels)
    unknown = sorted(set(stops) - set(labels_by_sample))
    if unknown:
        print(f"warning: decision log references unknown samples (skipped): {unknown}", file=sys.stderr)
    if not labels_by_sample:
        raise SystemExit("no decision-log samples matched the labels dir; nothing to score")
    return compute_metrics(labels_by_sample, stops, **kwargs)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m dmel.eval",
        description="dMel barge-in POC evaluation harness (contract v1)",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--labels-dir", help="shard or fixtures directory (required for run/sweep, and for report --recompute-from-log)")
    common.add_argument("--out-json", help="write metrics json here")
    common.add_argument("--stop-decision-beta", type=float, default=2.0)
    common.add_argument("--no-bootstrap", action="store_true", help="skip bootstrap CIs")
    common.add_argument("--ci-seed", type=int, default=0)
    common.add_argument(
        "--cancel-path-constants",
        help="measured 'vad_ms=...,cancel_send_ms=...,audio_stop_ms=...' or a json file; enables the E2E latency column",
    )

    run_p = sub.add_parser("run", parents=[common], help="stream a policy over a shard")
    run_p.add_argument("--policy", required=True, help="policy spec 'module:ClassName'")
    run_p.add_argument("--policy-kwargs", default="{}", help="JSON kwargs for the policy constructor")
    run_p.add_argument("--out-decisions", help="write decision-log jsonl here")
    run_p.add_argument("--no-agent-stem", action="store_true", help="pass agent_pcm_frame=None")
    run_p.add_argument("--max-samples", type=int, default=None)
    run_p.add_argument("--log-probs", action="store_true", help="include policy probs in decision log")
    run_p.add_argument(
        "--ignore-dmel-cache",
        action="store_true",
        help="contract flag-only path: policies receive dmel_token_ids=None",
    )

    sweep_p = sub.add_parser("sweep", parents=[common], help="threshold frontier for a policy factory")
    sweep_p.add_argument("--policy-factory", required=True, help="factory spec 'module:ClassName' called as factory(threshold_ms)")
    sweep_p.add_argument(
        "--thresholds",
        default="100,150,200,300,400",
        help="comma-separated thresholds in ms (default matches the contract's arm-A sweep)",
    )
    sweep_p.add_argument("--no-agent-stem", action="store_true")
    sweep_p.add_argument("--max-samples", type=int, default=None)
    sweep_p.add_argument("--ignore-dmel-cache", action="store_true")

    report_p = sub.add_parser("report", parents=[common], help="markdown A-E comparison report from metrics files")
    report_p.add_argument(
        "--metrics",
        action="append",
        required=True,
        metavar="ARM=FILE",
        help="arm letter to metrics json; repeat for seeds (e.g. --metrics B=s1.json --metrics B=s2.json)",
    )
    report_p.add_argument("--out-md", help="write markdown here (default: stdout)")
    report_p.add_argument(
        "--recompute-from-log",
        help="decision-log jsonl; requires --labels-dir to recompute metrics instead of reading metrics files",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.command in ("run", "sweep") and not args.labels_dir:
        raise SystemExit("--labels-dir is required for run/sweep")
    if (
        args.command == "report"
        and getattr(args, "recompute_from_log", None)
        and not args.labels_dir
    ):
        raise SystemExit("--labels-dir is required with --recompute-from-log")

    cancel_kwargs: dict = {}
    constants_spec = getattr(args, "cancel_path_constants", None)
    if constants_spec:
        constants, source, retrieved = _parse_cancel_constants(constants_spec)
        cancel_kwargs = {
            "cancel_path_constants": constants,
            "cancel_path_source": source,
            "cancel_path_retrieved": retrieved,
        }
    bootstrap_kwargs = {
        "bootstrap": not getattr(args, "no_bootstrap", False),
        "ci_seed": getattr(args, "ci_seed", 0),
    }

    if args.command == "report":
        from dmel.eval.report import render_report

        if getattr(args, "recompute_from_log", None):
            arm, _, _ = _split_arm_spec(args.metrics[0]) if args.metrics else ("A", None, None)
            metrics = _metrics_from_decision_log(args.recompute_from_log, args.labels_dir, **cancel_kwargs, **bootstrap_kwargs)
            metrics_by_arm = {arm: metrics}
        else:
            metrics_by_arm: dict[str, dict | list[dict]] = {}
            for spec in args.metrics:
                arm, path, err = _split_arm_spec(spec)
                if err:
                    raise SystemExit(err)
                with open(path, encoding="utf-8") as f:
                    metrics_by_arm.setdefault(arm, []).append(json.load(f))
            metrics_by_arm = {arm: (v[0] if len(v) == 1 else v) for arm, v in metrics_by_arm.items()}
        markdown = render_report(metrics_by_arm)
        if args.out_md:
            out = Path(args.out_md)
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(markdown, encoding="utf-8")
            print(f"wrote {out}")
        else:
            print(markdown)
        return 0

    if args.command == "run":
        from dmel.eval.runner import load_policy, persist_run, run_shard

        try:
            policy = load_policy(args.policy, json.loads(args.policy_kwargs))
        except (ImportError, AttributeError, ValueError, json.JSONDecodeError) as exc:
            raise SystemExit(f"--policy: {exc}")
        result = run_shard(
            policy,
            args.labels_dir,
            provide_agent_stem=not args.no_agent_stem,
            max_samples=args.max_samples,
            log_probs=args.log_probs,
            ignore_dmel_cache=args.ignore_dmel_cache,
            stop_decision_beta=args.stop_decision_beta,
            **cancel_kwargs,
            **bootstrap_kwargs,
        )
        persist_run(result, out_json=args.out_json, out_decisions=args.out_decisions)
        if args.out_json:
            print(f"wrote {args.out_json}")
        if args.out_decisions:
            print(f"wrote {args.out_decisions}")
        for warning in result.stream.warnings:
            print(f"warning: {warning}", file=sys.stderr)
        for skipped in result.stream.skipped_samples:
            print(f"skipped {skipped['sample_id']}: {skipped['reason']}", file=sys.stderr)
        return 0

    if args.command == "sweep":
        from dmel.eval.runner import load_policy_factory
        from dmel.eval.sweep import render_sweep_table, sweep_thresholds

        try:
            factory = load_policy_factory(args.policy_factory)
        except (ImportError, AttributeError, ValueError) as exc:
            raise SystemExit(f"--policy-factory: {exc}")
        thresholds = [float(x) for x in args.thresholds.split(",") if x.strip()]
        rows = sweep_thresholds(
            factory,
            args.labels_dir,
            thresholds,
            provide_agent_stem=not args.no_agent_stem,
            max_samples=args.max_samples,
            ignore_dmel_cache=args.ignore_dmel_cache,
            stop_decision_beta=args.stop_decision_beta,
            **cancel_kwargs,
            **bootstrap_kwargs,
        )
        output = render_sweep_table(rows, args.stop_decision_beta)
        if args.out_json:
            out = Path(args.out_json)
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps(rows, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
            print(f"wrote {out}")
        print(output)
        return 0

    raise SystemExit(f"unreachable command {args.command!r}")


def _split_arm_spec(spec: str) -> tuple[str, str, str | None]:
    arm, sep, path = spec.partition("=")
    if not sep or not arm.strip() or not path.strip():
        return "", "", f"--metrics must look like ARM=FILE, got {spec!r}"
    return arm.strip().upper(), path.strip(), None


if __name__ == "__main__":
    raise SystemExit(main())
