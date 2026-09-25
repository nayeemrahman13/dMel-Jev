"""Calibration — eval step zero (contract v1): Silero vs the acoustic reference.

Runs the vendored Silero VAD over a shard's mixes and scores speech detection
per scenario against the acoustic reference: the ``speech_present`` label
column, which in v1 counts speech from ANY non-agent speaker (user or
background). A low Silero probability on espeak-style synthesis means the
A-vs-B/C comparison would be a corpus artifact, so this report must exist
before any arm comparison is trusted.

If user-speech detection recall < 0.9, the report flags that a kokoro-voiced
calibration slice must be added to the corpus before the comparison is
trusted (the kokoro TTS adapter lives in the upstream runtime repo, not
vendored here), and the flag travels with the numbers.

Usage:

    python -m dmel.baselines.calibrate --shard data/pilot/shard-00001 \
        [--prob-threshold 0.5] [--out calib.json]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from dmel.baselines import shardio
from dmel.baselines.metrics import f_beta, precision_recall
from dmel.runtime.silero import SileroVAD

RECALL_FLOOR = 0.9
FRAME_SAMPLES = 800


def score_sample(mix: np.ndarray, speech_present: list[bool], vad: SileroVAD, *, prob_threshold: float) -> dict[str, int]:
    """Per-frame Silero speech prediction vs the acoustic reference."""
    counts = {"tp": 0, "fp": 0, "fn": 0, "tn": 0}
    frame_count = min(mix.size // FRAME_SAMPLES, len(speech_present))
    vad.reset()
    for index in range(frame_count):
        frame = mix[index * FRAME_SAMPLES : (index + 1) * FRAME_SAMPLES].astype(np.float32) / 32768.0
        vad.push(frame, float(index * 50), float(index * 50))
        predicted = vad.last_prob >= prob_threshold
        reference = bool(speech_present[index])
        key = ("tp" if predicted else "fn") if reference else ("fp" if predicted else "tn")
        counts[key] += 1
    return counts


def merge_counts(target: dict[str, int], source: dict[str, int]) -> None:
    for key, value in source.items():
        target[key] += value


def calibrate_shard(shard: Path, *, prob_threshold: float, samples: list[str] | None = None) -> dict:
    if not shard.is_dir():
        raise FileNotFoundError(f"shard directory not found: {shard}")

    mix_paths = (
        [shard / f"{s}.mix.pcm" for s in samples]
        if samples
        else sorted(shard.glob("*.mix.pcm"))
    )
    if not mix_paths:
        raise FileNotFoundError(f"no <sample>.mix.pcm files under {shard}")

    vad = SileroVAD.load()
    per_scenario: dict[str, dict[str, int]] = {}
    for mix_path in mix_paths:
        sample_id = shardio.sample_id_of(mix_path)
        mix = shardio.load_mix(mix_path)
        table = shardio.load_labels_table(shard / f"{sample_id}.labels.parquet")
        speech_present = [bool(v) for v in shardio.labels_column(table, "speech_present")]
        scenarios = shardio.labels_column(table, "scenario")
        if len(scenarios) < len(speech_present):
            raise ValueError(
                f"{sample_id}: scenario column shorter than speech_present — "
                "the scenario tag is required for per-scenario calibration"
            )
        counts = score_sample(mix, speech_present, vad, prob_threshold=prob_threshold)
        for scenario in set(scenarios[: len(speech_present)]):
            merge_counts(
                per_scenario.setdefault(str(scenario), {"tp": 0, "fp": 0, "fn": 0, "tn": 0}),
                counts,
            )

    report: dict = {"prob_threshold": prob_threshold, "scenarios": {}, "overall": {}}
    total = {"tp": 0, "fp": 0, "fn": 0, "tn": 0}
    for scenario in sorted(per_scenario):
        counts = per_scenario[scenario]
        precision, recall = precision_recall(counts["tp"], counts["fp"], counts["fn"])
        report["scenarios"][scenario] = {
            **counts,
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f_beta(precision, recall, beta=1.0), 4),
        }
        merge_counts(total, counts)
    precision, recall = precision_recall(total["tp"], total["fp"], total["fn"])
    report["overall"] = {
        **total,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f_beta(precision, recall, beta=1.0), 4),
    }
    report["user_speech_recall"] = report["overall"]["recall"]
    report["kokoro_slice_required"] = report["user_speech_recall"] < RECALL_FLOOR
    if report["kokoro_slice_required"]:
        print(
            f"user-speech detection recall {report['user_speech_recall']:.3f} < {RECALL_FLOOR}: "
            "add a kokoro-voiced calibration slice to the corpus before trusting "
            "the A-vs-B/C comparison (kokoro TTS is vendored in the runtime)",
            file=sys.stderr,
        )
    return report


def markdown_table(report: dict) -> str:
    lines = [
        "| scenario | precision | recall | f1 | tp | fp | fn | tn |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for scenario, row in report["scenarios"].items():
        lines.append(
            f"| {scenario} | {row['precision']} | {row['recall']} | {row['f1']} "
            f"| {row['tp']} | {row['fp']} | {row['fn']} | {row['tn']} |"
        )
    overall = report["overall"]
    lines.append(
        f"| **overall** | {overall['precision']} | {overall['recall']} | {overall['f1']} "
        f"| {overall['tp']} | {overall['fp']} | {overall['fn']} | {overall['tn']} |"
    )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--shard", type=Path, required=True)
    parser.add_argument("--sample", action="append", default=None, help="sample id (repeatable)")
    parser.add_argument("--prob-threshold", type=float, default=0.5)
    parser.add_argument("--out", type=Path, default=None, help="write json report here (also prints markdown)")
    args = parser.parse_args(argv)

    report = calibrate_shard(args.shard, prob_threshold=args.prob_threshold, samples=args.sample)
    print(markdown_table(report))
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
