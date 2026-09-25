"""Offline arm-A runner: stream contract-shard mixes through the policy.

Reads a contract shard's ``<sample>.mix.pcm`` (16 kHz mono int16) plus the
``agent_speaking`` column of ``<sample>.labels.parquet`` and writes one jsonl
decision log per sample:

    <out>/<sample>.armA.jsonl   one line per 50 ms frame:
        {"t_ms", "action", "p_stop", "speech_prob", "speech_ms", "latched",
         "stop_threshold_ms"}

Usage:

    python -m dmel.baselines.runner --shard data/pilot/shard-00001 \
        [--sample ID ...] [--threshold-ms 200] [--out DIR]

Requires the repo root on ``PYTHONPATH`` (pytest runs put it there via the
root config or the conftest bootstrap); ``pyarrow`` for the label column
(imported lazily so policy-only consumers do not need it).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from dmel.baselines import shardio

LOG_KEYS = ("t_ms", "action", "p_stop", "speech_prob", "speech_ms", "latched", "stop_threshold_ms")


def run_sample(mix_pcm_path: Path, labels_path: Path, out_path: Path, *, threshold_ms: float) -> dict:
    """Stream one sample through the policy and write its decision log."""
    mix = shardio.load_mix(mix_pcm_path)
    agent_speaking = shardio.agent_speaking_series(shardio.load_labels_table(labels_path))

    frame_count = min(mix.size // 800, len(agent_speaking))
    if mix.size // 800 < len(agent_speaking):
        print(
            f"warning: {mix_pcm_path.name} holds {mix.size // 800} frames "
            f"but labels cover {len(agent_speaking)}; logging the first {frame_count}",
            file=sys.stderr,
        )

    stops = 0
    with out_path.open("w", encoding="utf-8") as handle:
        for record in shardio.iter_decisions(mix, agent_speaking, threshold_ms=threshold_ms):
            handle.write(json.dumps({key: record[key] for key in LOG_KEYS}) + "\n")
            stops += record["action"] == "STOP_TTS"

    summary = {"sample": shardio.sample_id_of(mix_pcm_path), "frames": frame_count, "stop_frames": stops}
    print(json.dumps(summary))
    return summary


def run_shard(
    shard: Path, out_dir: Path, *, threshold_ms: float, samples: list[str] | None = None
) -> list[Path]:
    """Run every (or the named) sample in a shard; return written log paths."""
    if not shard.is_dir():
        raise FileNotFoundError(f"shard directory not found: {shard}")

    if samples:
        mix_paths = [shard / f"{sample}.mix.pcm" for sample in samples]
    else:
        mix_paths = sorted(shard.glob("*.mix.pcm"))
    if not mix_paths:
        raise FileNotFoundError(f"no <sample>.mix.pcm files under {shard}")

    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for mix_path in mix_paths:
        sample_id = shardio.sample_id_of(mix_path)
        labels_path = shard / f"{sample_id}.labels.parquet"
        if not labels_path.is_file():
            raise FileNotFoundError(
                f"{labels_path} not found — per-frame agent_speaking labels are required"
            )
        out_path = out_dir / f"{sample_id}.armA.jsonl"
        run_sample(mix_path, labels_path, out_path, threshold_ms=threshold_ms)
        written.append(out_path)
    return written


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--shard", type=Path, required=True, help="shard directory to process")
    parser.add_argument(
        "--sample",
        action="append",
        default=None,
        help="sample id to process (repeatable); default: every *.mix.pcm in the shard",
    )
    parser.add_argument("--threshold-ms", type=float, default=200.0, help="stop duration threshold (ms)")
    parser.add_argument("--out", type=Path, default=None, help="output dir (default: <shard>/decisions)")
    args = parser.parse_args(argv)

    out_dir = args.out if args.out is not None else args.shard / "decisions"
    run_shard(args.shard, out_dir, threshold_ms=args.threshold_ms, samples=args.sample)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
