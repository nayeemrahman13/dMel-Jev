"""Validation-split threshold selection for arm A (contract v1 protocol).

Sweeps the stop duration threshold on the VALIDATION split only (per-sample
``split`` from the shard manifest) and selects ONE operating point by
maximizing F-beta on stop decisions with beta=2 (recall-weighted). The chosen
threshold is the pre-registered operating point: test-split numbers are
reported at it and nowhere else. No eval-set tuning — for anyone.

Usage:

    python -m dmel.baselines.select --shard data/pilot/shard-00001 \
        [--thresholds 50,100,200,300] [--manifest auto] [--out selected.json]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from dmel.baselines import shardio
from dmel.baselines.metrics import f_beta, precision_recall

DEFAULT_GRID_MS = "25,50,75,100,125,150,175,200,250,300,400,500,600"


def manifest_splits(manifest_path: Path) -> dict[str, str]:
    """Extract {sample_id: split} from a shard manifest.

    Tolerates the shapes the contract leaves open: a mapping or list under a
    ``samples`` key, or a direct mapping/list at the top level. Every sample
    entry must carry a ``split`` (the contract makes it mandatory).
    """
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    entries = manifest.get("samples", manifest) if isinstance(manifest, dict) else manifest
    splits: dict[str, str] = {}
    if isinstance(entries, dict):
        items = list(entries.items())
    elif isinstance(entries, list):
        items = [(None, e) for e in entries]
    else:
        raise ValueError(f"{manifest_path}: unrecognized manifest shape ({type(entries).__name__})")
    for key, entry in items:
        if not isinstance(entry, dict) or "split" not in entry:
            raise ValueError(f"{manifest_path}: sample entry missing 'split': {key or entry}")
        sample_id = entry.get("id") or entry.get("sample_id") or key
        if not sample_id:
            raise ValueError(f"{manifest_path}: cannot determine sample id for entry {entry}")
        splits[str(sample_id)] = str(entry["split"])
    if not splits:
        raise ValueError(f"{manifest_path}: manifest lists no samples")
    return splits


def pooled_f_beta(
    shard: Path, sample_ids: list[str], thresholds_ms: list[float]
) -> list[dict]:
    """Sweep thresholds over the given samples; pooled stop-decision metrics."""
    labeled = []
    for sample_id in sample_ids:
        mix_path = shard / f"{sample_id}.mix.pcm"
        labels_path = shard / f"{sample_id}.labels.parquet"
        if not mix_path.is_file() or not labels_path.is_file():
            raise FileNotFoundError(f"{sample_id}: mix+labels parquet required for selection")
        mix = shardio.load_mix(mix_path)
        table = shardio.load_labels_table(labels_path)
        agent_speaking = shardio.agent_speaking_series(table)
        label_action = shardio.labels_column(table, "action")
        labeled.append((sample_id, mix, agent_speaking, [str(a) for a in label_action]))

    label_stops = sum(a == "STOP_TTS" for _, _, _, label_action in labeled for a in label_action)
    if label_stops == 0:
        raise ValueError("no STOP_TTS frames in the selection split labels — F-beta is undefined")

    rows = []
    for threshold_ms in thresholds_ms:
        tp = fp = fn = 0
        for _, mix, agent_speaking, label_action in labeled:
            for record in shardio.iter_decisions(mix, agent_speaking, threshold_ms=threshold_ms):
                predicted = record["action"] == "STOP_TTS"
                reference = label_action[record["t_ms"] // 50] == "STOP_TTS"
                if predicted and reference:
                    tp += 1
                elif predicted:
                    fp += 1
                elif reference:
                    fn += 1
        precision, recall = precision_recall(tp, fp, fn)
        rows.append(
            {
                "threshold_ms": threshold_ms,
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "precision": round(precision, 4),
                "recall": round(recall, 4),
                "f_beta": round(f_beta(precision, recall, beta=2.0), 4),
            }
        )
    return rows


def select_operating_point(
    shard: Path, *, manifest_path: Path | None, thresholds_ms: list[float], split: str = "val"
) -> dict:
    manifest_path = manifest_path if manifest_path is not None else shard / "manifest.json"
    splits = manifest_splits(manifest_path)
    sample_ids = sorted(s for s, split_name in splits.items() if split_name == split)
    if not sample_ids:
        raise ValueError(
            f"{manifest_path}: no samples assigned to split {split!r} — "
            "the threshold sweep must run on the validation split only"
        )

    candidates = pooled_f_beta(shard, sample_ids, thresholds_ms)
    # Ties break to the LARGEST threshold: at equal F-beta, the fewer false
    # stops win. Deterministic regardless of grid order.
    best = max(candidates, key=lambda row: (row["f_beta"], row["threshold_ms"]))
    return {
        "protocol": (
            "threshold swept on the validation split only; selected by max F-beta "
            "(beta=2) on stop decisions; test numbers report at this single "
            "pre-registered operating point"
        ),
        "beta": 2.0,
        "split": split,
        "samples": sample_ids,
        "selected_threshold_ms": best["threshold_ms"],
        "selected": best,
        "candidates": candidates,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--shard", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=None, help="default: <shard>/manifest.json")
    parser.add_argument("--thresholds", type=str, default=DEFAULT_GRID_MS, help="comma-separated ms grid")
    parser.add_argument("--split", type=str, default="val")
    parser.add_argument("--out", type=Path, default=None, help="write the pre-registered point here")
    args = parser.parse_args(argv)

    thresholds_ms = [float(t) for t in args.thresholds.split(",") if t.strip()]
    if not thresholds_ms or any(t <= 0 for t in thresholds_ms):
        raise ValueError(f"thresholds must be positive ms values, got {args.thresholds!r}")

    result = select_operating_point(
        args.shard, manifest_path=args.manifest, thresholds_ms=sorted(thresholds_ms), split=args.split
    )
    print(json.dumps(result["selected"], indent=2))
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
