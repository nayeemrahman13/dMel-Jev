"""Decision-log jsonl: policy outputs reduced to one record per 50 ms step.

Canonical record (one JSON object per line):

    {"sample_id": "shard00042-a013", "frame_index": 120, "t_ms": 6000, "action": "STOP_TTS"}

- ``frame_index``: 50 ms step within the sample (0-based).
- ``t_ms``: frame timestamp in ms from sample start (labels' ``t_ms``).
- ``action``: ``"KEEP"`` or ``"STOP_TTS"`` — every step is logged, not just stops.

The runner (eval-owned) is the canonical writer. The reader additionally accepts
common field aliases so decision logs emitted by other parallel tasks (e.g. the
baselines task) can be re-scored without re-running policies; anything unparseable
raises instead of being skipped.
"""

from __future__ import annotations

import json
from pathlib import Path

from dmel.eval.metrics import ACTIONS, MS_PER_STEP

_ALIASES: dict[str, tuple[str, ...]] = {
    "sample_id": ("sample_id", "sample", "clip_id"),
    "frame_index": ("frame_index", "frame", "step"),
    "t_ms": ("t_ms", "t"),
    "action": ("action", "action_pred", "pred_action", "policy_action"),
}


def decision_row(sample_id: str, frame_index: int, t_ms: int, action: str, probs: dict | None = None) -> dict:
    """One canonical decision-log record."""
    return {
        "sample_id": sample_id,
        "frame_index": int(frame_index),
        "t_ms": int(t_ms),
        "action": action,
        **({"probs": probs} if probs is not None else {}),
    }


def write_decisions(path: str | Path, rows: list[dict]) -> None:
    """Write decision records as jsonl (UTF-8, one compact object per line)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_decision_rows(path: str | Path) -> list[dict]:
    """Read + normalize a decision log into canonical records.

    Accepts alias field names (see module docstring); ``t_ms`` falls back to
    ``frame_index * MS_PER_STEP`` when absent. Raises ValueError on malformed
    lines, missing fields, or unknown actions — a corrupt log must not silently
    turn into wrong metrics.
    """
    rows: list[dict] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line_number, raw in enumerate(f, start=1):
            line = raw.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: not valid JSON ({exc})") from exc
            if not isinstance(record, dict):
                raise ValueError(f"{path}:{line_number}: expected a JSON object")
            fields: dict[str, object] = {}
            for canonical, aliases in _ALIASES.items():
                for alias in aliases:
                    if alias in record:
                        fields[canonical] = record[alias]
                        break
            missing = [k for k in ("sample_id", "frame_index", "action") if k not in fields]
            if missing:
                raise ValueError(f"{path}:{line_number}: decision record missing {missing}")
            action = str(fields["action"]).upper()
            if action not in ACTIONS:
                raise ValueError(f"{path}:{line_number}: unknown action {fields['action']!r}")
            frame_index = int(str(fields["frame_index"]))  # type: ignore[arg-type]
            t_ms = int(fields["t_ms"]) if "t_ms" in fields else frame_index * MS_PER_STEP  # type: ignore[arg-type]
            rows.append(decision_row(str(fields["sample_id"]), frame_index, t_ms, action, record.get("probs")))
    return rows


def stops_from_log(path: str | Path) -> dict[str, list[int]]:
    """Reduce a decision log to sample_id -> sorted STOP_TTS times in ms."""
    stops: dict[str, list[int]] = {}
    for row in read_decision_rows(path):
        if row["action"] == "STOP_TTS":
            stops.setdefault(row["sample_id"], []).append(row["t_ms"])
    return {sample_id: sorted(times) for sample_id, times in stops.items()}
