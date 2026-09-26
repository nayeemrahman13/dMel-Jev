"""Session recorder: dump mic frames + per-step decisions + timestamps.

``--record`` mode writes ONE ``.npz`` per session:

- ``frames`` — int16 ``(n_frames, 800)``: exactly what the microphone heard,
  16 kHz mono, in 50 ms frames, in order.
- ``decisions_json`` — JSON array of per-step decision records (step_index,
  t_ms, action, probs, agent_speaking, tokens_provided, stop_fired, dec_ms,
  wall_s).
- ``meta_json`` — JSON object: schema version, arm, voice/rate, script source,
  sample rate, frame size, saved-at timestamp, and caller extras.

Sessions can then be scored offline against ``dmel/eval`` (convert to the
eval harness's shard layout when needed — the record deliberately keeps the
raw streams, not eval's parquet format).
"""

from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from dmel.demo.session import StepDecision

SESSION_RECORD_SCHEMA = "dmel-demo-session-v1"


def save_session(
    path: str | Path,
    *,
    frames: list[np.ndarray] | np.ndarray | None,
    decisions: list[StepDecision],
    meta: dict | None = None,
) -> Path:
    """Write one session record; returns the written path."""
    if frames is None:
        frames_arr = np.zeros((0, 800), dtype=np.int16)
    elif isinstance(frames, np.ndarray):
        frames_arr = np.asarray(frames, dtype=np.int16)
    elif frames:
        frames_arr = np.stack([np.asarray(f, dtype=np.int16) for f in frames])
    else:
        frames_arr = np.zeros((0, 800), dtype=np.int16)
    payload = {
        "schema": SESSION_RECORD_SCHEMA,
        "frames": int(frames_arr.shape[0]),
        "decisions": len(decisions),
        "sample_rate": 16_000,
        "frame_samples": 800,
        "saved_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        **(meta or {}),
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        frames=frames_arr,
        decisions_json=np.array(json.dumps([asdict(d) for d in decisions])),
        meta_json=np.array(json.dumps(payload, default=str)),
    )
    return path


def load_session(path: str | Path) -> tuple[np.ndarray, list[dict], dict]:
    """Read one session record back: (frames (n, 800) int16, decisions, meta)."""
    with np.load(Path(path), allow_pickle=False) as data:
        frames = data["frames"]
        decisions = json.loads(str(data["decisions_json"]))
        meta = json.loads(str(data["meta_json"]))
    return frames, decisions, meta
