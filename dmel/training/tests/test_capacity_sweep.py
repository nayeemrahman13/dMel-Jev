"""Tests for capacity-sweep degenerate-run handling."""

import json
from pathlib import Path

import pytest

from dmel.training import capacity_sweep


class _FakeModel:
    def __init__(self, n: int = 123_456) -> None:
        self._n = n

    def num_parameters(self) -> int:
        return self._n


@pytest.fixture
def fake_loader(monkeypatch: pytest.MonkeyPatch) -> _FakeModel:
    model = _FakeModel()

    def _load(checkpoint: object, map_location: object = None):
        return model, {}

    monkeypatch.setattr(capacity_sweep, "load_model_from_checkpoint", _load)
    return model


def _row(threshold: float, predicted_stop_frames: int) -> dict:
    return {
        "threshold_ms": threshold,
        "metrics": {
            "totals": {
                "samples": 296,
                "predicted_stop_frames": predicted_stop_frames,
                "false_stops_total": 0,
            },
            "interruptions": {"recall": 0.0, "never_stopped_rate": 1.0},
            "meta": {"checkpoint": "/tmp/fake/best.pt", "threshold_ms": threshold},
        },
    }


def test_degenerate_never_stop_writes_metrics(
    tmp_path: Path, fake_loader: _FakeModel
) -> None:
    rows = [_row(t, 0) for t in (0.30, 0.50, 0.70)]
    out_dir = tmp_path / "w512_seed1"

    result = capacity_sweep.write_degenerate_metrics(
        rows, out_dir, Path("/tmp/fake/best.pt")
    )

    assert result is not None
    written = json.loads((out_dir / "metrics.json").read_text())
    # Row metrics copied verbatim; meta replaced with the degenerate selection.
    assert written["totals"] == rows[0]["metrics"]["totals"]
    assert written["meta"]["selected_stop_threshold"] is None
    assert written["meta"]["no_operating_point"] is True
    assert written["meta"]["params"] == fake_loader.num_parameters()
    assert result["selected_stop_threshold"] is None
    assert result["params"] == fake_loader.num_parameters()


def test_degenerate_refuses_frontier_with_stops(tmp_path: Path) -> None:
    rows = [_row(0.30, 0), _row(0.50, 12)]
    out_dir = tmp_path / "w256_seed0"

    result = capacity_sweep.write_degenerate_metrics(
        rows, out_dir, Path("/tmp/fake/best.pt")
    )

    assert result is None
    assert not (out_dir / "metrics.json").exists()
