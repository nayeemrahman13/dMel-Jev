"""End-to-end harness tests: runner, decisions, sweep, report, CLI.

Policies and labels come only from this test tree (contract interface); no
other dmel area's code is imported.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from dmel.eval import cli
from dmel.eval.cli import _metrics_from_decision_log
from dmel.eval.decisions import stops_from_log, write_decisions
from dmel.eval.labels import discover_samples, load_labels
from dmel.eval.metrics import compute_metrics
from dmel.eval.report import render_report
from dmel.eval.runner import load_policy, run_shard
from dmel.eval.sweep import render_sweep_table, select_operating_point, sweep_thresholds
from dmel.eval.tests.helpers import RecordingPolicy, ScriptedPolicy, interruption_sample, labels_df, write_sample

POLICY_SPEC = "dmel.eval.tests.helpers:ScriptedPolicy"


@pytest.fixture()
def shard(tmp_path):
    shard_dir = tmp_path / "shard-00001"
    write_sample(shard_dir, "int60", interruption_sample(), with_dmel_cache=True)
    bc = labels_df(40, agent_speaking={0: 0, 5: 1}, scenario={10: "backchannel"}, speech_present={10: 1})
    write_sample(shard_dir, "bc40", bc, with_dmel_cache=True)
    return shard_dir


def test_discover_samples_and_load_policy(shard):
    sample_ids = [sample.sample_id for sample in discover_samples(shard)]
    assert sample_ids == ["bc40", "int60"]
    policy = load_policy(POLICY_SPEC, {"stop_at_ms": 1400})
    assert isinstance(policy, ScriptedPolicy)
    with pytest.raises(ValueError):
        load_policy("no-such-spec", {})
    with pytest.raises((ImportError, ModuleNotFoundError)):
        load_policy("not_a_module:Policy", {})


def test_runner_matches_metric_engine(shard):
    """Runner metrics must equal compute_metrics over the runner's own decision log."""
    result = run_shard(ScriptedPolicy(stop_at_ms=1400), shard)
    stops = {sid: ts for sid, ts in result.stream.stops_by_sample.items() if ts}
    labels_by_sample = {s.sample_id: load_labels(s.labels) for s in discover_samples(shard)}
    expected = compute_metrics(labels_by_sample, stops)
    assert result.metrics["interruptions"]["total_events"] == expected["interruptions"]["total_events"]
    assert result.metrics["totals"] == expected["totals"]
    # int60: first in-window stop at 1400 (latency 200); its 21 later stops fall
    # beyond the 1900 grace end and bc40's 12 stops have no event -> 33 false stops
    assert result.metrics["interruptions"]["decision_latency_ms"]["median"] == pytest.approx(200.0)
    assert result.metrics["totals"]["false_stops_total"] == 33
    assert result.metrics["meta"]["samples_run"] == 2


def test_decision_log_roundtrip(shard, tmp_path):
    out_decisions = tmp_path / "decisions.jsonl"
    result = run_shard(ScriptedPolicy(stop_at_ms=1400), shard)
    write_decisions(out_decisions, result.stream.decisions)
    stops = stops_from_log(str(out_decisions))
    assert stops == {
        "int60": list(range(1400, 3000, 50)),
        "bc40": list(range(1400, 2000, 50)),
    }
    # Tolerant aliases: clip_id/frame/pred_action must read back identically.
    aliases = tmp_path / "aliases.jsonl"
    aliases.write_text(
        json.dumps({"clip_id": "int60", "frame": 30, "pred_action": "STOP_TTS"}) + "\n"
        + json.dumps({"clip_id": "int60", "frame": 31, "pred_action": "KEEP"}) + "\n",
        encoding="utf-8",
    )
    assert stops_from_log(str(aliases)) == {"int60": [1500]}
    (tmp_path / "corrupt.jsonl").write_text("{not json}\n", encoding="utf-8")
    with pytest.raises(ValueError):
        stops_from_log(str(tmp_path / "corrupt.jsonl"))


def test_recompute_metrics_from_log(shard, tmp_path):
    """Deliverable 2: pre-generated decision logs recompute without re-running."""
    out_decisions = tmp_path / "decisions.jsonl"
    result = run_shard(ScriptedPolicy(stop_at_ms=700), shard)  # 700 lands in bc40's backchannel span
    write_decisions(out_decisions, result.stream.decisions)
    labels_by_sample = {s.sample_id: load_labels(s.labels) for s in discover_samples(shard)}
    expected = compute_metrics(labels_by_sample, stops_from_log(str(out_decisions)))
    recomputed = _metrics_from_decision_log(str(out_decisions), str(shard))
    assert recomputed["totals"] == expected["totals"]
    assert recomputed["interruptions"]["total_events"] == expected["interruptions"]["total_events"]
    assert recomputed["false_stops_by_scenario"]["backchannel"]["false_stop_rate"] == pytest.approx(1.0)


def test_flag_only_path_warns_and_hides_tokens(shard):
    """--ignore-dmel-cache: policies receive dmel_token_ids=None + a warning per sample."""
    policy = RecordingPolicy(threshold_ms=10_000)  # never stops
    result = run_shard(policy, shard, ignore_dmel_cache=True)
    assert result.stream.warnings and all("flag-only" in w for w in result.stream.warnings)
    assert policy.seen_token_ids and all(tokens is None for tokens in policy.seen_token_ids)

    policy = RecordingPolicy(threshold_ms=10_000)
    result = run_shard(policy, shard)
    assert not result.stream.warnings
    assert policy.seen_token_ids and all(tokens is not None for tokens in policy.seen_token_ids)


def test_runner_skips_incomplete_samples(tmp_path):
    shard_dir = tmp_path / "shard-00001"
    write_sample(shard_dir, "good", interruption_sample())
    write_sample(shard_dir, "broken", interruption_sample(), with_user_pcm=False)
    result = run_shard(ScriptedPolicy(stop_at_ms=1400), shard_dir)
    assert [s["sample_id"] for s in result.stream.skipped_samples] == ["broken"]
    assert "user.pcm" in result.stream.skipped_samples[0]["reason"]
    assert result.metrics["meta"]["samples_run"] == 1
    assert result.metrics["meta"]["samples_skipped"] == 1


def test_runner_resets_between_samples(shard):
    policy = ScriptedPolicy(stop_at_ms=1400)
    run_shard(policy, shard)
    # reset() runs at the START of each sample, so the counter ends at the last
    # sample's step count (int60: 60 frames); without per-sample reset the
    # policy's own clock would drift and later samples' stop times would shift.
    assert policy._step == 60


def test_sweep_selects_max_fbeta_point(shard):
    """Hand-checked frontier on the two-sample shard: earlier stop-at wins on
    F-beta because it catches the GT STOP span (recall dominates at beta=2)."""
    rows = sweep_thresholds(
        lambda thr: ScriptedPolicy(stop_at_ms=thr),
        str(shard),
        [1200, 1950],
        bootstrap=False,
    )
    assert [row["threshold_ms"] for row in rows] == [1200, 1950]
    # Stop-at-1200 across both samples: tp=32, fp=20 (4 int60 + 16 bc40), fn=0
    # -> F-beta(2) = 5*p*r/(4p+r) with p=32/52, r=1.0 = 8/9.
    assert rows[0]["metrics"]["stop_decision"]["fbeta"] == pytest.approx(8 / 9)
    # Stop-at-1950: int60 stops on frames 39-59 (21 tp), bc40's stop is a false
    # stop (fp=1), fn=11 -> p=21/22, r=21/32, F-beta(2) = 2205/3150 = 0.7 exactly.
    assert rows[-1]["metrics"]["stop_decision"]["fbeta"] == pytest.approx(0.7)
    assert rows[-1]["metrics"]["interruptions"]["never_stopped_rate"] == 1.0
    assert rows[-1]["metrics"]["totals"]["false_stops_total"] == 22  # 21 int60 + 1 bc40, all past grace
    selected = select_operating_point(rows)
    assert selected is not None and selected["threshold_ms"] == 1200
    table = render_sweep_table(rows)
    assert "Pre-registered operating point" in table and "1200" in table
    assert "| Threshold (ms) |" in table
    assert select_operating_point([]) is None


def test_report_opens_with_load_bearing_assumption(shard, tmp_path):
    with_constants = run_shard(
        ScriptedPolicy(stop_at_ms=1400), shard,
        cancel_path_constants={"vad_ms": 10.0, "cancel_send_ms": 5.0, "audio_stop_ms": 35.0},
        cancel_path_source="tests", cancel_path_retrieved="2026-09-25",
    ).metrics
    without_constants = run_shard(ScriptedPolicy(stop_at_ms=1400), shard).metrics
    markdown = render_report({"A": with_constants, "C": without_constants})
    lines = [line for line in markdown.splitlines() if line.strip()]
    # First substantive lines after the title: the assumption header, then its paragraph.
    assert lines[0].startswith("# ")
    assert "Load-bearing assumption" in lines[1]
    assert "policy-vs-labeler agreement" in lines[2]
    assert "| D |" in markdown and "not trained in V1" in markdown
    assert "| E |" in markdown
    assert "retrieved:" in markdown and "tests" in markdown  # cancel-path provenance
    assert "not measured" in markdown  # arms without constants
    assert "F-beta" in markdown  # protocol section names the operating-point rule


def test_report_multi_seed_summary(shard):
    m1 = run_shard(ScriptedPolicy(stop_at_ms=1400), shard, ci_seed=1).metrics
    m2 = run_shard(ScriptedPolicy(stop_at_ms=1450), shard, ci_seed=2).metrics
    markdown = render_report({"B": [m1, m2]})
    assert "Seed summary (2 seed files" in markdown and "\u00b1" in markdown
    assert "Bootstrap CIs (95%" in markdown


def test_cli_run_and_report(shard, tmp_path):
    out_json = tmp_path / "metrics.json"
    out_decisions = tmp_path / "decisions.jsonl"
    code = cli.main(
        [
            "run",
            "--policy", POLICY_SPEC,
            "--policy-kwargs", '{"stop_at_ms": 1400}',
            "--labels-dir", str(shard),
            "--out-json", str(out_json),
            "--out-decisions", str(out_decisions),
            "--no-bootstrap",
        ]
    )
    assert code == 0
    metrics = json.loads(out_json.read_text())
    assert metrics["meta"]["samples_run"] == 2
    assert out_decisions.exists()

    out_md = tmp_path / "report.md"
    code = cli.main(["report", "--metrics", f"A={out_json}", "--out-md", str(out_md)])
    assert code == 0
    assert "Load-bearing assumption" in out_md.read_text()

    # Recompute path: report straight from a decision log.
    code = cli.main(
        ["report", "--metrics", "A=unused.json", "--recompute-from-log", str(out_decisions), "--labels-dir", str(shard), "--no-bootstrap"]
    )
    assert code == 0


def test_cli_cancel_path_constants(tmp_path):
    out_json = tmp_path / "m.json"
    out_json.write_text("{}", encoding="utf-8")
    with pytest.raises(SystemExit, match="missing"):
        cli.main(["report", "--metrics", f"A={out_json}", "--cancel-path-constants", "vad_ms=1,cancel_send_ms=2"])
    with pytest.raises(SystemExit, match="unexpected"):
        cli.main(
            ["report", "--metrics", f"A={out_json}", "--cancel-path-constants", "vad_ms=1,cancel_send_ms=2,audio_stop_ms=3,bogus=4"]
        )


def test_cli_sweep(shard, tmp_path, capsys):
    out_json = tmp_path / "sweep.json"
    code = cli.main(
        [
            "sweep",
            "--policy-factory", POLICY_SPEC,
            "--labels-dir", str(shard),
            "--thresholds", "1200,1400",
            "--out-json", str(out_json),
            "--no-bootstrap",
        ]
    )
    assert code == 0
    rows = json.loads(out_json.read_text())
    assert [row["threshold_ms"] for row in rows] == [1200.0, 1400.0]
    table = capsys.readouterr().out
    assert "Pre-registered operating point" in table


def test_dmel_cache_shapes_are_checked(shard):
    """A cache with the wrong number of steps is a warning + None tokens, not silent corruption."""
    bad = shard / "int60.dmel.npy"
    np.save(bad, np.zeros((3, 2), dtype=np.int32))
    policy = RecordingPolicy(threshold_ms=10_000)
    result = run_shard(policy, shard)
    assert any("dmel cache" in w for w in result.stream.warnings)
    int60_tokens = policy.seen_token_ids[-60:]  # bc40 ran first; the last 60 steps are int60
    assert all(tokens is None for tokens in int60_tokens)
