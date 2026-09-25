"""CLI behavior: shard walk, caches, skip/overwrite, stems, exit codes."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from _synth import sine, white_noise

from dmel.features.config import DmelConfig
from dmel.features.precompute import main
from dmel.features.tokenizer import tokenize

CFG = DmelConfig()
REPO_ROOT = Path(__file__).resolve().parents[3]


def _make_shard(tmp_path: Path) -> Path:
    shard = tmp_path / "shard-00001"
    shard.mkdir()
    (shard / "s0001.mix.pcm").write_bytes(sine(440.0, 1000.0).tobytes())
    (shard / "s0002.mix.pcm").write_bytes(white_noise(1000.0, amp=0.3).tobytes())
    return shard


def test_shard_precompute_writes_caches_next_to_pcms(tmp_path, capsys):
    shard = _make_shard(tmp_path)
    assert main(["--shard", str(shard)]) == 0

    cache = shard / "s0001.dmel.npy"
    assert cache.exists()
    loaded = np.load(cache)
    assert loaded.dtype == np.int32
    assert loaded.shape == (20, 400)
    assert np.array_equal(loaded, tokenize(sine(440.0, 1000.0), CFG))
    assert "written=2 skipped=0" in capsys.readouterr().out


def test_rerun_skips_existing_caches(tmp_path, capsys):
    shard = _make_shard(tmp_path)
    assert main(["--shard", str(shard)]) == 0
    capsys.readouterr()
    assert main(["--shard", str(shard)]) == 0
    assert "written=0 skipped=2" in capsys.readouterr().out


def test_overwrite_and_stem_selection(tmp_path):
    shard = _make_shard(tmp_path)
    (shard / "s0002.agent.pcm").write_bytes(sine(880.0, 1000.0).tobytes())

    # Default run: mix stem, cache <sample_id>.dmel.npy.
    assert main(["--shard", str(shard)]) == 0
    mix_cache = np.load(shard / "s0002.dmel.npy")
    assert np.array_equal(mix_cache, tokenize(white_noise(1000.0, amp=0.3), CFG))

    # Agent stem with a distinct suffix keeps both caches side by side
    # (ablation arm E path); the mix cache is untouched.
    assert main(
        ["--shard", str(shard), "--stem", "agent", "--out-suffix", ".agent.dmel.npy"]
    ) == 0
    agent_cache = np.load(shard / "s0002.agent.dmel.npy")
    assert np.array_equal(agent_cache, tokenize(sine(880.0, 1000.0), CFG))
    assert not np.array_equal(mix_cache, agent_cache)
    assert np.array_equal(np.load(shard / "s0002.dmel.npy"), mix_cache)

    # Default suffix: the agent cache REPLACES <sample_id>.dmel.npy (contract
    # naming) — use --out-suffix to keep both.
    assert main(["--shard", str(shard), "--stem", "agent", "--overwrite"]) == 0
    assert np.array_equal(np.load(shard / "s0002.dmel.npy"), agent_cache)


def test_single_pcm_mode(tmp_path):
    pcm = tmp_path / "solo.mix.pcm"
    pcm.write_bytes(sine(440.0, 500.0).tobytes())
    assert main(["--pcm", str(pcm)]) == 0
    assert np.load(tmp_path / "solo.dmel.npy").shape == (10, 400)


def test_dry_run_writes_nothing(tmp_path):
    shard = _make_shard(tmp_path)
    assert main(["--shard", str(shard), "--dry-run"]) == 0
    assert not list(shard.glob("*.dmel.npy"))


def test_usage_errors(tmp_path):
    assert main(["--shard", str(tmp_path / "missing")]) == 2
    empty = tmp_path / "empty"
    empty.mkdir()
    assert main(["--shard", str(empty)]) == 2
    with pytest.raises(SystemExit):  # argparse: exactly one of --shard/--pcm
        main([])
    with pytest.raises(SystemExit):
        main(["--shard", str(tmp_path), "--pcm", str(tmp_path)])


def test_corrupt_pcm_fails_fast_without_partial_caches(tmp_path):
    shard = _make_shard(tmp_path)
    (shard / "s0003.mix.pcm").write_bytes(b"\x01")  # odd byte count
    assert main(["--shard", str(shard)]) == 1
    assert not list(shard.glob("*.dmel.npy"))  # nothing written (tokenize-all-then-write)


def test_module_entrypoint_help():
    proc = subprocess.run(
        [sys.executable, "-m", "dmel.features.precompute", "--help"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0
    assert "dmel" in proc.stdout
