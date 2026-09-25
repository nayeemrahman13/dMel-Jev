"""Corpus generator CLI.

``python -m dmel.data.generate --preset pilot`` reproduces the pilot corpus
byte-for-byte (identical CLI args + seed ⇒ byte-identical outputs). Writes
shards of ``<sample_id>.{user,agent,mix}.pcm`` + ``.labels.parquet`` plus one
``manifest.json`` per shard; bulk audio directories are gitignored (the
committed fixture set is generated with ``--preset fixtures``).

Contract v1 structures applied here:

- cells: every sample draws a (voice_pair, template_slot) cell and inherits
  the cell's split; per-sample structural seeds follow the split ranges.
- censored hesitation pairs: one cell, one seed, shared levels/codec —
  byte-identical audio up to the censoring point.
- codec degradation: eval-only, decided from the augmentation RNG keyed on
  sample id (pair id for censored pairs, so the shared prefix cannot
  diverge — a required refinement of "keyed on sample id").
- manifests record per event: evidence-window jitter, world variant, and for
  pairs the shared-prefix length; per sample: split, agent_runs count, flag
  jitter, augmentation parameters.
"""

from __future__ import annotations

import argparse
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

from dmel.data import presets as presets_mod
from dmel.data.annotation import annotate
from dmel.data.constants import (
    CODEC_EVAL_RATE,
    MS_TO_SAMPLES,
    TAG_CELL,
    TAG_LEVELS,
    TAG_NOISE,
    TAG_STRUCTURE,
    augmentation_key,
)
from dmel.data.mixing import draw_codec, mix_sample
from dmel.data.presets import SCENARIO_ORDER, Preset
from dmel.data.scenarios import (
    SampleScript,
    _slot_class,
    bank_pools,
    build_pools,
    draw_cell,
    plan_hesitation_pair,
    plan_sample,
    sample_seed,
)
from dmel.data.shards import shard_dir, write_labels_parquet, write_manifest, write_pcm
from dmel.data.synthesis import RenderCache, espeak_version

DEFAULT_OUT = {
    "pilot": Path("data/pilot"),
    "main": Path("data/main"),
    "fixtures": Path("dmel/data/fixtures"),
}


def _codec_decision(key: str, split: str) -> dict | None:
    """Eval-only codec family, decided from the id-keyed augmentation RNG."""
    if split == "train":
        return None
    rng = np.random.default_rng([augmentation_key(key), 1])
    if rng.random() < CODEC_EVAL_RATE:
        return draw_codec(rng)
    return None


def _n_slots(scenario: str, scale: str) -> int:
    """Number of template slots in this scenario's semantic class."""
    return len(bank_pools(scale)[_slot_class(scenario)])


def _plan_corpus(preset: Preset, render) -> list[SampleScript]:
    """Plan every sample in deterministic order. ``render`` synthesizes on
    demand (placement uses measured utterance durations)."""
    salt = f"{preset.name}|{preset.seed}"
    pools = build_pools(salt=salt, scale=preset.scale)
    rng_cell = np.random.default_rng([preset.seed, TAG_CELL])
    scripts: list[SampleScript] = []

    # Singles for the base scenarios (hesitation here = pure events only;
    # pair members are planned below and add one more hesitation sample).
    for scenario in SCENARIO_ORDER:
        if scenario == "hesitation_commit":
            continue  # produced by the pair loop
        count = preset.scenario_counts.get(scenario, 0)
        for i in range(count):
            cell = draw_cell(scenario, _n_slots(scenario, preset.scale), rng_cell, pools)
            seed = sample_seed(preset.seed, len(scripts), cell.split)
            rng_struct = np.random.default_rng([seed, TAG_STRUCTURE])
            rng_levels = np.random.default_rng([seed, TAG_LEVELS])
            sample_id = f"{preset.name}-{scenario}-{i:04d}"
            script = plan_sample(
                scenario,
                sample_id,
                preset.scale,
                cell,
                rng_struct,
                rng_levels,
                render,
                forced_world=(preset.forced_worlds or {}).get(scenario),
            )
            script.codec = _codec_decision(sample_id, cell.split)
            scripts.append(script)

    # Censored hesitation pairs: one cell, one seed, shared levels/codec.
    for j in range(preset.censored_pairs):
        cell = draw_cell("hesitation", _n_slots("hesitation", preset.scale), rng_cell, pools)
        seed = sample_seed(preset.seed, 1_000_000 + j, cell.split)
        rng_struct = np.random.default_rng([seed, TAG_STRUCTURE])
        rng_levels = np.random.default_rng([seed, TAG_LEVELS])
        pair_id = f"{preset.name}-hespair-{j:04d}"
        script_a, script_b = plan_hesitation_pair(
            pair_id,
            f"{pair_id}-hes",
            f"{pair_id}-commit",
            preset.scale,
            cell,
            rng_struct,
            rng_levels,
            render,
            forced_world=(preset.forced_worlds or {}).get("hesitation_commit"),
        )
        decision = _codec_decision(pair_id, cell.split)
        script_a.codec = decision
        script_b.codec = decision
        scripts.append(script_a)
        scripts.append(script_b)
    return scripts


def _mix_and_label(scripts: list[SampleScript], cache: RenderCache) -> list[dict[str, np.ndarray]]:
    """Mix stems and derive labels for each planned script, in order."""
    out: list[dict[str, np.ndarray]] = []
    for script in scripts:
        rendered: dict[tuple[str, str, int], np.ndarray] = {}
        for u in script.utterances:
            key = (u.text, script.voices[u.role], script.rates[u.role])
            if key not in rendered:
                rendered[key] = cache.render(u.text, script.voices[u.role], script.rates[u.role])
        noise_key = script.pair_id or script.sample_id
        rng_noise = np.random.default_rng([augmentation_key(noise_key), TAG_NOISE])
        stems = mix_sample(script, rendered, rng_noise)
        labels = annotate(script, stems["mix"].size)
        out.append({**stems, "labels": labels})
    return out


def _sample_record(script: SampleScript, n_samples: int) -> dict:
    return {
        "id": script.sample_id,
        "scenario": script.scenario,
        "split": script.split,
        "world": script.world,
        "agent_runs": len(script.runs),
        "duration_ms": n_samples // MS_TO_SAMPLES,
        "agent_flag_jitter_ms": script.agent_flag_jitter_ms,
        "shared_prefix_ms": script.shared_prefix_ms,
        "pair_id": script.pair_id,
        "codec": script.codec,
        "snr_db": round(script.snr_db, 3),
        "background_db": round(script.background_db, 3),
        "noise_db": round(script.noise_db, 3),
        "voices": dict(script.voices),
        "rates": dict(script.rates),
        "events": [
            {
                "onset_ms": e.onset // MS_TO_SAMPLES,
                "earliest_stop_ms": e.earliest_stop // MS_TO_SAMPLES,
                "evidence_ms": e.evidence_ms,
                "world": e.world,
            }
            for e in script.interruptions
        ],
    }


def generate(preset: Preset, out_root: Path) -> dict:
    """Generate the full corpus for a preset under ``out_root``."""
    started = time.monotonic()
    cache = RenderCache()

    scripts = _plan_corpus(preset, cache)
    rendered = _mix_and_label(scripts, cache)

    manifest_common = {
        "preset": {
            "name": preset.name,
            "seed": preset.seed,
            "scale": preset.scale,
            "shard_size": preset.shard_size,
            "scenario_counts": dict(preset.scenario_counts),
            "censored_pairs": preset.censored_pairs,
            "forced_worlds": dict(preset.forced_worlds or {}),
        },
        "determinism": {
            "espeak": espeak_version(),
            "command": f"python -m dmel.data.generate --preset {preset.name}",
            "note": "identical CLI args + seed produce byte-identical outputs",
        },
        "split": build_pools(salt=f"{preset.name}|{preset.seed}", scale=preset.scale).summary(),
        "noise_events": {
            "source": "numpy shaped-noise synthesis (no CC0 pack fetched)",
            "kinds": ["cough", "breath", "burst"],
        },
        "codec_family": {
            "eval_only": True,
            "rate": CODEC_EVAL_RATE,
            "key": "sample id (pair id for censored pairs — keeps the shared prefix identical)",
        },
    }

    counts = Counter(s.scenario for s in scripts)
    split_counts = Counter(s.split for s in scripts)
    world_counts = Counter(s.world or "none" for s in scripts)
    runs_dist = Counter(len(s.runs) for s in scripts)

    n_shards = (len(scripts) + preset.shard_size - 1) // preset.shard_size
    for shard_index in range(n_shards):
        lo = shard_index * preset.shard_size
        chunk = scripts[lo : lo + preset.shard_size]
        sdir = out_root if preset.flat_layout else shard_dir(out_root, shard_index)
        sdir.mkdir(parents=True, exist_ok=True)
        records = []
        for script, stems in zip(chunk, rendered[lo : lo + preset.shard_size]):
            write_pcm(sdir / f"{script.sample_id}.user.pcm", stems["user"])
            write_pcm(sdir / f"{script.sample_id}.agent.pcm", stems["agent"])
            write_pcm(sdir / f"{script.sample_id}.mix.pcm", stems["mix"])
            write_labels_parquet(sdir / f"{script.sample_id}.labels.parquet", stems["labels"])
            records.append(_sample_record(script, stems["mix"].size))
        manifest = {
            **manifest_common,
            "shard_index": shard_index,
            "shard_samples": len(chunk),
            "counts": {
                "samples_total": len(scripts),
                "by_scenario": dict(sorted(counts.items())),
                "by_split": dict(sorted(split_counts.items())),
                "by_world": dict(sorted(world_counts.items())),
                "agent_runs_distribution": {str(k): v for k, v in sorted(runs_dist.items())},
            },
            "samples": records,
        }
        write_manifest(sdir / "manifest.json", manifest)

    return {
        "samples": len(scripts),
        "shards": n_shards,
        "by_scenario": dict(counts),
        "by_split": dict(split_counts),
        "by_world": dict(world_counts),
        "agent_runs": dict(runs_dist),
        "wall_s": time.monotonic() - started,
        "out": str(out_root),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="dmel.data.generate", description=__doc__)
    parser.add_argument("--preset", required=True, choices=sorted(presets_mod.PRESETS))
    parser.add_argument("--out", type=Path, default=None, help="output root (default per preset)")
    parser.add_argument("--seed", type=int, default=None, help="override preset seed")
    parser.add_argument("--shard-size", type=int, default=None, help="override shard size")
    args = parser.parse_args(argv)

    preset = presets_mod.resolve_preset(args.preset, seed=args.seed, shard_size=args.shard_size)
    out_root = args.out or DEFAULT_OUT[args.preset]

    summary = generate(preset, out_root)
    print(
        f"generated {summary['samples']} samples in {summary['shards']} shards "
        f"at {summary['out']} ({summary['wall_s']:.1f}s)"
    )
    print(f"  scenarios: {summary['by_scenario']}")
    print(f"  splits: {summary['by_split']}")
    print(f"  worlds: {summary['by_world']}")
    print(f"  agent_runs: {summary['agent_runs']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
