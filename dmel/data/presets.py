"""Preset definitions for the dMel POC corpus generator.

Presets fix the scenario mix, seed, shard size, and content scale so that
``python -m dmel.data.generate --preset <name>`` reproduces the corpus
byte-for-byte (see ``docs/dmel_data_contract.md``).

Scenario counts follow the contract v1 pilot mix: hesitation events are 60
pure samples plus 40 censored pairs, each pair producing one ``hesitation``
sample and one ``hesitation_commit`` sample (840 samples total, ~2 h).
``main`` scales the pilot counts x5 (contract: 10-20 h at x5-x10).

The ``fixtures`` preset is the tiny committed set (~10-20 s total audio)
covering every scenario, including one censored hesitation pair and one
truncated-world interruption (``forced_worlds`` pins the world variants so
the committed fixtures always exercise both worlds).
"""

from __future__ import annotations

from dataclasses import dataclass, replace

# Contract scenario order (also the iteration order when generating).
SCENARIO_ORDER = (
    "interruption",
    "backchannel",
    "normal_turn",
    "noise",
    "background_speaker",
    "hesitation",
    "hesitation_commit",
)

# Contract v1 pilot mix: 60 pure hesitation samples plus 40 censored pairs
# (each pair yields one more ``hesitation`` sample and one
# ``hesitation_commit`` sample). 840 samples total, ~2 h.
PILOT_COUNTS = {
    "interruption": 200,
    "backchannel": 150,
    "normal_turn": 150,
    "noise": 100,
    "background_speaker": 100,
    "hesitation": 60,
    "hesitation_commit": 40,
}

# Censored hesitation pairs per preset (planned by generate.py; each yields
# one ``hesitation`` + one ``hesitation_commit`` sample in addition to the
# pure hesitation count above).
CENSORED_PAIRS = {"pilot": 40, "main": 200, "fixtures": 1}

# main = pilot x5 -> ~4200 samples, ~10 h at ~9 s average.
MAIN_COUNTS = {name: count * 5 for name, count in PILOT_COUNTS.items()}

# Fixture set: every scenario exactly once (via one censored pair), short
# content. The pair covers hesitation + hesitation_commit.
FIXTURE_COUNTS = {name: 1 for name in SCENARIO_ORDER}
FIXTURE_FORCED_WORLDS = {
    "interruption": "truncated",  # contract: one truncated interruption
    "hesitation_commit": "truncated",
}


@dataclass(frozen=True)
class Preset:
    """Fully specified corpus preset. ``scale`` selects content size:

    - "full": 2-3 agent runs / multi-utterance samples (~7-12 s each)
    - "short": single short agent run per sample (~1.5-3 s, fixture set)
    """

    name: str
    seed: int
    scale: str
    shard_size: int
    scenario_counts: dict[str, int]
    censored_pairs: int = 0
    forced_worlds: dict[str, str] | None = None
    # Fixture consumers glob dmel/data/fixtures/ flat (features PR); the
    # fixtures preset therefore writes one directory, no shard-XXXXX level.
    flat_layout: bool = False


PILOT = Preset(
    name="pilot",
    seed=1234,
    scale="full",
    shard_size=100,
    scenario_counts=dict(PILOT_COUNTS),
    censored_pairs=CENSORED_PAIRS["pilot"],
)

MAIN = Preset(
    name="main",
    seed=1234,
    scale="full",
    shard_size=100,
    scenario_counts=dict(MAIN_COUNTS),
    censored_pairs=CENSORED_PAIRS["main"],
)

FIXTURES = Preset(
    name="fixtures",
    seed=1234,
    scale="short",
    shard_size=8,
    scenario_counts=dict(FIXTURE_COUNTS),
    censored_pairs=CENSORED_PAIRS["fixtures"],
    forced_worlds=dict(FIXTURE_FORCED_WORLDS),
    flat_layout=True,
)

PRESETS = {"pilot": PILOT, "main": MAIN, "fixtures": FIXTURES}


def resolve_preset(name: str, seed: int | None = None, shard_size: int | None = None) -> Preset:
    """Preset with optional CLI overrides applied."""
    preset = PRESETS[name]
    if seed is None and shard_size is None:
        return preset
    return replace(
        preset,
        seed=preset.seed if seed is None else seed,
        shard_size=preset.shard_size if shard_size is None else shard_size,
    )
