"""Contract v1 guardrail: the leakage linter.

The feature builder must consume only mix frames (input flags live at the
policy layer, not in features) — never label columns, scenario, or split.
This lints the area four ways:

1. builder signatures take only pcm/config — no label/flag/scenario/split
   parameters, so label data has no path into features;
2. the config surface rejects label-ish keys;
3. the production modules never reference label sidecars
   (``labels.parquet`` / ``manifest.json``) or scenario/split concepts;
4. the CLI leaves label/manifest sidecars untouched and produces caches
   invariant to sidecar contents.
"""

from __future__ import annotations

import inspect
from pathlib import Path

import numpy as np
import pytest
from _synth import sine

from dmel.features import config as config_module
from dmel.features import precompute as precompute_module
from dmel.features import tokenizer as tokenizer_module
from dmel.features.config import DmelConfig
from dmel.features.precompute import main
from dmel.features.tokenizer import (
    decode_tokens,
    log_mel_db,
    mel_filterbank,
    n_steps_for_samples,
    quantize_levels,
    tokenize,
    tokens_from_levels,
)

CFG = DmelConfig()
REPO_ROOT = Path(__file__).resolve().parents[3]

BUILDER_FUNCS = [
    n_steps_for_samples,
    mel_filterbank,
    log_mel_db,
    quantize_levels,
    tokens_from_levels,
    tokenize,
    decode_tokens,
]

# Input flags (agent_speaking, speaker_similarity) are policy-layer inputs per
# the contract's common policy interface — the feature builder must not take them.
FORBIDDEN_PARAM_NAMES = {
    "labels",
    "label",
    "label_columns",
    "scenario",
    "split",
    "split_assignment",
    "parquet",
    "manifest",
    "agent_speaking",
    "speaker_similarity",
}

PRODUCTION_MODULES = (config_module, tokenizer_module, precompute_module)
FORBIDDEN_SOURCE_TOKENS = ("labels.parquet", "manifest.json", "scenario", "split")


@pytest.mark.parametrize("func", BUILDER_FUNCS, ids=lambda f: f.__name__)
def test_builder_signature_has_no_label_or_flag_surface(func):
    param_names = set(inspect.signature(func).parameters)
    leaked = param_names & FORBIDDEN_PARAM_NAMES
    assert not leaked, f"{func.__name__} accepts forbidden parameters: {sorted(leaked)}"


def test_config_rejects_label_scenario_split_keys():
    for key in ("labels", "label_columns", "scenario", "split", "split_assignment"):
        with pytest.raises(ValueError):
            DmelConfig.from_mapping({key: "interruption"})


def test_production_modules_never_reference_label_sidecars():
    for module in PRODUCTION_MODULES:
        source = inspect.getsource(module)
        for token in FORBIDDEN_SOURCE_TOKENS:
            assert token not in source, (
                f"{module.__name__} references {token!r}; the feature builder "
                "consumes only mix frames"
            )


def test_cli_leaves_label_sidecars_untouched(tmp_path):
    shard = tmp_path / "shard-00001"
    shard.mkdir()
    (shard / "s0001.mix.pcm").write_bytes(sine(440.0, 500.0).tobytes())
    labels = shard / "s0001.labels.parquet"
    labels.write_bytes(b"parquet-bytes-with-label-columns")
    manifest = shard / "manifest.json"
    manifest.write_bytes(b'{"split": "train", "scenario": "interruption"}')

    assert main(["--shard", str(shard)]) == 0

    assert labels.read_bytes() == b"parquet-bytes-with-label-columns"
    assert manifest.read_bytes() == b'{"split": "train", "scenario": "interruption"}'
    cache = np.load(shard / "s0001.dmel.npy")
    assert np.array_equal(cache, tokenize(sine(440.0, 500.0), CFG))


def test_caches_invariant_to_label_sidecar_contents(tmp_path):
    caches = []
    for name, label_bytes in (("a", b"labels-v1"), ("b", b"TOTALLY-DIFFERENT-LABELS")):
        shard = tmp_path / name
        shard.mkdir()
        (shard / "s0001.mix.pcm").write_bytes(sine(440.0, 500.0).tobytes())
        (shard / "s0001.labels.parquet").write_bytes(label_bytes)
        assert main(["--shard", str(shard)]) == 0
        caches.append((shard / "s0001.dmel.npy").read_bytes())
    assert caches[0] == caches[1]
