"""Materialize dMel token caches for contract shards.

Thin wrapper over the features area's precompute CLI: committed fixtures
ship pcm stems and labels but never token caches (bulk caches are
gitignored), so consumers in this area materialize them on demand before
loading tokens. Idempotent — the CLI skips samples whose caches exist.
"""

from __future__ import annotations

from pathlib import Path

TOKEN_CACHE_SUFFIX = ".dmel.npy"


def has_token_caches(root: Path) -> bool:
    """True when every labels.parquet in the shard tree has a token cache."""
    parquets = list(root.glob("**/*.labels.parquet"))
    return bool(parquets) and all(
        (p.parent / (p.name.replace(".labels.parquet", "") + TOKEN_CACHE_SUFFIX)).exists()
        for p in parquets
    )


def ensure_token_caches(root: Path, *, stem: str = "mix") -> None:
    """Run the dMel features precompute over the shard tree at root.

    In-process call (same interpreter) so test/CI environments need no
    extra entry point; the CLI is deterministic and skips existing caches.
    """
    from dmel.features.precompute import main as precompute_main

    precompute_main(["--shard", str(root), "--stem", stem])
