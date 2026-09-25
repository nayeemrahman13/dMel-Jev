"""Materialize dMel token caches for contract shards.

Thin wrapper over the features area's precompute CLI: committed fixtures
ship pcm stems and labels but never token caches (bulk caches are
gitignored), so consumers in this area materialize them on demand before
loading tokens. Idempotent — the CLI skips samples whose caches exist.
"""

from __future__ import annotations

from pathlib import Path

from dmel.training.dataset import DatasetError

TOKEN_CACHE_SUFFIX = ".dmel.npy"


def has_token_caches(root: Path) -> bool:
    """True when every labels.parquet in the shard tree has a token cache."""
    parquets = list(root.glob("**/*.labels.parquet"))
    return bool(parquets) and all(
        (p.parent / (p.name.replace(".labels.parquet", "") + TOKEN_CACHE_SUFFIX)).exists()
        for p in parquets
    )


def _shard_dirs(root: Path) -> list[Path]:
    """Leaf shard dirs under root: root itself when it directly holds
    labels, otherwise every dir below it that does (what scan_samples sees)."""
    if list(root.glob("*.labels.parquet")):
        return [root]
    return sorted({p.parent for p in root.glob("**/*.labels.parquet")})


def ensure_token_caches(root: Path, *, stem: str = "mix") -> None:
    """Materialize dMel token caches for the shard tree at root.

    Accepts what scan_samples accepts: a single shard dir or a parent of
    shard dirs (the precompute CLI itself only walks one shard). A no-op
    when every label file already has its cache — uploaded snapshots skip
    the on-worker compute entirely. The CLI is deterministic and skips
    existing caches; a nonzero exit fails loudly instead of silently
    leaving the dataset half-materialized.
    """
    from dmel.features.precompute import main as precompute_main

    if has_token_caches(root):
        return
    shards = _shard_dirs(root)
    for shard in shards:
        if precompute_main(["--shard", str(shard), "--stem", stem]) != 0:
            raise DatasetError(f"token-cache precompute failed for {shard}")
