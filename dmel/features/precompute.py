"""CLI: precompute dMel token caches for contract shards.

    python -m dmel.features.precompute --shard data/pilot/shard-00001
    python -m dmel.features.precompute --shard data/pilot/shard-00001 --stem agent --out-suffix .agent.dmel.npy
    python -m dmel.features.precompute --pcm /path/to/<id>.mix.pcm --config my-config.yaml

Walks ``<sample_id>.<stem>.pcm`` files (sorted order, deterministic) and writes
``<sample_id>.dmel.npy`` token caches next to them (``--out-suffix`` overrides
the cache name — the agent-stem run for ablation arm E can use a distinct
suffix so both caches coexist). Every pcm is read and tokenized before anything
is written, so a corrupt file can never leave a partial cache. Existing caches
are skipped unless ``--overwrite``.

Exit codes: 0 ok, 1 runtime error (e.g. corrupt pcm), 2 usage error.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from dmel.features.config import DEFAULT_CONFIG_PATH, DmelConfig
from dmel.features.tokenizer import read_pcm_i16, tokenize

EXIT_OK = 0
EXIT_ERROR = 1
EXIT_USAGE = 2


class _UsageError(ValueError):
    """Bad CLI input (wrong paths / no matching files) -> exit code 2."""


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m dmel.features.precompute",
        description="Precompute dMel token caches (<sample_id>.dmel.npy) for a contract shard.",
    )
    parser.add_argument("--shard", type=Path, help="shard directory containing <sample_id>.<stem>.pcm files")
    parser.add_argument("--pcm", type=Path, help="tokenize a single <sample_id>.<stem>.pcm file instead")
    parser.add_argument("--stem", default="mix", choices=["mix", "agent", "user"], help="which stem to tokenize (default: mix)")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH, help="YAML DmelConfig (default: bundled default)")
    parser.add_argument("--overwrite", action="store_true", help="recompute caches that already exist")
    parser.add_argument("--out-suffix", default=".dmel.npy", help="cache filename suffix (default: .dmel.npy)")
    parser.add_argument("--dry-run", action="store_true", help="report what would be written without writing")
    args = parser.parse_args(argv)
    if (args.shard is None) == (args.pcm is None):
        parser.error("exactly one of --shard or --pcm is required")
    return args


def _collect_targets(base: Path, stem: str) -> list[Path]:
    suffix = f".{stem}.pcm"
    if base.is_dir():
        files = sorted(base.glob(f"*{suffix}"))
        if not files:
            raise _UsageError(f"no *{suffix} files in {base}")
        return files
    if base.is_file() and base.name.endswith(suffix):
        return [base]
    raise _UsageError(f"{base} is not a shard directory and not a *{suffix} file")


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    base = args.shard if args.shard is not None else args.pcm
    try:
        config = DmelConfig.from_yaml(args.config)
        files = _collect_targets(base, args.stem)
    except (ValueError, OSError) as exc:
        print(f"precompute: {exc}", file=sys.stderr)
        return EXIT_USAGE

    stem_suffix = f".{args.stem}.pcm"
    pending: list[tuple[Path, np.ndarray, str]] = []
    skipped = 0
    for path in files:
        sample_id = path.name.removesuffix(stem_suffix)
        out = path.parent / f"{sample_id}{args.out_suffix}"
        if out.exists() and not args.overwrite:
            skipped += 1
            continue
        try:
            pcm = read_pcm_i16(path)
            tokens = tokenize(pcm, config)
        except (ValueError, OSError) as exc:
            print(f"precompute: {path}: {exc}", file=sys.stderr)
            return EXIT_ERROR
        pending.append((out, tokens, sample_id))

    if not args.dry_run:
        for out, tokens, _ in pending:
            with out.open("wb") as handle:
                np.save(handle, tokens)

    for out, tokens, sample_id in pending:
        print(f"{sample_id}: tokens shape={tokens.shape} dtype={tokens.dtype} -> {out.name}")
    print(f"precompute: files={len(files)} written={len(pending)} skipped={skipped}")
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
