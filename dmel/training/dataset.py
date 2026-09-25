"""Dataset loader for dMel contract shards.

Layout (per contract v1): ``<root>/shard-XXXXX/manifest.json`` plus
``<sample_id>.labels.parquet`` (one row per 50 ms frame) and
``<sample_id>.dmel.npy`` token caches written by the dMel features
precompute.

The train/val/test assignment is the manifest ``split`` column — never a
local re-split. Training consumes train, model/threshold selection consumes
val; test is reserved for the eval harness's final report numbers.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

ACTION_TO_INDEX = {"KEEP": 0, "STOP_TTS": 1}
LABEL_COLUMNS = (
    "frame_index",
    "t_ms",
    "agent_speaking",
    "speech_present",
    "primary_user",
    "backchannel",
    "interrupt_intent",
    "action",
    "earliest_reasonable_stop_ms",
    "scenario",
)
AUX_LABEL_NAMES = ("speech_present", "primary_user", "backchannel", "interrupt_intent")
SPLITS = ("train", "val", "test")


class DatasetError(ValueError):
    """Raised when shard contents do not conform to the contract."""


@dataclass(frozen=True)
class SampleData:
    sample_id: str
    path: Path
    tokens: np.ndarray  # (n_steps, tokens_per_step) int64
    action: np.ndarray  # (n_steps,) int64 in {0, 1}
    agent_speaking: np.ndarray  # (n_steps,) float32 in {0., 1.}
    aux: dict[str, np.ndarray]  # name -> (n_steps,) float32 in {0., 1.}
    scenario: str
    split: str  # "train" | "val" | "test" — from the shard manifest, v1

    @property
    def n_steps(self) -> int:
        return int(self.tokens.shape[0])


def _split_from_manifest(manifest: dict, sample_id: str, manifest_path: Path) -> str:
    """Look up a sample's contract split in its shard manifest.

    The v1 contract requires per-sample ``split`` in the manifest but the
    data PR defines the exact nesting; accept the plausible shapes and fail
    loudly (never default) when the assignment cannot be found.
    """
    samples = manifest.get("samples")
    if isinstance(samples, dict):
        record = samples.get(sample_id)
        if isinstance(record, dict) and "split" in record:
            split = record["split"]
        elif isinstance(record, str):
            split = record
        else:
            split = None
    elif isinstance(samples, list):
        split = None
        for record in samples:
            if isinstance(record, dict) and record.get("id", record.get("sample_id")) == sample_id:
                split = record.get("split")
                break
    else:
        split = manifest.get(sample_id) if isinstance(manifest, dict) else None
        if isinstance(split, dict):
            split = split.get("split")
    if split not in SPLITS:
        raise DatasetError(
            f"{manifest_path}: no valid split for sample {sample_id!r} "
            f"(expected one of {SPLITS})"
        )
    return str(split)


def _load_sample(parquet_path: Path) -> SampleData:
    frame = pd.read_parquet(parquet_path)
    missing = [column for column in LABEL_COLUMNS if column not in frame.columns]
    if missing:
        raise DatasetError(f"{parquet_path}: missing label columns {missing}")

    sample_id = parquet_path.name[: -len(".labels.parquet")]
    shard_dir = parquet_path.parent
    manifest_path = shard_dir / "manifest.json"
    if not manifest_path.exists():
        raise DatasetError(
            f"{parquet_path}: shard manifest {manifest_path} not found — the v1 "
            "contract requires a manifest with per-sample split assignment"
        )
    manifest = json.loads(manifest_path.read_text())

    token_path = shard_dir / f"{sample_id}.dmel.npy"
    if not token_path.exists():
        raise DatasetError(
            f"{parquet_path}: token cache {token_path.name} not found — run the "
            "dMel features precompute (python -m dmel.features ...) for this shard"
        )
    tokens = np.load(token_path)
    if tokens.ndim != 2:
        raise DatasetError(
            f"{token_path}: tokens must be (n_steps, tokens_per_step), got {tokens.shape}"
        )
    n_steps = len(frame)
    if tokens.shape[0] != n_steps:
        raise DatasetError(
            f"{parquet_path}: {n_steps} label rows but {tokens.shape[0]} token steps"
        )

    actions = frame["action"].map(ACTION_TO_INDEX)
    unknown = actions.isna()
    if unknown.any():
        bad = sorted(set(frame.loc[unknown, "action"]))
        raise DatasetError(f"{parquet_path}: unknown action values {bad}")
    aux = {
        name: frame[name].to_numpy(dtype=np.float32, copy=True)
        for name in AUX_LABEL_NAMES
    }
    for name, values in aux.items():
        if not np.isin(values, (0.0, 1.0)).all():
            raise DatasetError(f"{parquet_path}: {name} must be binary")

    return SampleData(
        sample_id=sample_id,
        path=parquet_path,
        tokens=tokens.astype(np.int64),
        action=actions.to_numpy(dtype=np.int64, copy=True),
        agent_speaking=frame["agent_speaking"].to_numpy(dtype=np.float32),
        aux=aux,
        scenario=str(frame["scenario"].iloc[0]),
        split=_split_from_manifest(manifest, sample_id, manifest_path),
    )


def scan_samples(root: str | Path) -> list[SampleData]:
    """Load every sample under ``root`` (sorted by path for determinism)."""
    root = Path(root)
    if not root.exists():
        raise DatasetError(f"data root does not exist: {root}")
    parquet_paths = sorted(root.glob("**/*.labels.parquet"))
    if not parquet_paths:
        raise DatasetError(f"no *.labels.parquet found under {root}")
    return [_load_sample(path) for path in parquet_paths]


def group_by_split(
    samples: list[SampleData],
) -> dict[str, list[SampleData]]:
    grouped: dict[str, list[SampleData]] = {split: [] for split in SPLITS}
    for sample in samples:
        grouped[sample.split].append(sample)
    return grouped


class WindowDataset(Dataset):
    """Fixed-length windows over contract shards, one split's samples.

    Validation mode enumerates deterministic non-overlapping full windows
    (tail partial windows dropped, documented). Train mode crops a fresh
    random window from the mapped sample on every __getitem__, seeded from
    the global torch RNG for run reproducibility.
    """

    def __init__(
        self,
        samples: list[SampleData],
        window_steps: int,
        *,
        train: bool,
        token_vocab_size: int | None = None,
    ) -> None:
        if window_steps < 1:
            raise ValueError("window_steps must be >= 1")
        self.window_steps = window_steps
        self.train = train
        self._by_id = {sample.sample_id: sample for sample in samples}
        tokens_per_step = {int(sample.tokens.shape[1]) for sample in samples}
        if len(tokens_per_step) != 1:
            raise DatasetError(
                f"inconsistent tokens_per_step across samples: {sorted(tokens_per_step)}"
            )
        self.tokens_per_step = tokens_per_step.pop()
        if token_vocab_size is not None:
            worst = max(int(sample.tokens.max()) for sample in samples)
            if worst >= token_vocab_size:
                raise DatasetError(
                    f"token id {worst} exceeds configured vocab {token_vocab_size}"
                )
        # Map each window index to (sample, deterministic start); train mode
        # replaces the start with a random crop at fetch time.
        self._windows: list[tuple[SampleData, int]] = []
        for sample in samples:
            if sample.n_steps < window_steps:
                continue
            starts = range(0, sample.n_steps - window_steps + 1, window_steps)
            for start in starts:
                self._windows.append((sample, start))

    def __len__(self) -> int:
        return len(self._windows)

    def __getitem__(self, index: int) -> dict:
        sample, start = self._windows[index]
        if self.train and sample.n_steps > self.window_steps:
            start = int(torch.randint(0, sample.n_steps - self.window_steps + 1, (1,)).item())
        stop = start + self.window_steps
        return {
            "tokens": torch.from_numpy(sample.tokens[start:stop]),
            "agent_speaking": torch.from_numpy(sample.agent_speaking[start:stop]),
            "action": torch.from_numpy(sample.action[start:stop]),
            **{
                name: torch.from_numpy(values[start:stop])
                for name, values in sample.aux.items()
            },
            "sample_id": sample.sample_id,
            "start_step": start,
        }


def balanced_action_weights(dataset: WindowDataset) -> torch.Tensor:
    """Inverse-frequency weights over window action classes.

    Windows containing >= 1 STOP_TTS frame form the minority class; this
    reweights the sampler so STOP windows appear ~as often as KEEP windows.
    Loss-level class weighting (pos_weight from train positive rates) is
    applied separately per the v1 contract.
    """
    stop_windows = 0
    for sample, start in dataset._windows:
        stop = int(sample.action[start : start + dataset.window_steps].sum())
        if stop > 0:
            stop_windows += 1
    total = len(dataset)
    keep_windows = total - stop_windows
    if stop_windows == 0 or keep_windows == 0:
        return torch.ones(total, dtype=torch.float64)
    weight_stop = total / (2.0 * stop_windows)
    weight_keep = total / (2.0 * keep_windows)
    weights = torch.empty(total, dtype=torch.float64)
    for index, (sample, start) in enumerate(dataset._windows):
        stop = int(sample.action[start : start + dataset.window_steps].sum())
        weights[index] = weight_stop if stop > 0 else weight_keep
    return weights
