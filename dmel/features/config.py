"""Configuration for the dMel feature pipeline (dmel/features area).

Parameters follow the dMel-Jev POC data & interface contract v0: 16 kHz
mono int16 audio, log-mel with a 25 ms window / 10 ms hop, grouped into 50 ms
policy steps (5 subframes per step).

Token id layout (see dmel/features/tokenizer.py): one id per
(bin-group, level) with ``id = group_index * n_levels + level``. For the
default config (80 bins, 4 bits, 1 bin per group) the value vocabulary is
80 * 16 = 1280 and each 50 ms step carries 5 * 80 = 400 ids.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

import yaml

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent / "configs" / "default.yaml"


def _optional_float(value: Any) -> float | None:
    return None if value is None else float(value)


# YAML/config-file values arrive untyped; coerce each field explicitly so a
# quoted "25" or a float like 50.0 cannot slip through into validation.
_COERCIONS: dict[str, Any] = {
    "sample_rate_hz": int,
    "window_ms": float,
    "hop_ms": float,
    "step_ms": int,
    "n_mels": int,
    "n_fft": int,
    "fmin_hz": float,
    "fmax_hz": _optional_float,
    "window": str,
    "quant_bits": int,
    "mel_min_db": float,
    "mel_max_db": float,
    "log_eps": float,
    "bins_per_group": int,
}


@dataclass(frozen=True)
class DmelConfig:
    """Config-driven parameters for dMel tokenization.

    All fields are validated in ``__post_init__``; invalid combinations raise
    ``ValueError`` at construction time so misconfiguration can never reach
    the tokenizer. The dataclass is frozen and hashable, so it can key caches
    (the mel filterbank is memoized per config).
    """

    # Analysis grid: mel frames (subframes) are window_ms long, hop_ms apart;
    # step_ms groups whole subframes into one policy step.
    sample_rate_hz: int = 16_000
    window_ms: float = 25.0
    hop_ms: float = 10.0
    step_ms: int = 50

    # Mel filterbank / STFT.
    n_mels: int = 80
    n_fft: int = 512
    fmin_hz: float = 0.0
    fmax_hz: float | None = None  # None -> Nyquist
    window: str = "hann"

    # Intensity quantization: one shared linear codebook over log-mel dB
    # (dMel eq. 2-4). The range is pinned in config instead of derived from
    # dataset statistics so a step's tokens never depend on future audio.
    quant_bits: int = 4
    mel_min_db: float = -70.0
    mel_max_db: float = 50.0
    log_eps: float = 1e-10

    # Bins per token group: 1 -> one token id per mel bin (paper-faithful).
    bins_per_group: int = 1

    def __post_init__(self) -> None:
        if self.sample_rate_hz <= 0:
            raise ValueError(f"sample_rate_hz must be > 0, got {self.sample_rate_hz}")
        for name in ("window_ms", "hop_ms", "step_ms"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be > 0, got {getattr(self, name)}")
        if self.step_ms % self.hop_ms != 0:
            raise ValueError(
                f"step_ms ({self.step_ms}) must be a whole multiple of hop_ms ({self.hop_ms})"
            )
        if self.window_samples < 2:
            raise ValueError(f"window_ms resolves to {self.window_samples} samples; need >= 2")
        if self.hop_samples < 1:
            raise ValueError(f"hop_ms resolves to {self.hop_samples} samples; need >= 1")
        if self.n_mels <= 0:
            raise ValueError(f"n_mels must be > 0, got {self.n_mels}")
        if self.n_fft < self.window_samples:
            raise ValueError(
                f"n_fft ({self.n_fft}) must be >= window_samples ({self.window_samples})"
            )
        if self.fmin_hz < 0:
            raise ValueError(f"fmin_hz must be >= 0, got {self.fmin_hz}")
        nyquist = self.sample_rate_hz / 2.0
        if self.fmax_hz is not None and not self.fmin_hz < self.fmax_hz <= nyquist:
            raise ValueError(
                f"fmax_hz must be in ({self.fmin_hz}, {nyquist}], got {self.fmax_hz}"
            )
        if self.window != "hann":
            raise ValueError(f"unsupported window {self.window!r}; only 'hann' is implemented")
        if not 1 <= self.quant_bits <= 16:
            raise ValueError(f"quant_bits must be in [1, 16], got {self.quant_bits}")
        if self.mel_min_db >= self.mel_max_db:
            raise ValueError(
                f"mel_min_db ({self.mel_min_db}) must be < mel_max_db ({self.mel_max_db})"
            )
        if self.log_eps <= 0:
            raise ValueError(f"log_eps must be > 0, got {self.log_eps}")
        if not 1 <= self.bins_per_group <= self.n_mels:
            raise ValueError(
                f"bins_per_group must be in [1, {self.n_mels}], got {self.bins_per_group}"
            )
        if self.n_mels % self.bins_per_group != 0:
            raise ValueError(
                f"n_mels ({self.n_mels}) must be a whole multiple of "
                f"bins_per_group ({self.bins_per_group})"
            )

    # -- derived sizes (all exact integer arithmetic) ------------------------

    @property
    def window_samples(self) -> int:
        return round(self.window_ms * self.sample_rate_hz / 1000.0)

    @property
    def hop_samples(self) -> int:
        return round(self.hop_ms * self.sample_rate_hz / 1000.0)

    @property
    def step_samples(self) -> int:
        return round(self.step_ms * self.sample_rate_hz / 1000.0)

    @property
    def subframes_per_step(self) -> int:
        return int(self.step_ms // self.hop_ms)

    @property
    def n_levels(self) -> int:
        return 2**self.quant_bits

    @property
    def n_groups(self) -> int:
        return self.n_mels // self.bins_per_group

    @property
    def tokens_per_step(self) -> int:
        return self.subframes_per_step * self.n_groups

    @property
    def vocab_size(self) -> int:
        return self.n_groups * self.n_levels

    # -- (de)serialization ---------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        return {f.name: getattr(self, f.name) for f in fields(self)}

    @classmethod
    def from_mapping(cls, raw: dict[str, Any]) -> DmelConfig:
        known = {f.name for f in fields(cls)}
        unknown = sorted(set(raw) - known)
        if unknown:
            raise ValueError(f"unknown DmelConfig keys: {unknown}")
        return cls(**{name: _COERCIONS[name](raw[name]) for name in raw})

    @classmethod
    def from_yaml(cls, path: str | Path) -> DmelConfig:
        raw = yaml.safe_load(Path(path).read_text())
        if raw is None:
            raw = {}
        if not isinstance(raw, dict):
            raise TypeError(f"{path}: expected a mapping at the top level, got {type(raw)!r}")
        return cls.from_mapping(raw)
