"""Configuration for the dMel learned barge-in policies (arms B and C).

The same frozen dataclasses serve training and runtime: a checkpoint embeds
its full config, so loading a policy never depends on a parallel runtime
config staying in sync. Values are loaded from dmel/training/config.yaml.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any

import yaml


class ConfigError(ValueError):
    """Raised when a config is internally inconsistent or incomplete."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ConfigError(message)


@dataclass(frozen=True)
class FeaturesConfig:
    """Must match the dmel/features precompute config that produced the caches.

    Defaults mirror dmel/features/configs/default.yaml: 80 mel bins with 1
    bin per bin-group (the contract's per-bin quantization), 4-bit intensity
    (16 levels), and 5 mel subframes per 50 ms step, so a step carries
    5 x 80 = 400 token ids drawn from an 80 x 16 = 1280-token vocabulary.
    The dataset loader validates cache shapes against these values and
    fails loudly on mismatch.
    """

    token_vocab_size: int = 1280
    tokens_per_step: int = 400


@dataclass(frozen=True)
class ModelConfig:
    arm: str = "transformer"  # "lstm" (arm B) | "transformer" (arm C)
    d_model: int = 256
    lstm_layers: int = 2
    transformer_layers: int = 6
    nhead: int = 8
    dim_feedforward: int = 640
    dropout: float = 0.1
    # Ablation plumbing (arms D/E): wired end to end but disabled in V1 runs.
    use_speaker_similarity: bool = False
    use_agent_stem_dmel: bool = False


@dataclass(frozen=True)
class PolicyConfig:
    context_steps: int = 40  # 40 x 50 ms = 2 s rolling window
    stop_threshold: float = 0.5
    latch: bool = True  # hold STOP_TTS until agent_speaking goes low
    # Trace+freeze the sequence forward for the fixed deployment shape; falls
    # back to eager (with a warning) if tracing fails.
    jit_inference: bool = True


@dataclass(frozen=True)
class LossWeights:
    """Per-head loss weights; the contract fixes action as dominant."""

    action: float = 1.0
    interrupt_intent: float = 0.3
    backchannel: float = 0.2
    speech_present: float = 0.1
    primary_user: float = 0.1

    def validate(self) -> None:
        _require(self.action > 0.0, "action weight must be positive")
        aux = {
            "interrupt_intent": self.interrupt_intent,
            "backchannel": self.backchannel,
            "speech_present": self.speech_present,
            "primary_user": self.primary_user,
        }
        for name, weight in aux.items():
            _require(
                weight >= 0.0,
                f"loss weight for {name} must be non-negative, got {weight}",
            )
            _require(
                self.action > weight,
                f"action weight ({self.action}) must dominate {name} ({weight})",
            )


@dataclass(frozen=True)
class DmelModelConfig:
    features: FeaturesConfig = FeaturesConfig()
    model: ModelConfig = ModelConfig()
    policy: PolicyConfig = PolicyConfig()
    loss_weights: LossWeights = LossWeights()

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        _require(
            self.model.arm in ("lstm", "transformer"),
            f"arm must be 'lstm' or 'transformer', got {self.model.arm!r}",
        )
        _require(self.features.token_vocab_size > 0, "token_vocab_size must be positive")
        _require(self.features.tokens_per_step > 0, "tokens_per_step must be positive")
        _require(self.policy.context_steps > 0, "context_steps must be positive")
        _require(
            0.0 < self.policy.stop_threshold < 1.0,
            f"stop_threshold must be in (0, 1), got {self.policy.stop_threshold}",
        )
        _require(self.model.d_model > 0, "d_model must be positive")
        _require(
            self.model.d_model % self.model.nhead == 0,
            f"d_model ({self.model.d_model}) must be divisible by nhead ({self.model.nhead})",
        )
        self.loss_weights.validate()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "DmelModelConfig":
        sections = {}
        for section, dataclass_type in (
            ("features", FeaturesConfig),
            ("model", ModelConfig),
            ("policy", PolicyConfig),
            ("loss_weights", LossWeights),
        ):
            value = raw.get(section, {})
            known = {f.name for f in fields(dataclass_type)}
            unknown = sorted(set(value) - known)
            _require(
                not unknown,
                f"unknown {section} keys: {unknown}",
            )
            sections[section] = dataclass_type(**value)
        return cls(**sections)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "DmelModelConfig":
        with open(path, encoding="utf-8") as handle:
            raw = yaml.safe_load(handle) or {}
        return cls.from_dict(raw)
