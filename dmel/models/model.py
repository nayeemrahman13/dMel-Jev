"""dMel barge-in network: step embeddings -> causal backbone -> heads.

Input per 50 ms step: dMel token ids (from the features precompute) plus the
agent_speaking scalar (TTS playback state, legitimately available at
deployment per the contract). Optional ablation inputs — speaker similarity
(arm D) and agent-stem dMel tokens (arm E) — are wired end to end behind
config flags but disabled in V1 runs.

The model separates two entry points used by different callers:
  - ``forward``: vectorized over (B, T, ...) windows — training and tests.
  - ``step_embed`` + ``sequence_forward``: the streaming path. ``step_embed``
    is a pure per-step function, so the runtime policy memoizes it per step
    and only the backbone/heads are recomputed over the <=40-step window.
    Both paths share the same modules, so they produce identical values.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from dmel.models.config import DmelModelConfig, ModelConfig
from dmel.models.backbones import CausalTransformerBackbone, LSTMBackbone

CHECKPOINT_VERSION = 1
AUX_HEAD_NAMES = ("speech_present", "primary_user", "backchannel", "interrupt_intent")
ALL_HEAD_NAMES = ("action",) + AUX_HEAD_NAMES


class _MLPHead(nn.Module):
    def __init__(self, d_model: int, out_dim: int) -> None:
        super().__init__()
        self.hidden = nn.Linear(d_model, d_model // 2)
        self.out = nn.Linear(d_model // 2, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.out(torch.nn.functional.gelu(self.hidden(x)))


class DmelBargeInNet(nn.Module):
    """Learned barge-in policy network (arms B and C share this shape)."""

    def __init__(self, config: DmelModelConfig) -> None:
        super().__init__()
        self.config = config
        d_model = config.model.d_model

        self.token_embed = nn.Embedding(config.features.token_vocab_size, d_model)
        # Per-step token representation: mean-pool token embeddings, then mix.
        # A flattened (K*D -> D) projection scales with tokens_per_step — at
        # the per-bin geometry (400 tokens/step) it alone is ~26M params and
        # ~2x the per-step latency budget — while the pooled form is O(d_model).
        self.step_proj = nn.Linear(d_model, d_model)
        self.agent_speaking_embed = nn.Embedding(2, d_model)

        # Arm D plumbing: zero-initialized so enabling the flag mid-research
        # starts as a no-op rather than a random perturbation.
        self.use_speaker_similarity = config.model.use_speaker_similarity
        if self.use_speaker_similarity:
            self.speaker_proj = nn.Linear(1, d_model)
            nn.init.zeros_(self.speaker_proj.weight)
            nn.init.zeros_(self.speaker_proj.bias)

        # Arm E plumbing: second token stream from the agent stem.
        self.use_agent_stem_dmel = config.model.use_agent_stem_dmel
        if self.use_agent_stem_dmel:
            self.agent_token_embed = nn.Embedding(config.features.token_vocab_size, d_model)
            self.agent_step_proj = nn.Linear(d_model, d_model)

        self.backbone = self._build_backbone(config.model)
        self.action_head = _MLPHead(d_model, 2)
        self.aux_heads = nn.ModuleDict(
            {name: _MLPHead(d_model, 1) for name in AUX_HEAD_NAMES}
        )

    @staticmethod
    def _build_backbone(model: ModelConfig) -> nn.Module:
        if model.arm == "lstm":
            return LSTMBackbone(model.d_model, model.lstm_layers, model.dropout)
        if model.arm == "transformer":
            return CausalTransformerBackbone(
                d_model=model.d_model,
                nhead=model.nhead,
                num_layers=model.transformer_layers,
                dim_feedforward=model.dim_feedforward,
                dropout=model.dropout,
            )
        raise ValueError(f"unknown arm {model.arm!r}")

    def _embed_steps(
        self,
        token_ids: torch.Tensor,
        agent_speaking: torch.Tensor,
        agent_token_ids: torch.Tensor | None,
        speaker_similarity: torch.Tensor | None,
    ) -> torch.Tensor:
        """token_ids (..., T, K) int64, agent_speaking (..., T) bool -> (..., T, D)."""
        embedded = self.token_embed(token_ids)
        pooled = embedded.mean(dim=-2)
        x = self.step_proj(pooled)
        x = x + self.agent_speaking_embed(agent_speaking.long())
        if self.use_speaker_similarity:
            if speaker_similarity is None:
                similarity = torch.zeros(
                    x.shape[:-1], device=x.device, dtype=x.dtype
                )
            else:
                similarity = speaker_similarity.to(dtype=x.dtype)
            x = x + self.speaker_proj(similarity.unsqueeze(-1))
        if self.use_agent_stem_dmel:
            if agent_token_ids is None:
                agent_x = torch.zeros_like(x)
            else:
                agent_x = self.agent_step_proj(
                    self.agent_token_embed(agent_token_ids).mean(dim=-2)
                )
            x = x + agent_x
        return x

    def step_embed(
        self,
        token_ids: torch.Tensor,
        agent_speaking: torch.Tensor,
        agent_token_ids: torch.Tensor | None = None,
        speaker_similarity: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Embed a single step: token_ids (B, K), agent_speaking (B,) -> (B, D)."""
        return self._embed_steps(
            token_ids.unsqueeze(1),
            agent_speaking.unsqueeze(1),
            None if agent_token_ids is None else agent_token_ids.unsqueeze(1),
            None if speaker_similarity is None else speaker_similarity.unsqueeze(1),
        ).squeeze(1)

    def sequence_forward(self, embeddings: torch.Tensor) -> dict[str, torch.Tensor]:
        """Backbone + heads over precomputed step embeddings (B, T, D)."""
        hidden = self.backbone(embeddings)
        outputs: dict[str, torch.Tensor] = {"action": self.action_head(hidden)}
        for name, head in self.aux_heads.items():
            outputs[name] = head(hidden).squeeze(-1)
        return outputs

    def forward(
        self,
        token_ids: torch.Tensor,
        agent_speaking: torch.Tensor,
        agent_token_ids: torch.Tensor | None = None,
        speaker_similarity: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """token_ids (B, T, K) int64, agent_speaking (B, T) bool -> head logits.

        Returns one entry per head: "action" is (B, T, 2) logits; each aux
        head is (B, T) logits.
        """
        embeddings = self._embed_steps(
            token_ids, agent_speaking, agent_token_ids, speaker_similarity
        )
        return self.sequence_forward(embeddings)

    @torch.no_grad()
    def num_parameters(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())


def build_model(config: DmelModelConfig) -> DmelBargeInNet:
    return DmelBargeInNet(config)


def save_checkpoint(
    model: DmelBargeInNet,
    path: str | Path,
    *,
    extra: dict[str, Any] | None = None,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "checkpoint_version": CHECKPOINT_VERSION,
        "config": model.config.to_dict(),
        "model_state": model.state_dict(),
    }
    if extra:
        payload.update(extra)
    torch.save(payload, path)


def load_model_from_checkpoint(
    path: str | Path,
    map_location: str | torch.device = "cpu",
) -> tuple[DmelBargeInNet, dict[str, Any]]:
    """Rebuild a model from a self-describing checkpoint."""
    payload = torch.load(path, map_location=map_location, weights_only=False)
    version = payload.get("checkpoint_version")
    if version != CHECKPOINT_VERSION:
        raise ValueError(f"unsupported checkpoint version {version!r} in {path}")
    config = DmelModelConfig.from_dict(payload["config"])
    model = build_model(config)
    model.load_state_dict(payload["model_state"])
    model.eval()
    model_state_extra = {k: v for k, v in payload.items() if k not in ("model_state", "config", "checkpoint_version")}
    return model, model_state_extra


def action_probs_from_logits(logits: torch.Tensor) -> torch.Tensor:
    """(B, T, 2) or (T, 2) action logits -> stop probability (...,)."""
    return torch.softmax(logits, dim=-1)[..., 1]


def aux_probs_from_logits(logits: torch.Tensor) -> np.ndarray:
    return torch.sigmoid(logits).cpu().numpy()
