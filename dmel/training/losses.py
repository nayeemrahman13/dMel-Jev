"""Multi-task loss for the learned barge-in policies.

Contract: L_action + 0.3*L_interrupt + 0.2*L_backchannel + 0.1*L_speech +
0.1*L_primary_user, weights config-driven with action strictly dominant.
Per-head BCE pos_weight comes from train-split positive rates (capped at 20)
— computed once over the train split and passed in, never fitted.
Pure function: tensors in, (total, per-part) out — no state, no I/O.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from dmel.models.config import LossWeights
from dmel.models.model import AUX_HEAD_NAMES

POS_WEIGHT_CAP = 20.0


def compute_loss(
    outputs: dict[str, torch.Tensor],
    labels: dict[str, torch.Tensor],
    weights: LossWeights,
    pos_weight: dict[str, float] | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Weighted multi-task loss.

    outputs: "action" (B, T, 2) logits; aux heads (B, T) logits.
    labels: "action" (B, T) int64 in {0, 1}; aux heads (B, T) in {0., 1.}.
    pos_weight: optional per-aux-head BCE pos_weight (train-split derived,
        capped at POS_WEIGHT_CAP by the caller).

    Returns (total scalar tensor, {head: float loss}) for logging.
    """
    parts: dict[str, torch.Tensor] = {}
    action_logits = outputs["action"]
    action_labels = labels["action"]
    parts["action"] = F.cross_entropy(
        action_logits.reshape(-1, action_logits.size(-1)),
        action_labels.reshape(-1).long(),
    )
    for name in AUX_HEAD_NAMES:
        head_pos_weight = None
        if pos_weight is not None and name in pos_weight:
            head_pos_weight = torch.tensor(
                [pos_weight[name]], dtype=outputs[name].dtype, device=outputs[name].device
            )
        parts[name] = F.binary_cross_entropy_with_logits(
            outputs[name].reshape(-1).float(),
            labels[name].reshape(-1).float(),
            pos_weight=head_pos_weight,
        )

    total = weights.action * parts["action"]
    for name in AUX_HEAD_NAMES:
        total = total + getattr(weights, name) * parts[name]

    detached = {name: float(value.detach().item()) for name, value in parts.items()}
    detached["total"] = float(total.detach().item())
    return total, detached


def pos_weight_from_rates(
    positive_rates: dict[str, float],
    cap: float = POS_WEIGHT_CAP,
) -> dict[str, float]:
    """neg/pos per head, capped at ``cap`` (contract: capped at 20)."""
    capped: dict[str, float] = {}
    for name, rate in positive_rates.items():
        if not 0.0 < rate < 1.0:
            raise ValueError(
                f"{name}: train positive rate {rate} is degenerate; pos_weight "
                "needs both classes present in the train split"
            )
        capped[name] = min((1.0 - rate) / rate, cap)
    return capped
