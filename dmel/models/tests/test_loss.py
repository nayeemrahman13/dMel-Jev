"""Loss weighting tests: action dominance and train-derived pos_weight."""

import math

import pytest
import torch

from dmel.models.config import DmelModelConfig, LossWeights
from dmel.models.model import AUX_HEAD_NAMES
from dmel.training.losses import POS_WEIGHT_CAP, compute_loss, pos_weight_from_rates


def make_outputs_labels(batch: int = 2, steps: int = 5, stop_every: int = 3):
    torch.manual_seed(0)
    outputs = {"action": torch.randn(batch, steps, 2)}
    for name in AUX_HEAD_NAMES:
        outputs[name] = torch.randn(batch, steps)
    labels = {
        "action": torch.tensor([[1 if (b * steps + t) % stop_every == 0 else 0
                                 for t in range(steps)] for b in range(batch)])
    }
    for name in AUX_HEAD_NAMES:
        labels[name] = (labels["action"] == 1).float() * 0.8 + 0.1
    return outputs, labels


def test_action_weight_dominates_total():
    outputs, labels = make_outputs_labels()
    weights = LossWeights()  # 1.0 action, 0.3/0.2/0.1/0.1 aux
    total, parts = compute_loss(outputs, labels, weights)
    dominated = weights.action * parts["action"] + sum(
        getattr(weights, name) * parts[name] for name in AUX_HEAD_NAMES
    )
    assert math.isclose(total, dominated, rel_tol=1e-6)
    # Strictly: action term alone exceeds the largest weighted aux term.
    assert parts["action"] > 0.3 * parts["interrupt_intent"]


def test_pos_weight_scales_imbalanced_head_loss():
    outputs, labels = make_outputs_labels()
    weights = LossWeights()
    _total_unweighted, unweighted = compute_loss(outputs, labels, weights)
    pos_weight = {name: 5.0 for name in AUX_HEAD_NAMES}
    _total_weighted, weighted = compute_loss(outputs, labels, weights, pos_weight)
    # Action CE is untouched by pos_weight; aux losses grow.
    assert math.isclose(unweighted["action"], weighted["action"], rel_tol=1e-6)
    for name in AUX_HEAD_NAMES:
        assert weighted[name] > unweighted[name], name


def test_pos_weight_from_rates_caps_and_validates():
    rates = {"speech_present": 0.01, "primary_user": 1e-6, "backchannel": 0.5,
             "interrupt_intent": 0.2}
    capped = pos_weight_from_rates(rates)
    assert capped["primary_user"] == POS_WEIGHT_CAP  # neg/pos ~1e6 -> capped at 20
    assert math.isclose(capped["speech_present"], 99.0) or capped["speech_present"] == POS_WEIGHT_CAP
    assert capped["speech_present"] == min(99.0, POS_WEIGHT_CAP)
    assert math.isclose(capped["backchannel"], 1.0)
    with pytest.raises(ValueError):
        pos_weight_from_rates({"speech_present": 0.0})
    with pytest.raises(ValueError):
        pos_weight_from_rates({"speech_present": 1.0})
