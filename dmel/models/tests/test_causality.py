"""Causality tests: padding a future frame must not change past outputs."""

import pytest
import torch

from dmel.models.config import DmelModelConfig
from dmel.models.model import AUX_HEAD_NAMES, DmelBargeInNet


def tiny_config(arm: str) -> DmelModelConfig:
    base = DmelModelConfig()
    return DmelModelConfig(
        features=base.features,
        model=type(base.model)(arm=arm, d_model=32, lstm_layers=2, transformer_layers=2,
                               nhead=4, dim_feedforward=64, dropout=0.0),
        policy=base.policy,
        loss_weights=base.loss_weights,
    )


@pytest.mark.parametrize("arm", ["lstm", "transformer"])
def test_future_frame_cannot_change_past_outputs(arm):
    torch.manual_seed(0)
    model = DmelBargeInNet(tiny_config(arm))
    model.eval()
    steps = 10
    future = 7  # outputs at steps < future must be invariant
    tokens = torch.randint(0, model.config.features.token_vocab_size, (1, steps, model.config.features.tokens_per_step))
    agent_speaking = torch.randint(0, 2, (1, steps))

    tokens_future = tokens.clone()
    tokens_future[:, future:] = torch.randint(
        0, model.config.features.token_vocab_size, (1, steps - future, model.config.features.tokens_per_step)
    )
    agent_future = agent_speaking.clone()
    agent_future[:, future:] = 1 - agent_future[:, future:]

    with torch.inference_mode():
        base_out = model(tokens, agent_speaking)
        padded_out = model(tokens_future, agent_future)

    assert torch.equal(base_out["action"][:, :future], padded_out["action"][:, :future])
    for name in AUX_HEAD_NAMES:
        assert torch.equal(base_out[name][:, :future], padded_out[name][:, :future]), name
