"""Guardrails for the model-side geometry documented in docs/dmel_geometry.md.

Asserts the embedding table, the mean pooling over the 400 per-step token
embeddings, the transformer structure, and the measured parameter count
against the live default config.
"""

from __future__ import annotations

import numpy as np
import torch

from dmel.models.config import DmelModelConfig
from dmel.models.model import DmelBargeInNet

# Per docs/dmel_geometry.md @ a543b54.
EXPECTED = {
    "token_vocab_size": 1280,
    "tokens_per_step": 400,
    "d_model": 256,
    "transformer_layers": 6,
    "nhead": 8,
    "dim_feedforward": 640,
    "context_steps": 40,
    "total_params": 4_115_846,  # measured; changes must update the doc
}


def default_model() -> DmelBargeInNet:
    torch.manual_seed(0)
    return DmelBargeInNet(DmelModelConfig())


def test_embedding_table_geometry():
    model = default_model()
    assert model.token_embed.num_embeddings == EXPECTED["token_vocab_size"]
    assert model.token_embed.embedding_dim == EXPECTED["d_model"]
    assert model.agent_speaking_embed.num_embeddings == 2
    assert model.agent_speaking_embed.embedding_dim == EXPECTED["d_model"]


def test_pooling_is_mean_over_tokens():
    """The step vector must be step_proj(mean(token embeddings)) + flag embedding."""
    model = default_model()
    model.eval()
    ids = torch.from_numpy(
        np.random.default_rng(0).integers(0, EXPECTED["token_vocab_size"], size=(2, EXPECTED["tokens_per_step"]))
    ).long()
    flag = torch.tensor([False, True])
    with torch.inference_mode():
        pooled = model.step_embed(ids, flag)
        expected = model.step_proj(model.token_embed(ids).mean(dim=-2)) + model.agent_speaking_embed(
            flag.long()
        )
    assert pooled.shape == (2, EXPECTED["d_model"])
    torch.testing.assert_close(pooled, expected)


def test_transformer_structure():
    model = default_model()
    assert model.config.model.arm == "transformer"
    assert model.backbone.encoder.num_layers == EXPECTED["transformer_layers"]
    layer = model.backbone.encoder.layers[0]
    assert layer.self_attn.embed_dim == EXPECTED["d_model"]
    assert layer.self_attn.num_heads == EXPECTED["nhead"]
    assert layer.linear1.out_features == EXPECTED["dim_feedforward"]
    assert model.config.policy.context_steps == EXPECTED["context_steps"]


def test_total_param_count():
    assert default_model().num_parameters() == EXPECTED["total_params"]
