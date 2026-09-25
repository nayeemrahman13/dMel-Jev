"""Contract interface tests: BargeInPolicy conformance, frozen probs schema,
and the documented flag-only fallback."""

import numpy as np
import pytest

from dmel.models.config import DmelModelConfig
from dmel.models.model import DmelBargeInNet
from dmel.models.policy import ACTION_KEEP, ACTION_STOP, PROBS_SCHEMA, CheckpointedBargeInPolicy

FRAME = 800


def make_policy(arm: str = "lstm", latch: bool = True) -> CheckpointedBargeInPolicy:
    base = DmelModelConfig()
    config = DmelModelConfig(
        features=base.features,
        model=type(base.model)(arm=arm, d_model=32, lstm_layers=2, transformer_layers=2,
                               nhead=4, dim_feedforward=64, dropout=0.0),
        policy=type(base.policy)(context_steps=8, stop_threshold=0.5, latch=latch),
        loss_weights=base.loss_weights,
    )
    return CheckpointedBargeInPolicy(DmelBargeInNet(config), config)


def frame() -> np.ndarray:
    return np.zeros(FRAME, dtype=np.int16)


def tokens_step(rng: np.random.Generator, vocab: int | None = None) -> np.ndarray:
    """One step of token ids with the model-default token geometry."""
    features = DmelModelConfig().features
    vocab = vocab if vocab is not None else features.token_vocab_size
    return rng.integers(0, vocab, size=(features.tokens_per_step,))


def test_step_contract_shapes_and_schema():
    policy = make_policy()
    rng = np.random.default_rng(0)
    for step in range(5):
        result = policy.step(
            frame(), tokens_step(rng), agent_speaking=True
        )
        assert set(result) == {"action", "probs", "t_ms"}
        assert result["action"] in (ACTION_KEEP, ACTION_STOP)
        assert set(result["probs"]) == set(PROBS_SCHEMA)  # frozen, no extra keys
        assert all(isinstance(v, float) for v in result["probs"].values())
        assert result["t_ms"] == step * 50
    policy.reset()
    first = policy.step(frame(), tokens_step(rng), agent_speaking=True)
    assert first["t_ms"] == 0  # reset clears the step counter


@pytest.mark.parametrize("arm", ["lstm", "transformer"])
def test_latch_holds_stop_until_agent_speaking_drops(arm):
    policy = make_policy(arm)
    rng = np.random.default_rng(1)
    # Force the latch through the internal state the way a threshold hit would.
    policy._latched = True
    result = policy.step(frame(), tokens_step(rng), agent_speaking=True)
    assert result["action"] == ACTION_STOP
    result = policy.step(frame(), tokens_step(rng), agent_speaking=True)
    assert result["action"] == ACTION_STOP
    result = policy.step(frame(), tokens_step(rng), agent_speaking=False)
    assert result["action"] == ACTION_KEEP  # agent_speaking low clears the latch
    assert policy._latched is False


def test_flag_only_fallback_never_initiates_stop():
    """dmel_token_ids=None must follow the documented flag-only rule:
    no new stops, latch sustained only while agent_speaking is true."""
    policy = make_policy()
    rng = np.random.default_rng(2)
    for step in range(4):
        result = policy.step(frame(), None, agent_speaking=True)
        assert result["action"] == ACTION_KEEP
        assert set(result["probs"]) == set(PROBS_SCHEMA)

    # A latched policy sustains STOP on cache misses while the agent speaks,
    # and returns to KEEP when it goes low.
    policy._latched = True
    assert policy.step(frame(), None, agent_speaking=True)["action"] == ACTION_STOP
    assert policy.step(frame(), None, agent_speaking=False)["action"] == ACTION_KEEP


def test_window_rolls_at_context_limit():
    # The context window is a rolling deque (contract: last 40 steps), so
    # running past the limit must neither raise nor grow the window.
    policy = make_policy()  # context_steps=8
    rng = np.random.default_rng(3)
    for step in range(20):
        result = policy.step(frame(), tokens_step(rng), agent_speaking=True)
        assert result["t_ms"] == step * 50
        assert result["action"] in (ACTION_KEEP, ACTION_STOP)
    assert len(policy._window) == policy.config.policy.context_steps
    result = policy.step(frame(), None, agent_speaking=False)
    assert result["action"] == ACTION_KEEP


@pytest.mark.parametrize("arm", ["lstm", "transformer"])
def test_jit_traced_policy_matches_eager(arm):
    """The traced fast path (jit_inference=True, deployment default) must
    reproduce eager inference: identical actions and probabilities on
    identical weights over a short streaming rollout."""
    import torch

    base = DmelModelConfig()
    model_kwargs = dict(arm=arm, d_model=32, lstm_layers=2, transformer_layers=2,
                        nhead=4, dim_feedforward=64, dropout=0.0)
    rng = np.random.default_rng(4)
    token_seq = [tokens_step(rng) for _ in range(4)]

    results = {}
    for jit_flag in (False, True):
        torch.manual_seed(11)  # identical weights for both policies
        config = DmelModelConfig(
            features=base.features,
            model=type(base.model)(**model_kwargs),
            policy=type(base.policy)(context_steps=8, stop_threshold=0.5,
                                     latch=True, jit_inference=jit_flag),
            loss_weights=base.loss_weights,
        )
        policy = CheckpointedBargeInPolicy(DmelBargeInNet(config), config)
        if jit_flag and policy._traced_forward is None:
            pytest.skip("torch.jit tracing unavailable in this environment")
        results[jit_flag] = [
            policy.step(frame(), tokens, agent_speaking=True) for tokens in token_seq
        ]

    for eager_step, traced_step in zip(results[False], results[True]):
        assert eager_step["action"] == traced_step["action"]
        assert eager_step["t_ms"] == traced_step["t_ms"]
        assert set(traced_step["probs"]) == set(PROBS_SCHEMA)
        assert all(
            abs(eager_step["probs"][key] - traced_step["probs"][key]) < 1e-6
            for key in PROBS_SCHEMA
        )


def test_pcm_validation():
    policy = make_policy()
    with pytest.raises(ValueError):
        policy.step(np.zeros(FRAME, dtype=np.float32), None, agent_speaking=True)
    with pytest.raises(ValueError):
        policy.step(np.zeros(FRAME - 1, dtype=np.int16), None, agent_speaking=True)
