"""Shuffled-label control (contract v1, Splits guardrails).

Training on permuted action labels must yield chance-level fit — catches any
pipeline path where labels (or label-derived information) reach the model
inputs. The randomized-labels test compares TRAIN fit: a leak makes permuted
labels as fittable as true ones (train balanced accuracy ~1.0), while a
leak-free pipeline cannot fit permuted labels (~0.5). Training runs on the
stochastic window sampler so a positional model cannot pass by rote
memorization. Runs on synthetic scaffolding; the data PR's fixtures train
the same control against real contract data.
"""

import dataclasses

import numpy as np
import pytest
import torch

from dmel.models.config import DmelModelConfig
from dmel.models.model import DmelBargeInNet
from dmel.training.dataset import WindowDataset, group_by_split, scan_samples
from dmel.training.losses import compute_loss, pos_weight_from_rates
from dmel.training.synthetic import write_synthetic_shard
from dmel.training.train import train_positive_rates

STEPS = 200
BATCH = 8
LR = 1e-3
CHANCE_CEILING = 0.65  # permuted train fit must stay near chance (0.5)
LEARNED_FLOOR = 0.9  # true train fit must be near-perfect


def _tiny_config(arm: str) -> DmelModelConfig:
    base = DmelModelConfig()
    return DmelModelConfig(
        features=type(base.features)(token_vocab_size=48, tokens_per_step=4),
        model=type(base.model)(arm=arm, d_model=32, lstm_layers=1, transformer_layers=2,
                               nhead=4, dim_feedforward=64, dropout=0.0),
        policy=base.policy,
        loss_weights=base.loss_weights,
    )


def _permute_action(sample, rng: np.random.Generator):
    return dataclasses.replace(sample, action=rng.permutation(sample.action))


def _balanced_accuracy_on(model, dataset, indices, device):
    model.eval()
    hits = [0, 0]
    total = [0, 0]
    with torch.inference_mode():
        for index in indices:
            item = dataset[index]
            outputs = model(
                item["tokens"].unsqueeze(0).to(device),
                item["agent_speaking"].unsqueeze(0).to(device),
            )
            predicted = outputs["action"][0].argmax(dim=-1).cpu()
            actual = item["action"]
            for value in (0, 1):
                mask = actual == value
                total[value] += int(mask.sum())
                hits[value] += int((predicted[mask] == value).sum())
    recalls = [h / t for h, t in zip(hits, total) if t > 0]
    return sum(recalls) / len(recalls)


def _train(model, dataset, indices, steps, lr, device, seed, pos_weight):
    """Train on random-offset windows (train=True) evaluated nowhere — the
    stochastic sampler is part of the control: with deterministic windows a
    positional transformer can memorize permuted labels by rote."""
    torch.manual_seed(seed)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    model.train()
    rng = np.random.default_rng(seed)
    for _step in range(steps):
        items = [dataset[i] for i in rng.integers(0, len(indices), size=BATCH)]
        tokens = torch.stack([item["tokens"] for item in items]).to(device)
        agent_speaking = torch.stack(
            [item["agent_speaking"] for item in items]
        ).to(device)
        labels = {"action": torch.stack([item["action"] for item in items]).to(device)}
        for name in ("speech_present", "primary_user", "backchannel", "interrupt_intent"):
            labels[name] = torch.stack([item[name] for item in items]).to(device)
        outputs = model(tokens, agent_speaking)
        loss, _ = compute_loss(outputs, labels, model.config.loss_weights, pos_weight)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


@pytest.mark.parametrize("arm", ["lstm", "transformer"])
def test_shuffled_action_labels_give_chance_level_fit(arm, tmp_path):
    device = torch.device("cpu")
    shard = write_synthetic_shard(tmp_path, n_samples=16, n_steps=96, seed=5)
    grouped = group_by_split(scan_samples(shard))
    train_samples, val_samples = grouped["train"], grouped["val"]
    assert train_samples and val_samples

    config = _tiny_config(arm)
    # train=True: random window offsets during training (real sampler
    # behavior); train=False datasets are the deterministic eval views.
    train_fit_view = WindowDataset(train_samples, window_steps=32, train=False,
                                   token_vocab_size=config.features.token_vocab_size)
    train_stream = WindowDataset(train_samples, window_steps=32, train=True,
                                 token_vocab_size=config.features.token_vocab_size)
    train_indices = np.arange(len(train_fit_view))

    # Control: permute each sample's action column (fixed RNG), train, and
    # measure fit on the SAME permuted train windows. A leak would make the
    # permuted labels as fittable as the true ones.
    rng = np.random.default_rng(123)
    permuted_train = [_permute_action(sample, rng) for sample in train_samples]
    permuted_stream = WindowDataset(permuted_train, window_steps=32, train=True,
                                    token_vocab_size=config.features.token_vocab_size)

    model = DmelBargeInNet(config).to(device)
    _train(
        model, permuted_stream, train_indices, STEPS, lr=LR, device=device,
        seed=7, pos_weight=pos_weight_from_rates(train_positive_rates(permuted_train)),
    )
    control_fit = _balanced_accuracy_on(model, train_fit_view, train_indices, device)
    assert control_fit <= CHANCE_CEILING, (
        f"shuffled-label control failed: train fit {control_fit:.3f} > "
        f"{CHANCE_CEILING} — labels (or label-derived features) are leaking "
        "into the model inputs"
    )

    # Sanity: the identical loop on the true labels must fit them, so the
    # control cannot pass merely because the model or loop is broken.
    model2 = DmelBargeInNet(config).to(device)
    _train(
        model2, train_stream, train_indices, STEPS, lr=LR, device=device,
        seed=7, pos_weight=pos_weight_from_rates(train_positive_rates(train_samples)),
    )
    learned_fit = _balanced_accuracy_on(model2, train_fit_view, train_indices, device)
    assert learned_fit > LEARNED_FLOOR, f"sanity run failed to learn: {learned_fit:.3f}"
