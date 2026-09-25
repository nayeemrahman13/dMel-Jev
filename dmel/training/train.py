"""Training entry point for the learned barge-in policies (arms B and C).

Usage (from the repo root):
    python -m dmel.training.train --config dmel/training/config.yaml --arm transformer

Contract v1 conformance:
  - split assignment is the shard manifest ``split`` column: training uses
    train, model/threshold selection uses val; the test split is never
    touched here (eval-harness territory, final report only).
  - per-head BCE pos_weight from train-split positive rates, capped at 20;
    the summary reports per-head train positive rates.
  - every learned arm trains >= 3 seeds; the summary aggregates mean +- sd.

Config layout: the features/model/policy/loss_weights sections form the
model's DmelModelConfig (embedded in every checkpoint); the data/optim/out
sections drive this loop only. AdamW, per-epoch validation, best+last
checkpoints per seed, and loss curves written to json.
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

from dmel.models.config import DmelModelConfig
from dmel.models.model import AUX_HEAD_NAMES, DmelBargeInNet, save_checkpoint
from dmel.training.dataset import (
    SampleData,
    WindowDataset,
    balanced_action_weights,
    group_by_split,
    scan_samples,
)
from dmel.training.losses import compute_loss, pos_weight_from_rates


@dataclass(frozen=True)
class DataConfig:
    root: str = "data/pilot"
    window_steps: int = 40
    sampler: str = "balanced_stop"  # "balanced_stop" | "uniform"


@dataclass(frozen=True)
class OptimConfig:
    seeds: int = 3  # contract v1: every learned arm trains >= 3 seeds
    epochs: int = 5
    batch_size: int = 32
    lr: float = 3e-4
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    num_workers: int = 2


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _read_yaml(path: Path) -> dict:
    import yaml

    with open(path, encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle) or {}
    return loaded


def _load_training_config(path: Path) -> tuple[DmelModelConfig, DataConfig, OptimConfig, Path]:
    raw = _read_yaml(path)
    model_config = DmelModelConfig.from_dict(raw)
    data = DataConfig(**raw.get("data", {}))
    optim = OptimConfig(**raw.get("optim", {}))
    out_dir = Path(raw.get("out_dir", "runs/dmel"))
    return model_config, data, optim, out_dir


def train_positive_rates(train_samples: list[SampleData]) -> dict[str, float]:
    """Per-head positive-frame rates over the train split (for the summary
    and the pos_weight derivation)."""
    frames = sum(sample.n_steps for sample in train_samples)
    if frames == 0:
        raise ValueError("train split is empty")
    rates = {
        "action_stop": float(sum(int(sample.action.sum()) for sample in train_samples) / frames)
    }
    for name in AUX_HEAD_NAMES:
        rates[name] = float(
            sum(float(sample.aux[name].sum()) for sample in train_samples) / frames
        )
    return rates


def evaluate(
    model: DmelBargeInNet,
    loader: DataLoader,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    totals: dict[str, float] = {}
    frames = 0
    correct = 0
    with torch.inference_mode():
        for batch in loader:
            tokens = batch["tokens"].to(device)
            agent_speaking = batch["agent_speaking"].to(device)
            labels = {
                "action": batch["action"].to(device),
                "speech_present": batch["speech_present"].to(device),
                "primary_user": batch["primary_user"].to(device),
                "backchannel": batch["backchannel"].to(device),
                "interrupt_intent": batch["interrupt_intent"].to(device),
            }
            outputs = model(tokens, agent_speaking)
            _, parts = compute_loss(outputs, labels, model.config.loss_weights)
            for name, value in parts.items():
                totals[name] = totals.get(name, 0.0) + value
            predicted = outputs["action"].argmax(dim=-1)
            correct += int((predicted == labels["action"]).sum().item())
            frames += int(labels["action"].numel())
    batches = max(1, len(loader))
    results = {name: value / batches for name, value in totals.items()}
    results["action_accuracy"] = correct / max(1, frames)
    return results


def balanced_accuracy(model: DmelBargeInNet, loader: DataLoader, device: torch.device) -> float:
    """Mean of per-class recall on the action head (chance level 0.5)."""
    model.eval()
    per_class_hits = [0, 0]
    per_class_total = [0, 0]
    with torch.inference_mode():
        for batch in loader:
            outputs = model(batch["tokens"].to(device), batch["agent_speaking"].to(device))
            predicted = outputs["action"].argmax(dim=-1)
            actual = batch["action"]
            for value in (0, 1):
                mask = actual == value
                per_class_total[value] += int(mask.sum().item())
                per_class_hits[value] += int((predicted[mask] == value).sum().item())
    recalls = [
        hits / total for hits, total in zip(per_class_hits, per_class_total) if total > 0
    ]
    return sum(recalls) / len(recalls) if recalls else 0.0


def train_one_seed(
    model_config: DmelModelConfig,
    data_config: DataConfig,
    optim_config: OptimConfig,
    out_dir: Path,
    seed: int,
    train_samples: list[SampleData],
    val_samples: list[SampleData],
    *,
    device: torch.device,
    max_train_batches: int | None = None,
    num_workers: int | None = None,
) -> dict:
    set_seed(seed)
    train_dataset = WindowDataset(
        train_samples,
        data_config.window_steps,
        train=True,
        token_vocab_size=model_config.features.token_vocab_size,
    )
    val_dataset = WindowDataset(
        val_samples,
        data_config.window_steps,
        train=False,
        token_vocab_size=model_config.features.token_vocab_size,
    )

    workers = optim_config.num_workers if num_workers is None else num_workers
    if data_config.sampler == "balanced_stop":
        weights = balanced_action_weights(train_dataset)
        sampler: WeightedRandomSampler | None = WeightedRandomSampler(
            weights, num_samples=len(train_dataset), replacement=True
        )
        shuffle = False
    elif data_config.sampler == "uniform":
        sampler = None
        shuffle = True
    else:
        raise ValueError(f"unknown sampler {data_config.sampler!r}")
    train_loader = DataLoader(
        train_dataset,
        batch_size=optim_config.batch_size,
        sampler=sampler,
        shuffle=shuffle,
        num_workers=workers,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=optim_config.batch_size,
        shuffle=False,
        num_workers=workers,
    )

    model = DmelBargeInNet(model_config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=optim_config.lr,
        weight_decay=optim_config.weight_decay,
    )
    # Contract v1: per-head pos_weight from train-split positive rates.
    pos_weight = pos_weight_from_rates(train_positive_rates(train_samples))

    arm_dir = out_dir / model_config.model.arm
    seed_dir = arm_dir / f"seed_{seed}"
    seed_dir.mkdir(parents=True, exist_ok=True)
    curves: dict[str, list[dict]] = {"train": [], "val": []}
    best_val = float("inf")

    for epoch in range(optim_config.epochs):
        model.train()
        for batch_index, batch in enumerate(train_loader):
            tokens = batch["tokens"].to(device)
            agent_speaking = batch["agent_speaking"].to(device)
            labels = {
                "action": batch["action"].to(device),
                "speech_present": batch["speech_present"].to(device),
                "primary_user": batch["primary_user"].to(device),
                "backchannel": batch["backchannel"].to(device),
                "interrupt_intent": batch["interrupt_intent"].to(device),
            }
            outputs = model(tokens, agent_speaking)
            loss, parts = compute_loss(
                outputs, labels, model_config.loss_weights, pos_weight
            )
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), optim_config.grad_clip)
            optimizer.step()
            if batch_index % 20 == 0:
                print(
                    f"seed {seed} epoch {epoch} batch {batch_index}/{len(train_loader)} "
                    f"loss {parts['total']:.4f} (action {parts['action']:.4f})",
                    flush=True,
                )
            curves["train"].append({"epoch": epoch, "batch": batch_index, **parts})
            if max_train_batches is not None and batch_index + 1 >= max_train_batches:
                break

        val_parts = evaluate(model, val_loader, device)
        curves["val"].append({"epoch": epoch, **val_parts})
        print(f"seed {seed} epoch {epoch} val {val_parts}", flush=True)

        save_checkpoint(model, seed_dir / "last.pt", extra={"epoch": epoch, "val": val_parts})
        if val_parts["total"] < best_val:
            best_val = val_parts["total"]
            save_checkpoint(model, seed_dir / "best.pt", extra={"epoch": epoch, "val": val_parts})
        (seed_dir / "loss_curves.json").write_text(json.dumps(curves, indent=2))

    return {
        "seed": seed,
        "params": model.num_parameters(),
        "best_val_total": best_val,
        "final_val": curves["val"][-1] if curves["val"] else {},
        "seed_dir": str(seed_dir),
    }


def train(
    model_config: DmelModelConfig,
    data_config: DataConfig,
    optim_config: OptimConfig,
    out_dir: Path,
    *,
    device: torch.device | None = None,
    max_train_batches: int | None = None,
    num_workers: int | None = None,
) -> dict:
    device = device or torch.device("cpu")
    grouped = group_by_split(scan_samples(data_config.root))
    train_samples, val_samples = grouped["train"], grouped["val"]
    if not train_samples or not val_samples:
        raise ValueError(
            "manifest splits must yield non-empty train and val sets "
            f"(train={len(train_samples)}, val={len(val_samples)}, "
            f"test={len(grouped['test'])}); the test split is intentionally "
            "not consumed here — it belongs to the eval harness"
        )

    pos_weight = pos_weight_from_rates(train_positive_rates(train_samples))
    seed_summaries = [
        train_one_seed(
            model_config,
            data_config,
            optim_config,
            out_dir,
            seed,
            train_samples,
            val_samples,
            device=device,
            max_train_batches=max_train_batches,
            num_workers=num_workers,
        )
        for seed in range(optim_config.seeds)
    ]

    val_totals = [entry["best_val_total"] for entry in seed_summaries]
    summary = {
        "arm": model_config.model.arm,
        "data_root": data_config.root,
        "train_samples": len(train_samples),
        "val_samples": len(val_samples),
        "test_samples_untouched": len(grouped["test"]),
        "train_positive_rates": train_positive_rates(train_samples),
        "pos_weight": pos_weight,
        "seeds": seed_summaries,
        "best_val_total_mean": statistics.fmean(val_totals),
        "best_val_total_sd": (
            statistics.stdev(val_totals) if len(val_totals) > 1 else 0.0
        ),
        "out_dir": str(out_dir / model_config.model.arm),
    }
    (out_dir / model_config.model.arm / "summary.json").write_text(
        json.dumps(summary, indent=2)
    )
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="dmel/training/config.yaml")
    parser.add_argument("--arm", choices=("lstm", "transformer"), help="override config arm")
    parser.add_argument("--data-root", help="override config data root")
    parser.add_argument("--device", default=None, help="cpu or cuda")
    parser.add_argument("--seeds", type=int, help="override config seeds")
    parser.add_argument("--epochs", type=int, help="override config epochs")
    parser.add_argument("--max-train-batches", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    args = parser.parse_args(argv)

    model_config, data_config, optim_config, out_dir = _load_training_config(Path(args.config))
    if args.arm:
        model_config = DmelModelConfig(
            features=model_config.features,
            model=replace(model_config.model, arm=args.arm),
            policy=model_config.policy,
            loss_weights=model_config.loss_weights,
        )
    if args.data_root:
        data_config = replace(data_config, root=args.data_root)
    if args.seeds is not None:
        optim_config = replace(optim_config, seeds=args.seeds)
    if args.epochs is not None:
        optim_config = replace(optim_config, epochs=args.epochs)

    device = (
        torch.device(args.device)
        if args.device
        else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )
    summary = train(
        model_config,
        data_config,
        optim_config,
        out_dir,
        device=device,
        max_train_batches=args.max_train_batches,
        num_workers=args.num_workers,
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
