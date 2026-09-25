"""CPU smoke: overfit a small batch and assert the loss decreases.

Runs before any GPU time is spent: if the loop cannot drive the loss down on
a tiny dataset, something is broken in the model/loss/data plumbing, not in
the scale.

Data: dmel/data/fixtures/ when the data-engine PR has merged (consumed via
the same loader as real shards); otherwise a deterministic synthetic shard
from dmel.training.synthetic (clearly not evidence about real data).

Usage (from the repo root):
    python -m dmel.training.smoke --arm transformer
"""

from __future__ import annotations

import argparse
import json
import tempfile
from dataclasses import replace
from pathlib import Path

import torch

from dmel.models.config import DmelModelConfig
from dmel.models.model import DmelBargeInNet
from dmel.training.dataset import WindowDataset, scan_samples
from dmel.training.features_cache import ensure_token_caches
from dmel.training.losses import compute_loss
from dmel.training.synthetic import (
    write_synthetic_shard,
    scaffold_features,
)
from dmel.training.train import set_seed

FIXTURES_ROOT = Path("dmel/data/fixtures")


def resolve_data_root(fixtures_root: Path | None) -> tuple[Path, str]:
    """Prefer contract fixtures; fall back to synthetic scaffolding."""
    root = fixtures_root or FIXTURES_ROOT
    if root.exists() and any(root.glob("**/*.labels.parquet")):
        return root, "fixtures"
    tmp = Path(tempfile.mkdtemp(prefix="dmel-smoke-"))
    write_synthetic_shard(tmp, n_samples=8, n_steps=64, seed=11)
    return tmp, "synthetic"


def run_smoke(
    arm: str,
    *,
    steps: int = 120,
    batch_size: int = 8,
    window_steps: int = 40,
    lr: float = 1e-3,
    min_improvement: float = 0.3,
    fixtures_root: Path | None = None,
    data_root: Path | None = None,
) -> dict:
    set_seed(3)
    resolved_root, source = (
        (data_root, "given") if data_root is not None else resolve_data_root(fixtures_root)
    )
    if source == "fixtures":
        # Committed fixtures lack token caches; materialize before loading.
        ensure_token_caches(resolved_root)
    base = DmelModelConfig()
    # The model's token geometry must match the data actually consumed:
    # contract fixtures use the FeaturesConfig defaults; the synthetic
    # scaffold uses its own tiny geometry.
    features = (
        base.features
        if source in ("fixtures", "given")
        else scaffold_features(base.features)
    )
    model_config = DmelModelConfig(
        features=features,
        model=replace(base.model, arm=arm),
        policy=base.policy,
        loss_weights=base.loss_weights,
    )
    dataset = WindowDataset(
        scan_samples(resolved_root),
        window_steps,
        train=True,
        token_vocab_size=model_config.features.token_vocab_size,
    )
    batch = torch.utils.data.default_collate(
        [dataset[i] for i in range(min(batch_size, len(dataset)))]
    )
    tokens = batch["tokens"]
    agent_speaking = batch["agent_speaking"]
    labels = {
        "action": batch["action"],
        "speech_present": batch["speech_present"],
        "primary_user": batch["primary_user"],
        "backchannel": batch["backchannel"],
        "interrupt_intent": batch["interrupt_intent"],
    }

    model = DmelBargeInNet(model_config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    model.train()
    curve: list[float] = []
    for step in range(steps):
        outputs = model(tokens, agent_speaking)
        loss, _parts = compute_loss(outputs, labels, model_config.loss_weights)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if step % 10 == 0 or step == steps - 1:
            curve.append(float(loss.detach().item()))

    first = curve[0]
    last = curve[-1]
    improvement = (first - last) / first if first > 0 else 0.0
    result = {
        "arm": arm,
        "data_source": source,
        "data_root": str(resolved_root),
        "steps": steps,
        "loss_curve": curve,
        "first_loss": first,
        "last_loss": last,
        "improvement": improvement,
        "required_improvement": min_improvement,
        "passed": bool(last < first and improvement >= min_improvement),
    }
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arms", nargs="+", default=["lstm", "transformer"])
    parser.add_argument("--steps", type=int, default=120)
    parser.add_argument("--fixtures-root", default=None)
    parser.add_argument("--min-improvement", type=float, default=0.3)
    args = parser.parse_args(argv)

    torch.set_num_threads(min(4, torch.get_num_threads()))
    fixtures_root = Path(args.fixtures_root) if args.fixtures_root else None
    results = []
    for arm in args.arms:
        result = run_smoke(
            arm,
            steps=args.steps,
            min_improvement=args.min_improvement,
            fixtures_root=fixtures_root,
        )
        results.append(result)
        out_dir = Path("runs/dmel/smoke")
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / f"{arm}.json").write_text(json.dumps(result, indent=2))
        print(f"[{arm}] loss {result['first_loss']:.4f} -> {result['last_loss']:.4f} "
              f"({result['improvement']:.0%} improvement, {result['data_source']} data) "
              f"{'PASS' if result['passed'] else 'FAIL'}")

    if not all(result["passed"] for result in results):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
