"""Modal GPU runner for dMel barge-in policy training (arms B and C).

Modal patterns: image builders, @app.cls with @modal.enter-style lifecycle,
region pinning, Volume mounts.

Local GPU smoke (no data upload needed — deterministic synthetic shards are
generated on the worker):
    modal run dmel/training/train_modal.py --arm transformer --epochs 1

Real runs upload contract shards (produced by
`python -m dmel.data.generate --preset ...`) to the dmel-jev-data Volume,
e.g. `modal volume put dmel-jev-data <shard-dir> /pilot/<shard-dir>`, then:
    modal run dmel/training/train_modal.py --arm transformer --epochs 10 \
        --no-generate-synthetic --data-root /data/pilot
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import modal

_here = Path(__file__).resolve()
# Local checkout: dmel/training/train_modal.py → repo root is parents[2].
# Modal copies this module to /root/train_modal.py, which has no parents[2].
ROOT = _here.parents[2] if len(_here.parents) >= 3 else _here.parent

REGION = "us-east"
MINUTES = 60
SYNTHETIC_ROOT = "/data/dmel-synthetic"

train_image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "torch==2.9.0",  # CUDA-enabled build on Linux; L4 needs no extra index
        "numpy==2.2.4",
        "pandas==2.3.3",
        "pyarrow==21.0.0",
        "pyyaml==6.0.3",
    )
    .env({"PYTHONPATH": "/pkg"})
    .add_local_dir(str(ROOT / "dmel"), remote_path="/pkg/dmel")
)

runs_volume = modal.Volume.from_name("dmel-jev-runs", create_if_missing=True)
data_volume = modal.Volume.from_name("dmel-jev-data", create_if_missing=True)

app = modal.App("dmel-jev-training", image=train_image)


@app.cls(
    gpu="L4",
    region=REGION,
    volumes={"/runs": runs_volume, "/data": data_volume},
    # Pilot-scale ceiling (cache materialization + 5 epochs x 3 seeds), not a
    # reservation — Modal bills actual usage. A hung run burns at most this.
    timeout=150 * MINUTES,
)
class DmelTrainer:
    @modal.method()
    def train(
        self,
        arm: str,
        epochs: int,
        seeds: int,
        data_root: str,
        batch_size: int,
        lr: float,
        max_train_batches: int | None,
        out_dir: str,
        generate_synthetic: bool,
        d_model: int = 256,
        nhead: int = 8,
        dim_feedforward: int = 640,
        seed: int | None = None,
    ) -> dict:
        import sys

        sys.path.insert(0, "/pkg")
        from dmel.models.config import DmelModelConfig
        from dmel.training.dataset import DatasetError
        from dmel.training.features_cache import ensure_token_caches
        from dmel.training.synthetic import scaffold_features, write_synthetic_shard
        from dmel.training.train import DataConfig, OptimConfig, train as run_training

        if generate_synthetic:
            write_synthetic_shard(Path(SYNTHETIC_ROOT), n_samples=16, n_steps=64, seed=13)
            data_root = SYNTHETIC_ROOT
        elif not Path(data_root).exists():
            raise DatasetError(
                f"data root {data_root!r} not found on the worker — upload shards to "
                "the dmel-jev-data Volume or run with --generate-synthetic"
            )
        else:
            # Real shards ship pcm + labels; materialize any missing dMel
            # token caches on the worker (deterministic, skips existing).
            ensure_token_caches(Path(data_root))

        base = DmelModelConfig()
        # Synthetic shards carry the scaffold's tiny token geometry — the
        # model must be built to match. Contract shards use the defaults.
        features = (
            scaffold_features(base.features) if generate_synthetic else base.features
        )
        summary = run_training(
            DmelModelConfig(
                features=features,
                # Capacity sweep: width and its necessary scalars only
                # (heads keep head_dim = d_model/nhead; FFN keeps the 2.5x
                # ratio; step_proj scales with d_model automatically). The
                # frontend (vocab 1280, 400 tokens/step) never changes.
                model=replace(
                    base.model,
                    arm=arm,
                    d_model=d_model,
                    nhead=nhead,
                    dim_feedforward=dim_feedforward,
                ),
                policy=base.policy,
                loss_weights=base.loss_weights,
            ),
            DataConfig(root=data_root),
            OptimConfig(epochs=epochs, seeds=seeds, batch_size=batch_size, lr=lr, num_workers=2),
            Path(out_dir),
            max_train_batches=max_train_batches,
            seeds_list=None if seed is None else [seed],
        )
        return summary


@app.local_entrypoint()
def main(
    arm: str = "transformer",
    epochs: int = 1,
    seeds: int = 3,  # v1: learned arms train >= 3 seeds; GPU smoke uses --seeds 1
    data_root: str = "/data/pilot",
    generate_synthetic: bool = True,
    batch_size: int = 32,
    lr: float = 3e-4,
    max_train_batches: int | None = None,
    out_dir: str = "/runs/smoke",
    d_model: int = 256,
    nhead: int = 8,
    dim_feedforward: int = 640,
    seed: int | None = None,  # set to train ONE (width, seed) for the capacity sweep
) -> None:
    trainer = DmelTrainer()
    summary = trainer.train.remote(
        arm=arm,
        epochs=epochs,
        seeds=seeds,
        data_root=data_root,
        batch_size=batch_size,
        lr=lr,
        max_train_batches=max_train_batches,
        out_dir=out_dir,
        generate_synthetic=generate_synthetic,
        d_model=d_model,
        nhead=nhead,
        dim_feedforward=dim_feedforward,
        seed=seed,
    )
    print(json.dumps(summary, indent=2))
