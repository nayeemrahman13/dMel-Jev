"""Per-step policy latency and parameter counts vs the contract budgets.

Contract budgets:
  - one policy step (window recompute over <= 40 frames) < 10 ms on a
    CPU-class core
  - transformer arm ~6 layers x 256 hidden, 10-30 M params

Usage (from the repo root):
    python -m dmel.models.bench_inference --arms lstm transformer --assert-budget

For stable single-core numbers, pin the process (the budget is per core and
the sandbox CPU is shared): taskset -c 0 python -m dmel.models.bench_inference ...

The measurement pins torch to a single thread: the budget is per core.
Per-arm calibration_before/after columns flag CPU contention (sandbox
neighbors) — before/after drifting apart means the run was noisy.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import replace

import numpy as np
import torch

from dmel.models.config import DmelModelConfig
from dmel.models.model import build_model
from dmel.models.policy import CheckpointedBargeInPolicy

BUDGET_MS = 10.0


def machine_calibration_seconds(repeats: int = 20, size: int = 256) -> float:
    """Coarse single-core sanity number so results are interpretable."""
    a = np.random.default_rng(0).standard_normal((size, size)).astype(np.float32)
    b = np.random.default_rng(1).standard_normal((size, size)).astype(np.float32)
    start = time.perf_counter()
    for _ in range(repeats):
        a @ b
    return (time.perf_counter() - start) / repeats


def bench_policy(
    config: DmelModelConfig,
    *,
    n_steps: int = 300,
    warmup: int = 30,
    seed: int = 0,
) -> dict[str, float]:
    calibration_before = machine_calibration_seconds()
    model = build_model(config)
    policy = CheckpointedBargeInPolicy(model, config)
    rng = np.random.default_rng(seed)
    pcm = np.zeros(800, dtype=np.int16)
    tokens = rng.integers(
        0, config.features.token_vocab_size, size=config.features.tokens_per_step
    ).astype(np.int64)
    for _ in range(warmup):
        policy.step(pcm, tokens, agent_speaking=True)
    durations_ms: list[float] = []
    for _ in range(n_steps):
        start = time.perf_counter()
        policy.step(pcm, tokens, agent_speaking=True)
        durations_ms.append((time.perf_counter() - start) * 1000.0)
    durations = np.asarray(durations_ms)
    calibration_after = machine_calibration_seconds()
    return {
        "p50_ms": float(np.percentile(durations, 50)),
        "p95_ms": float(np.percentile(durations, 95)),
        "p99_ms": float(np.percentile(durations, 99)),
        "mean_ms": float(durations.mean()),
        # Contention probe: a mid-bench slowdown vs the pre-run calibration
        # flags CPU-noisy results (sandbox neighbors), which the reader must
        # weigh before comparing p95 against the budget.
        "calibration_before_s": calibration_before,
        "calibration_after_s": calibration_after,
        "params": model.num_parameters(),
        "window_steps": config.policy.context_steps,
    }


def bench_all(
    arms: list[str],
    config: DmelModelConfig,
    **bench_kwargs: float,
) -> dict[str, dict[str, float]]:
    results: dict[str, dict[str, float]] = {}
    for arm in arms:
        arm_config = DmelModelConfig(
            features=config.features,
            model=replace(config.model, arm=arm),
            policy=config.policy,
            loss_weights=config.loss_weights,
        )
        results[arm] = bench_policy(arm_config, **bench_kwargs)
    return results


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arms", nargs="+", default=["lstm", "transformer"])
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--assert-budget", action="store_true")
    parser.add_argument("--budget-ms", type=float, default=BUDGET_MS)
    args = parser.parse_args(argv)

    torch.set_num_threads(1)
    config = DmelModelConfig()
    calibration = machine_calibration_seconds()
    results = bench_all(args.arms, config, n_steps=args.steps)
    report = {
        "calibration_matmul_s": calibration,
        "budget_ms": args.budget_ms,
        "arms": results,
    }
    print(json.dumps(report, indent=2))

    if args.assert_budget:
        over = {
            arm: stats["p95_ms"]
            for arm, stats in results.items()
            if stats["p95_ms"] > args.budget_ms
        }
        if over:
            print(f"BUDGET EXCEEDED (p95 > {args.budget_ms} ms): {over}")
            return 1
        print(f"p95 within {args.budget_ms} ms budget for all arms")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
