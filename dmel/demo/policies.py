"""Arm resolution for the demo: the two interruption brains behind the toggle.

- ``--arm=c`` — the LEARNED policy: the capacity sweep's 6×256 transformer
  (4,115,846 params), loaded from the committed checkpoint via
  ``dmel.models.policy.CheckpointedBargeInPolicy`` (torch, lazy import).
- ``--arm=a`` — the TYPICAL S2S barge-in baseline: Silero VAD + speech-duration
  threshold (200 ms default), ``dmel.baselines.policy.SileroVADPolicy``
  (onnxruntime; the ONNX is vendored in-repo).

Both satisfy the contract ``BargeInPolicy`` interface, so the demo loop, the
recorder, and the tests treat them interchangeably. Neither arm's own module
is modified — this module only instantiates them.
"""

from __future__ import annotations

from pathlib import Path

ARM_LEARNED = "c"
ARM_BASELINE = "a"
ARM_CHOICES = (ARM_BASELINE, ARM_LEARNED)

# Committed checkpoint (see checkpoints/PROVENANCE.md for provenance).
CHECKPOINT_PATH = Path(__file__).resolve().parent / "checkpoints" / "w256_seed0_best.pt"

# Expected identity of the committed checkpoint — validated on every load so a
# corrupt or swapped file fails the demo (and the tests) loudly, not silently.
EXPECTED_PARAMS = 4_115_846  # 6×256 transformer, per docs/capacity_sweep.md
EXPECTED_ARM = "transformer"
EXPECTED_D_MODEL = 256

ARM_DISPLAY = {
    ARM_LEARNED: "learned 6x256 transformer (capacity-sweep w256 seed 0)",
    ARM_BASELINE: "Silero VAD + duration threshold (typical S2S barge-in)",
}


def resolve_arm(name: str) -> str:
    """Normalize an arm argument; accepts c/learned and a/baseline spellings."""
    lowered = name.strip().lower()
    aliases = {
        ARM_LEARNED: ARM_LEARNED,
        "learned": ARM_LEARNED,
        ARM_BASELINE: ARM_BASELINE,
        "baseline": ARM_BASELINE,
    }
    if lowered not in aliases:
        raise ValueError(f"unknown arm {name!r}; expected one of {', '.join(ARM_CHOICES)}")
    return aliases[lowered]


def describe_arm(arm: str) -> str:
    return ARM_DISPLAY[resolve_arm(arm)]


def build_policy(
    arm: str,
    checkpoint_path: Path | None = None,
    baseline_threshold_ms: float | None = None,
):
    """Instantiate one arm's contract policy, fresh (never shared between runs).

    Raises FileNotFoundError with the recovery command when the committed
    checkpoint is missing, so a broken checkout fails with a fix instead of a
    traceback into torch internals.
    """
    arm = resolve_arm(arm)
    if arm == ARM_LEARNED:
        path = Path(checkpoint_path) if checkpoint_path is not None else CHECKPOINT_PATH
        if not path.is_file():
            raise FileNotFoundError(
                f"learned-policy checkpoint not found at {path} — run "
                "`python -m dmel.demo.recover_checkpoint` to fetch or retrain it "
                "(see dmel/demo/checkpoints/PROVENANCE.md)"
            )
        # Lazy import: arm-A-only users don't pay for torch.
        from dmel.models.policy import CheckpointedBargeInPolicy

        policy = CheckpointedBargeInPolicy.from_checkpoint(path)
        _validate_learned_checkpoint(policy, path)
        return policy
    from dmel.baselines.policy import SileroVADPolicy

    if baseline_threshold_ms is not None:
        return SileroVADPolicy(stop_threshold_ms=baseline_threshold_ms)
    return SileroVADPolicy()


def _validate_learned_checkpoint(policy, path: Path) -> None:
    config = policy.config
    actual = policy.model.num_parameters()
    if (
        config.model.arm != EXPECTED_ARM
        or config.model.d_model != EXPECTED_D_MODEL
        or actual != EXPECTED_PARAMS
    ):
        raise ValueError(
            f"checkpoint {path} is not the expected 6x256 capacity-sweep model "
            f"(arm={config.model.arm}, d_model={config.model.d_model}, params={actual}; "
            f"expected {EXPECTED_ARM}/{EXPECTED_D_MODEL}/{EXPECTED_PARAMS})"
        )
