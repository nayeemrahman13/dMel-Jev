"""Runtime barge-in policy for the learned arms (B and C).

This module owns the dmel/models copy of the contract's BargeInPolicy
interface; signatures must match the contract exactly. The learned
implementation loads a self-describing checkpoint and runs streaming
inference over the last ``context_steps`` (40) steps.

Streaming cost: ``step_embed`` is a pure per-step function, so the policy
memoizes the projected embedding of each new step and recomputes only
backbone + heads over the <=40 cached embeddings per call — the contract's
"window recompute over <=40 frames" budget. Results are identical to a full
recompute because nothing else depends on step order.

Flag-only fallback (documented per contract v1): when ``dmel_token_ids`` is
None (feature cache unavailable mid-stream), the policy never initiates a
new stop and never re-decides from stale probabilities — it only sustains an
existing STOP_TTS latch while ``agent_speaking`` stays true, and emits KEEP
otherwise. The eval harness runs this path on the fixture set.

``probs`` schema is frozen (contract v1): p_stop, p_interrupt, p_backchannel,
p_speech, p_primary_user — no extra keys.
"""

from __future__ import annotations

import warnings
from abc import ABC, abstractmethod
from collections import deque
from pathlib import Path

import numpy as np
import torch

from dmel.models.config import DmelModelConfig
from dmel.models.model import DmelBargeInNet, load_model_from_checkpoint

ACTION_KEEP = "KEEP"
ACTION_STOP = "STOP_TTS"
FRAME_SAMPLES = 800  # 50 ms at 16 kHz
STEP_MS = 50

# Frozen probs schema (contract v1). Order matters only for readability;
# the key set is exactly these five.
PROBS_SCHEMA = (
    "p_stop",
    "p_interrupt",
    "p_backchannel",
    "p_speech",
    "p_primary_user",
)
# Aux head name -> frozen probs key.
_AUX_PROBS_KEYS = {
    "interrupt_intent": "p_interrupt",
    "backchannel": "p_backchannel",
    "speech_present": "p_speech",
    "primary_user": "p_primary_user",
}
_AUX_HEAD_ORDER = tuple(_AUX_PROBS_KEYS)


class _TracedSequenceForward(torch.nn.Module):
    """Trace wrapper for ``sequence_forward``: jit.trace cannot capture a dict
    return, so the traced callable yields (action, aux_tuple) instead."""

    def __init__(self, model: DmelBargeInNet) -> None:
        super().__init__()
        self.model = model

    def forward(self, embeddings: torch.Tensor) -> tuple:  # type: ignore[override]
        outputs = self.model.sequence_forward(embeddings)
        return outputs["action"], tuple(outputs[name] for name in _AUX_HEAD_ORDER)


class BargeInPolicy(ABC):
    """Contract interface: called once per 50 ms while the agent speaks."""

    @abstractmethod
    def reset(self) -> None: ...

    @abstractmethod
    def step(
        self,
        user_pcm_frame: "np.ndarray",  # 800 int16 samples
        dmel_token_ids: "np.ndarray | None",
        *,
        agent_speaking: bool,
        agent_pcm_frame: "np.ndarray | None" = None,
        speaker_similarity: "float | None" = None,
    ) -> dict:
        # returns {"action": "KEEP"|"STOP_TTS", "probs": dict, "t_ms": int}
        ...


class CheckpointedBargeInPolicy(BargeInPolicy):
    """Learned policy loaded from a training checkpoint."""

    def __init__(self, model: DmelBargeInNet, config: DmelModelConfig) -> None:
        self.config = config
        self.model = model
        self.model.eval()
        self._window: deque[torch.Tensor] = deque(maxlen=config.policy.context_steps)
        self._traced_forward = (
            self._build_traced_forward() if config.policy.jit_inference else None
        )
        self._reset_state()

    def _build_traced_forward(self) -> "torch.jit.ScriptModule | None":
        """Trace+freeze the sequence forward for the fixed (1, context_steps,
        d_model) deployment shape — roughly a third off eager per-step latency.
        Trace failure must degrade to eager, never crash policy load."""
        try:
            device = next(self.model.parameters()).device
            dummy = torch.zeros(
                (1, self.config.policy.context_steps, self.config.model.d_model),
                device=device,
            )
            with warnings.catch_warnings():
                # torch.jit.freeze is deprecated but still the fastest frozen-CPU
                # path for this fixed-shape workload on our torch build.
                warnings.simplefilter("ignore", FutureWarning)
                traced = torch.jit.trace(
                    _TracedSequenceForward(self.model).eval(), (dummy,), check_trace=False
                )
                return torch.jit.freeze(traced.eval())
        except Exception as exc:  # surfaced as a warning; eager inference is correct, just slower
            warnings.warn(f"jit trace build failed; using eager inference: {exc}")
            return None

    @classmethod
    def from_checkpoint(
        cls,
        path: str | Path,
        map_location: str | torch.device = "cpu",
    ) -> "CheckpointedBargeInPolicy":
        model, _ = load_model_from_checkpoint(path, map_location=map_location)
        return cls(model, model.config)

    def _reset_state(self) -> None:
        self._steps_seen = 0
        self._latched = False
        self._last_probs: dict[str, float] = {key: 0.0 for key in PROBS_SCHEMA}

    def _latch_action(self, agent_speaking: bool) -> str:
        """Action for a step that does not decide from probabilities (flag-only
        fallback, or an already-latched step): STOP only while a latch is held
        AND the agent is still speaking; otherwise KEEP (clearing the latch)."""
        if self._latched and agent_speaking:
            return ACTION_STOP
        self._latched = False
        return ACTION_KEEP

    def reset(self) -> None:
        self._reset_state()

    def _validate_pcm(self, pcm: np.ndarray, name: str) -> None:
        if not isinstance(pcm, np.ndarray):
            raise ValueError(f"{name} must be a np.ndarray, got {type(pcm)}")
        if pcm.shape != (FRAME_SAMPLES,):
            raise ValueError(
                f"{name} must be {FRAME_SAMPLES} samples (50 ms), got shape {pcm.shape}"
            )
        if pcm.dtype != np.int16:
            raise ValueError(f"{name} must be int16 PCM, got {pcm.dtype}")

    def _sequence_outputs(self, embeddings: torch.Tensor) -> dict[str, torch.Tensor]:
        if self._traced_forward is not None:
            action, aux = self._traced_forward(embeddings)
            outputs = {"action": action}
            outputs.update(zip(_AUX_HEAD_ORDER, aux))
            return outputs
        return self.model.sequence_forward(embeddings)

    def _run_model(self, agent_speaking: bool) -> dict[str, float]:
        embeddings = torch.stack(list(self._window)).unsqueeze(0)
        with torch.inference_mode():
            outputs = self._sequence_outputs(embeddings)
        action_logits = outputs["action"][0, -1]
        stop_prob = float(torch.softmax(action_logits, dim=-1)[1].item())
        probs = {"p_stop": stop_prob}
        for name, key in _AUX_PROBS_KEYS.items():
            probs[key] = float(torch.sigmoid(outputs[name][0, -1]).item())
        return probs

    def step(
        self,
        user_pcm_frame: "np.ndarray",
        dmel_token_ids: "np.ndarray | None",
        *,
        agent_speaking: bool,
        agent_pcm_frame: "np.ndarray | None" = None,
        speaker_similarity: "float | None" = None,
    ) -> dict:
        self._validate_pcm(user_pcm_frame, "user_pcm_frame")
        self._steps_seen += 1

        if dmel_token_ids is not None:
            expected = (self.config.features.tokens_per_step,)
            tokens = np.asarray(dmel_token_ids)
            if tokens.shape != expected:
                raise ValueError(
                    f"dmel_token_ids must have shape {expected}, got {tokens.shape}"
                )
            token_tensor = torch.from_numpy(tokens.astype(np.int64)).unsqueeze(0)
            flag_tensor = torch.tensor([agent_speaking], dtype=torch.long)
            agent_tokens = None
            if self.config.model.use_agent_stem_dmel:
                if agent_pcm_frame is None:
                    raise ValueError(
                        "use_agent_stem_dmel is enabled: agent_pcm_frame is required"
                    )
                self._validate_pcm(agent_pcm_frame, "agent_pcm_frame")
                # Arm E feature extraction is not wired in V1; the stream is
                # represented by zeros until that ablation is trained.
                agent_tokens = torch.zeros(
                    (1, self.config.features.tokens_per_step), dtype=torch.long
                )
            similarity = None
            if self.config.model.use_speaker_similarity:
                similarity = torch.tensor(
                    [0.0 if speaker_similarity is None else float(speaker_similarity)],
                    dtype=torch.float32,
                )
            with torch.inference_mode():
                embedding = self.model.step_embed(
                    token_tensor, flag_tensor, agent_tokens, similarity
                )
            self._window.append(embedding[0])
            self._last_probs = self._run_model(agent_speaking)
            if self._latched:
                action = self._latch_action(agent_speaking)
            else:
                stop_prob = self._last_probs["p_stop"]
                action = (
                    ACTION_STOP
                    if stop_prob >= self.config.policy.stop_threshold
                    else ACTION_KEEP
                )
                if action == ACTION_STOP and self.config.policy.latch:
                    self._latched = True
        else:
            # Flag-only fallback: never initiate a new stop from stale
            # probabilities; only sustain the latch while agent_speaking is true.
            action = self._latch_action(agent_speaking)

        return {
            "action": action,
            "probs": dict(self._last_probs),
            "t_ms": (self._steps_seen - 1) * STEP_MS,
        }


def load_policy(
    checkpoint_path: str | Path,
    map_location: str | torch.device = "cpu",
) -> CheckpointedBargeInPolicy:
    """Convenience entry point for the runtime integration task."""
    return CheckpointedBargeInPolicy.from_checkpoint(checkpoint_path, map_location)
