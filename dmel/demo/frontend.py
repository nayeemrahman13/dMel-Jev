"""Streaming dMel frontend for the live demo: rolling 800-sample frames -> step tokens.

Wraps the frozen offline tokenizer (``dmel.features.tokenizer``) so the live
demo consumes exactly the token geometry the corpus was trained on
(``docs/dmel_geometry.md``): 400 ids per 50 ms step (5 mel subframes x 80
bins, shared 16-level codebook), computed with the SAME public functions the
offline precompute path uses (``log_mel_db`` -> ``quantize_levels`` ->
``tokens_from_levels``). Row-independent float64 ops on identical window
vectors give bit-identical ids — ``dmel/demo/tests`` asserts equality with
``tokenize()`` on whole clips and with the committed fixture caches.

Causality — no lookahead, no mid-stream fabrication
---------------------------------------------------
The offline geometry's 25 ms mel windows at 10 ms hops overlap 240 samples
(15 ms) past each step's own 800 samples: step t's last subframe window spans
samples ``[t*800 + 640, t*800 + 1040)``. The frontend therefore emits step t's
tokens as soon as samples through ``t*800 + 1040`` have arrived — in practice
one mic frame later, because frames arrive in whole 800-sample blocks. At
emission time every sample the tokens depend on has already been received;
the only zero-padding ever applied is the offline tokenizer's documented
end-of-clip edge padding, applied here only by :meth:`StreamingFeaturizer.flush`
at end of stream (and only over samples past the last pushed frame).

The one-frame emission lag is the frontend's, not the policy's: the decision
for step t still happens once per 50 ms tick, and the demo loop simply pairs
each emitted token step with the mic frame it belongs to (see
``dmel.demo.session``).

Deliberately NOT used: ``dmel.features.precompute`` (offline/cache-oriented
CLI — the streaming path must not depend on it).
"""

from __future__ import annotations

import numpy as np

from dmel.features.config import DmelConfig
from dmel.features.tokenizer import (
    log_mel_db,
    n_steps_for_samples,
    quantize_levels,
    tokens_from_levels,
)

FRAME_SAMPLES = 800  # 50 ms at 16 kHz — the contract's mic frame
STEP_WINDOW_SAMPLES = 1040  # samples step t's tokens depend on: t*800 + 1040


def validate_frame(frame: np.ndarray) -> np.ndarray:
    """Validate one contract mic frame (800 int16 samples)."""
    arr = np.asarray(frame)
    if arr.ndim != 1 or arr.shape[0] != FRAME_SAMPLES:
        raise ValueError(f"frame must be {FRAME_SAMPLES} samples (50 ms), got shape {arr.shape}")
    if arr.dtype != np.int16:
        raise ValueError(f"frame must be int16 PCM, got dtype {arr.dtype}")
    return arr


def step_tokens(samples: np.ndarray, config: DmelConfig) -> np.ndarray:
    """Token ids for exactly ONE 50 ms step from the step's sample slice.

    ``samples`` starts at the step's base and spans through the step's last
    subframe window (1040 samples mid-stream; zero-padded to 1040 at end of
    stream exactly like the offline tokenizer's edge padding). Uses the same
    public tokenizer functions as the whole-clip path, so identical samples
    give bit-identical ids (each subframe's row is an independent float64
    computation — see module docstring).
    """
    x = np.asarray(samples)
    if x.ndim != 1 or x.dtype != np.int16:
        raise ValueError(f"step slice must be 1-D int16, got shape {x.shape} dtype {x.dtype}")
    window = config.window_samples + (config.subframes_per_step - 1) * config.hop_samples
    if x.shape[0] < window:
        x = np.pad(x, (0, window - x.shape[0]))
    n_steps = n_steps_for_samples(x.shape[0], config)
    if n_steps != 1:
        raise ValueError(
            f"step slice must yield exactly 1 step at window {window} samples, got {n_steps}"
        )
    levels = quantize_levels(log_mel_db(x, config), config)
    return tokens_from_levels(levels, config)[0]


class StreamingFeaturizer:
    """Causal streaming frontend: push 800-sample int16 frames, emit 400-id steps.

    Emission lags one frame (see module docstring): after pushing frame ``i``
    the frontend returns the token ids for step ``i - 1``, or ``None`` when no
    step is complete yet. ``flush()`` emits the final step(s) with the
    offline tokenizer's end-of-clip zero-padding. Callers feed whole
    800-sample frames (mic reality; ``dmel.eval.labels.frame_view`` for file
    streams).
    """

    def __init__(self, config: DmelConfig | None = None) -> None:
        self.config = config if config is not None else DmelConfig()
        # The demo's streaming loop and the training corpus share one geometry;
        # anything else would silently feed the policy a different frontend.
        if (self.config.step_samples, self.config.tokens_per_step) != (FRAME_SAMPLES, 400):
            raise ValueError(
                "live demo requires the default 50 ms / 80-bin / 4-bit dMel geometry "
                f"(800-sample steps, 400 ids), got step_samples={self.config.step_samples}, "
                f"tokens_per_step={self.config.tokens_per_step}"
            )
        self._buffer = np.zeros(0, dtype=np.int16)  # samples from the next step's base onward
        self._next_step = 0
        self._samples_seen = 0

    @property
    def next_step(self) -> int:
        """Index of the step the next emitted tokens will carry."""
        return self._next_step

    @property
    def samples_seen(self) -> int:
        """Total mic samples pushed so far (frames are never held back)."""
        return self._samples_seen

    def push_frame(self, frame: np.ndarray) -> np.ndarray | None:
        """Push one 800-sample int16 frame; return the completed step's tokens or None.

        At most one step can complete per push (a step needs 1040 samples of
        buffer and each push adds exactly 800), so the return is a single
        step's tokens or None while the frontend is still waiting on the
        next frame.
        """
        arr = validate_frame(frame)
        self._buffer = np.concatenate([self._buffer, arr]) if self._buffer.size else arr.copy()
        self._samples_seen += FRAME_SAMPLES
        return self._emit_one(end_of_stream=False)

    def flush(self) -> list[np.ndarray]:
        """End of stream: emit ALL remaining steps, oldest first.

        Steps are zero-padded past the last pushed sample exactly like the
        offline tokenizer's end-of-clip edge padding. Returns [] when every
        pushed sample has already been tokenized (including an empty stream).
        Terminal — do not push frames afterwards.
        """
        remaining: list[np.ndarray] = []
        while True:
            tokens = self._emit_one(end_of_stream=True)
            if tokens is None:
                return remaining
            remaining.append(tokens)

    def _emit_one(self, *, end_of_stream: bool) -> np.ndarray | None:
        # Mid-stream: a step is emittable when its full 1040-sample window has
        # arrived. End of stream: also emit steps the offline step count still
        # expects, padding past the last pushed sample like tokenize() does.
        if not end_of_stream and self._buffer.shape[0] < STEP_WINDOW_SAMPLES:
            return None
        total_steps = n_steps_for_samples(self._samples_seen, self.config)
        if self._next_step >= total_steps:
            return None
        tokens = step_tokens(self._buffer[:STEP_WINDOW_SAMPLES], self.config)
        self._buffer = self._buffer[FRAME_SAMPLES:]
        self._next_step += 1
        return tokens
