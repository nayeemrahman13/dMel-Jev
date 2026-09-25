"""dMel tokenization: 16 kHz int16 PCM -> log-mel -> quantized levels -> token ids.

Implements the dmel/features area of the dMel-Jev POC data & interface
contract v0:

    16 kHz mono int16 -> log-mel (25 ms window / 10 ms hop, 80 bins default)
    -> per-cell intensity quantization (4-bit default) -> 50 ms steps
    (5 subframes per step) -> token ids, one per (bin-group, level).

Shapes and encoding
-------------------
- :func:`tokenize` returns an int32 array of shape ``(n_steps, tokens_per_step)``
  where ``n_steps == round(duration_ms / 50)`` (ties round up) and
  ``tokens_per_step == subframes_per_step * n_groups`` (400 for the default
  80 bins x 4 bits x 1 bin/group).
- The token *value* encodes ``(bin-group, level)``: ``id = group * n_levels +
  level``. The subframe index and bin group of position ``p`` within a step
  are ``p // n_groups`` and ``p % n_groups``. The value vocabulary is
  ``n_groups * n_levels`` (1280 for the defaults) — size embeddings
  accordingly. A per-step slice ``tokens[t]`` is what the common policy
  interface receives as ``dmel_token_ids``.
- Edge padding: subframe ``i`` is the 25 ms window starting at sample
  ``i * hop_samples``; windows reaching past the end of the clip are
  zero-padded. There is no start padding, so subframe 0 is aligned to sample
  0 exactly like the streaming runtime. Audio beyond the last emitted
  subframe's window is not represented — that can only happen when a clip is
  not a whole multiple of 50 ms *and* its duration rounds down; corpus clips
  are whole-step multiples by construction.

Reference: "dMel: Speech Tokenization Made Simple", arXiv 2407.15835.
Deliberate deviations from the paper, per the POC contract:
- 25/10 ms window/hop (paper: 50/25 ms).
- The paper derives the codebook range from dataset min/max log-mel; we pin
  the range in config (``mel_min_db``/``mel_max_db``) so a step's tokens never
  depend on future audio and precompute matches streaming exactly.
- Ties at codebook midpoints round half-to-even (numpy ``rint``); the paper's
  argmin breaks ties toward the lower index. Measure-zero for real audio.

Every function here is pure and deterministic: identical input bytes and
config give identical token ids, always. Only numpy is required (torch is not
needed to produce tokens; downstream models may consume the int32 ids
directly).
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np

from dmel.features.config import DmelConfig

__all__ = [
    "decode_tokens",
    "log_mel_db",
    "mel_filterbank",
    "n_steps_for_samples",
    "quantize_levels",
    "read_pcm_i16",
    "tokenize",
    "tokens_from_levels",
]


def n_steps_for_samples(n_samples: int, config: DmelConfig) -> int:
    """Number of policy steps for a clip of ``n_samples`` samples.

    ``round(duration_ms / step_ms)`` in exact integer arithmetic
    (``duration_ms = n_samples / sample_rate * 1000``). A duration exactly
    halfway between two step counts rounds up; this differs from Python's
    builtin half-to-even ``round`` only at those exact ties, which whole-step
    corpus clips never hit.
    """
    if n_samples < 0:
        raise ValueError(f"n_samples must be >= 0, got {n_samples}")
    half = config.step_samples // 2
    return (n_samples + half) // config.step_samples


def _hz_to_mel(hz: float) -> np.ndarray:
    return 2595.0 * np.log10(1.0 + np.asarray(hz, dtype=np.float64) / 700.0)


def _mel_to_hz(mel: np.ndarray) -> np.ndarray:
    return 700.0 * (10.0 ** (np.asarray(mel, dtype=np.float64) / 2595.0) - 1.0)


@lru_cache(maxsize=8)
def mel_filterbank(config: DmelConfig) -> np.ndarray:
    """Triangular mel filterbank, shape ``(n_mels, n_fft // 2 + 1)``, float64.

    HTK-style mel scale (``2595 * log10(1 + f / 700)``); each triangle has
    unit peak and spans three mel-spaced FFT-bin breakpoints. Memoized per
    config; the returned array is read-only.
    """
    n_freq = config.n_fft // 2 + 1
    fmax = config.fmax_hz if config.fmax_hz is not None else config.sample_rate_hz / 2.0
    mel_pts = np.linspace(_hz_to_mel(config.fmin_hz), _hz_to_mel(float(fmax)), config.n_mels + 2)
    hz_pts = _mel_to_hz(mel_pts)
    bin_pts = hz_pts * (config.n_fft / config.sample_rate_hz)
    k = np.arange(n_freq, dtype=np.float64)
    bank = np.zeros((config.n_mels, n_freq), dtype=np.float64)
    for i in range(config.n_mels):
        left, center, right = bin_pts[i], bin_pts[i + 1], bin_pts[i + 2]
        rising = (k - left) / (center - left)
        falling = (right - k) / (right - center)
        bank[i] = np.maximum(0.0, np.minimum(rising, falling))
    bank.setflags(write=False)
    return bank


def _as_float(pcm_i16: np.ndarray) -> np.ndarray:
    """int16 PCM -> float32 in [-1, 1), matching the runtime's ``Pcm.i16_to_f32``."""
    arr = np.asarray(pcm_i16)
    if arr.ndim != 1:
        raise ValueError(f"pcm must be 1-D, got shape {arr.shape}")
    if arr.dtype != np.int16:
        raise ValueError(f"pcm must be int16, got dtype {arr.dtype}")
    return arr.astype(np.float32) / 32768.0


def log_mel_db(pcm_i16: np.ndarray, config: DmelConfig) -> np.ndarray:
    """Log-mel spectrogram in dB, shape ``(n_subframes, n_mels)`` float64.

    ``n_subframes == n_steps * subframes_per_step`` — mel frames are computed
    for whole steps only, which is what makes the frame count align exactly
    with the per-50 ms label rows. Power spectrum (Hann window, rfft), mel
    energies as ``power @ filterbank.T`` in float64, then
    ``10 * log10(energy + log_eps)``. Digital silence lands at
    ``10 * log10(log_eps)`` dB and quantizes to level 0.
    """
    x = _as_float(pcm_i16)
    n = x.shape[0]
    n_steps = n_steps_for_samples(n, config)
    n_sub = n_steps * config.subframes_per_step
    if n_sub == 0:
        return np.zeros((0, config.n_mels), dtype=np.float64)

    window = np.hanning(config.window_samples)
    pad_end = max(0, (n_sub - 1) * config.hop_samples + config.window_samples - n)
    padded = np.concatenate([x, np.zeros(pad_end, dtype=x.dtype)])
    frames = np.lib.stride_tricks.sliding_window_view(padded, config.window_samples)
    frames = frames[:: config.hop_samples][:n_sub]

    spectrum = np.fft.rfft((frames * window).astype(np.float64), n=config.n_fft, axis=1)
    power = spectrum.real**2 + spectrum.imag**2
    mel_energy = power @ mel_filterbank(config).T
    return 10.0 * np.log10(mel_energy + config.log_eps)


def quantize_levels(logmel_db: np.ndarray, config: DmelConfig) -> np.ndarray:
    """Quantize dB values to levels in ``[0, n_levels)`` (int32, same shape).

    Nearest point on the shared linear codebook (dMel eq. 4): the codebook is
    evenly spaced over ``[mel_min_db, mel_max_db]`` and each cell is
    discretized independently. Values below/above the range clip to 0 /
    ``n_levels - 1``; midpoint ties round half-to-even.
    """
    x = np.asarray(logmel_db, dtype=np.float64)
    if not np.all(np.isfinite(x)):
        raise ValueError("log-mel input must be finite")
    step_db = (config.mel_max_db - config.mel_min_db) / config.n_levels
    levels = np.rint((x - config.mel_min_db) / step_db)
    return np.clip(levels, 0, config.n_levels - 1).astype(np.int32)


def tokens_from_levels(levels: np.ndarray, config: DmelConfig) -> np.ndarray:
    """Pack per-subframe levels ``(n_subframes, n_mels)`` into token ids.

    Returns int32 ``(n_steps, tokens_per_step)``. Bins are grouped into
    ``n_groups`` consecutive bin-groups per subframe; when
    ``bins_per_group > 1`` the group's level is the max (dominant intensity)
    of its bins. Token value = ``group_index * n_levels + level``; position
    within a step encodes (subframe, bin-group).
    """
    levels = np.asarray(levels, dtype=np.int32)
    if levels.ndim != 2 or levels.shape[1] != config.n_mels:
        raise ValueError(
            f"levels must have shape (n_subframes, {config.n_mels}), got {levels.shape}"
        )
    n_steps, remainder = divmod(levels.shape[0], config.subframes_per_step)
    if remainder:
        raise ValueError(
            f"got {levels.shape[0]} subframes, not a whole multiple of "
            f"subframes_per_step ({config.subframes_per_step})"
        )
    grouped = levels.reshape(n_steps, config.subframes_per_step, config.n_groups, config.bins_per_group)
    if config.bins_per_group > 1:
        grouped = grouped.max(axis=3)
    else:
        grouped = grouped[..., 0]
    group_ids = np.arange(config.n_groups, dtype=np.int32)
    tokens = group_ids * config.n_levels + grouped
    return np.ascontiguousarray(tokens.reshape(n_steps, config.tokens_per_step), dtype=np.int32)


def tokenize(pcm_i16: np.ndarray, config: DmelConfig) -> np.ndarray:
    """Full pipeline: int16 PCM -> token ids ``(n_steps, tokens_per_step)`` int32.

    Deterministic and streaming-faithful: the tokens for step ``t`` depend
    only on samples in that step's subframe windows (plus documented tail
    zero-padding on the final step).
    """
    return tokens_from_levels(quantize_levels(log_mel_db(pcm_i16, config), config), config)


def decode_tokens(tokens: np.ndarray, config: DmelConfig) -> np.ndarray:
    """Recover quantization levels from token ids (round-trip helper).

    Returns int32 ``(n_steps, subframes_per_step, n_mels)`` when
    ``bins_per_group == 1`` (exact inverse of :func:`tokenize`), otherwise
    ``(n_steps, subframes_per_step, n_groups)`` at bin-group granularity.
    """
    tokens = np.asarray(tokens, dtype=np.int32)
    if tokens.ndim != 2 or tokens.shape[1] != config.tokens_per_step:
        raise ValueError(
            f"tokens must have shape (n_steps, {config.tokens_per_step}), got {tokens.shape}"
        )
    if tokens.size and (tokens.min() < 0 or tokens.max() >= config.vocab_size):
        raise ValueError(f"token ids out of vocabulary [0, {config.vocab_size})")
    n_steps = tokens.shape[0]
    per_group = (tokens % config.n_levels).reshape(
        n_steps, config.subframes_per_step, config.n_groups
    )
    if config.bins_per_group == 1:
        return per_group.reshape(n_steps, config.subframes_per_step, config.n_mels)
    return np.ascontiguousarray(per_group, dtype=np.int32)


def read_pcm_i16(path: str | Path) -> np.ndarray:
    """Read a raw headerless int16 little-endian PCM file (contract stem format)."""
    data = Path(path).read_bytes()
    if len(data) % 2:
        raise ValueError(f"truncated int16 PCM ({len(data)} bytes, odd length)")
    return np.frombuffer(data, dtype="<i2").astype(np.int16)
