"""Stem mixing and augmentation.

Contract stems per sample: ``<id>.agent.pcm`` (TTS playback), ``<id>.user.pcm``
(primary-user speech plus that user's non-speech vocal events), and
``<id>.mix.pcm`` (what the microphone hears: agent playback + user +
background voice). The background speaker appears only in the mix.

Gains are dB relative to the agent (playback) level: user speech lands at the
drawn SNR over the agent stem, background voice at -18..-10 dB, noise events
at the drawn noise level. Truncated-world agent runs arrive with clamped
UtteranceSpan ends; only the surviving audio is placed.

The codec-ish degradation family (contract "Splits": eval-only augmentation)
is applied to the mix of eval samples only, with parameters drawn from the
sample-id-keyed augmentation RNG.
"""

from __future__ import annotations

import numpy as np

from dmel.data.constants import FRAME_SAMPLES, SAMPLE_RATE
from dmel.data.scenarios import (
    ROLE_AGENT,
    ROLE_BACKGROUND,
    ROLE_USER,
    SampleScript,
)
from dmel.data.synthesis import synth_noise


def _rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(x**2))) if x.size else 0.0


def _db_to_gain(db: float) -> float:
    return 10.0 ** (db / 20.0)


def _pad_to_frames(n: int) -> int:
    """Smallest frame-aligned sample count >= n (stems end on a 50 ms boundary)."""
    return ((n + FRAME_SAMPLES - 1) // FRAME_SAMPLES) * FRAME_SAMPLES


def _bandpass(x: np.ndarray, lo_hz: float, hi_hz: float) -> np.ndarray:
    """FFT band-limitation with soft edges (codec-ish narrowband effect)."""
    n = x.size
    if n == 0:
        return x
    spec = np.fft.rfft(x)
    freqs = np.fft.rfftfreq(n, d=1.0 / SAMPLE_RATE)
    edge = (hi_hz - lo_hz) * 0.1
    mask = np.clip((freqs - lo_hz) / max(edge, 1.0), 0.0, 1.0)
    mask *= np.clip((hi_hz - freqs) / max(edge, 1.0), 0.0, 1.0)
    return np.fft.irfft(spec * mask, n)


def _mulaw_roundtrip(x: np.ndarray, bits: int) -> np.ndarray:
    """mu-law quantize then reconstruct — cheap codec-like grit."""
    mu = 255.0
    x_norm = np.clip(x, -1.0, 1.0)
    enc = np.sign(x_norm) * np.log1p(mu * np.abs(x_norm)) / np.log1p(mu)
    levels = 2**bits
    enc_q = np.round(enc * (levels // 2)) / (levels // 2)
    return np.sign(enc_q) * ((1.0 + mu) ** np.abs(enc_q) - 1.0) / mu


def codec_degrade(mix: np.ndarray, params: dict) -> np.ndarray:
    """Apply the eval-only codec degradation family to a float mix."""
    variant = params["variant"]
    if variant in ("narrowband", "narrowband_mulaw"):
        mix = _bandpass(mix, 300.0, 3400.0)
    if variant in ("mulaw", "narrowband_mulaw"):
        mix = _mulaw_roundtrip(mix, int(params["bits"]))
    if variant == "narrowband_low":
        mix = _bandpass(mix, 300.0, 2700.0)
    return mix


def draw_codec(rng: np.random.Generator) -> dict:
    """Draw codec parameters from the (sample-id-keyed) augmentation RNG."""
    variant = ("narrowband", "mulaw", "narrowband_mulaw", "narrowband_low")[
        int(rng.integers(0, 4))
    ]
    return {"variant": variant, "bits": int(rng.choice([6, 8]))}


def mix_sample(
    script: SampleScript,
    rendered: dict[tuple[str, str, int], np.ndarray],
    rng_noise: np.random.Generator,
) -> dict[str, np.ndarray]:
    """Assemble agent/user/mix stems for a planned script.

    ``rendered`` maps (text, voice, rate) -> int16 PCM for every utterance in
    ``script``; ``rng_noise`` shapes the noise-event waveforms.
    """
    total = max(
        [u.end for u in script.utterances]
        + [n.end for n in script.noises]
        + [0]
    )
    total = _pad_to_frames(total + int(0.4 * SAMPLE_RATE))  # small trailing pad

    agent = np.zeros(total, dtype=np.float64)
    user = np.zeros(total, dtype=np.float64)
    background = np.zeros(total, dtype=np.float64)

    def pcm_of(u) -> np.ndarray:
        return rendered[(u.text, script.voices[u.role], script.rates[u.role])]

    # Reference level: mean RMS of the agent (playback) utterances.
    agent_rms_values = [
        _rms(pcm_of(u).astype(np.float64) / 32768.0)
        for u in script.utterances
        if u.role == ROLE_AGENT
    ]
    agent_rms = float(np.mean(agent_rms_values)) if agent_rms_values else 0.08

    agent_spans = [u for u in script.utterances if u.role == ROLE_AGENT]
    for u in script.utterances:
        pcm = pcm_of(u)[: max(u.end - u.start, 0)]  # truncated worlds place clipped audio
        dst = user if u.role == ROLE_USER else (background if u.role == ROLE_BACKGROUND else agent)
        gain = 1.0
        if u.role == ROLE_USER:
            # SNR vs the agent: referenced to the overlapping agent audio when
            # the utterance overlaps a run, else the sample's agent level.
            segs = []
            for a in agent_spans:
                lo, hi = max(a.start, u.start), min(a.end, u.end)
                seg = agent[lo:hi]
                if seg.size:
                    segs.append(seg)
            overlap_rms = _rms(np.concatenate(segs)) if segs else agent_rms
            pcm_rms = _rms(pcm.astype(np.float64) / 32768.0)
            gain = (overlap_rms / max(pcm_rms, 1e-6)) * _db_to_gain(script.snr_db)
        elif u.role == ROLE_BACKGROUND:
            pcm_rms = _rms(pcm.astype(np.float64) / 32768.0)
            gain = (agent_rms / max(pcm_rms, 1e-6)) * _db_to_gain(script.background_db)
        if pcm.size:
            dst[u.start : u.end] += pcm.astype(np.float64) / 32768.0 * gain

    # Non-speech vocal events land on the user channel (speech_present=0).
    for n in script.noises:
        length = n.end - n.start
        wave = synth_noise(n.kind, length, rng_noise)
        wave = wave / max(_rms(wave), 1e-6) * agent_rms * _db_to_gain(script.noise_db)
        user[n.start : n.end] += wave

    mix = agent + user + background
    if script.codec is not None:
        mix = codec_degrade(mix, script.codec)

    def to_i16(x: np.ndarray) -> np.ndarray:
        return np.clip(np.round(x * 32768.0), -32768, 32767).astype(np.int16)

    return {"agent": to_i16(agent), "user": to_i16(user), "mix": to_i16(mix)}
