# dMel geometry: from PCM to transformer token

Every number in this document was verified against the code at commit `a543b54`
(and re-verified at runtime by `dmel/features/tests/test_dmel_geometry.py` and
`dmel/models/tests/test_geometry.py`, which assert the same constants against
the live code so this document cannot silently rot).

**TL;DR — the working hypothesis is confirmed.** Per 50 ms JEV timestep the
tokenizer emits **400 token IDs = 5 mel subframes × 80 channels**, each ID
encoding one channel's quantized intensity as `id = bin · 16 + level` from a
`80 × 16 = 1280`-entry vocabulary. The 400 per-step embeddings are **mean-pooled**
into a single 256-dim vector (+ `agent_speaking` flag embedding), and that
pooled vector — not the 400 IDs — is the transformer token.

## 1. The pipeline, end to end

1. **PCM in.** 16 kHz mono int16, converted to float32 in `[-1, 1)`.
   (`dmel/features/config.py:60`, `tokenizer.py:_as_float`)
2. **Log-mel spectrogram.** Hann window of 400 samples (25 ms), hop of 160
   samples (10 ms), `n_fft = 512`, power spectrum via `rfft`, HTK mel
   filterbank with **80 bins** (fmin 0 Hz, fmax Nyquist), then
   `10·log10(E + 1e-10)` dB.
   (`dmel/features/config.py:61-62,66-69,77-79`,
   `tokenizer.py:log_mel_db`, `tokenizer.py:mel_filterbank`)
3. **Subframes per JEV step.** A 50 ms policy step groups
   `step_ms // hop_ms = 5` consecutive 10 ms mel frames ("subframes").
   Subframe *i* of step *t* is the 25 ms window starting at sample
   `t·800 + i·160`; windows reaching past the end of the clip are
   zero-padded (no start padding). Mel frames are computed for whole steps
   only, so the frame count always aligns with the per-50 ms label rows.
   (`dmel/features/config.py:63,145-147`, `tokenizer.py:log_mel_db`)
4. **Quantization.** Every (subframe, bin) cell is discretized independently
   to the **nearest of 16 levels** (`quant_bits = 4`) on **one shared linear
   codebook** evenly spaced over `[mel_min_db, mel_max_db] = [-70, 50]` dB —
   7.5 dB per level. Out-of-range values clip; midpoint ties round
   half-to-even (`numpy.rint`). The range is pinned in config (not derived
   from dataset min/max) so a step's tokens never depend on future audio and
   precompute matches streaming exactly.
   (`dmel/features/config.py:75-77`, `tokenizer.py:quantize_levels`)
5. **Token IDs — 400 per step.** One ID per (bin-group, level) pair:
   `id = group_index · n_levels + level`, with `bins_per_group = 1` by default
   so `group_index` **is** the mel bin index. Position `p` within a step's ID
   vector encodes `(subframe, bin-group) = (p // 80, p % 80)`. The step's
   slice `tokens[t]` (shape `(400,)` int32) is exactly what the common policy
   interface receives as `dmel_token_ids`.
   (`tokenizer.py:tokens_from_levels`, `tokenizer.py:194`,
   `tokenizer.py:14-19`; interface at `docs/CONTRACT.md:106-114`)
6. **Embedding.** A single shared `nn.Embedding(1280, 256)` table; each of
   the 400 IDs is looked up to a 256-dim vector → `(400, 256)`.
   (`dmel/models/model.py:52`, `dmel/models/config.py:45`)
7. **Pooling → the step vector.** The 400 per-step token embeddings are
   **mean-pooled over the token axis** (`embedded.mean(dim=-2)` — a plain
   arithmetic mean, no weights, no attention), then mixed by a learned
   `Linear(256 → 256)` (`step_proj`), then the `agent_speaking` flag's
   embedding (`nn.Embedding(2, 256)`) is **added**. Result: one `(256,)`
   vector per 50 ms step.
   (`dmel/models/model.py:103-105`; rationale comment `model.py:54-56`)
8. **Transformer token.** That pooled step vector **is** the transformer
   token. The causal backbone (6 layers × 256 hidden, 8 heads, FFN 640,
   pre-norm, GELU) runs over a rolling 40-step (2 s) window with
   **window-relative sinusoidal positions** and a strict causal mask; heads
   emit `action(2)` plus the four aux logits per step.
   (`dmel/models/config.py:44-49,58`, `dmel/models/backbones.py:36-87`,
   `dmel/models/model.py:116-122`, `docs/CONTRACT.md:129`)

Measured model size (transformer arm, default config): **4,115,846
parameters (~4.12M)**. Note that `docs/CONTRACT.md:129` still says
"~6 layers × 256 hidden (10–30 M params)" — the layer/hidden description is
accurate, the parameter estimate is a stale projection from before the
frontend was fixed at per-bin geometry (a flattened `400·256 → 256` projection
alone would add ~26M params, which is why pooling was chosen; see
`model.py:54-56`). Contract copy update is tracked separately.

## 2. Constants table

| Constant | Value | Where (file:line @ `a543b54`) |
|---|---|---|
| Sample rate | 16 000 Hz | `dmel/features/config.py:60` |
| Mel window | 25 ms → 400 samples | `dmel/features/config.py:61,133-135` |
| Mel hop | 10 ms → 160 samples | `dmel/features/config.py:62,137-139` |
| JEV step | 50 ms → 800 samples | `dmel/features/config.py:63,141-143` |
| Subframes per step | 5 | `dmel/features/config.py:145-147` |
| Mel bins (`n_mels`) | 80 | `dmel/features/config.py:66` |
| FFT size | 512 | `dmel/features/config.py:67` |
| Quantization | 4 bits → 16 levels | `dmel/features/config.py:75,149-151` |
| Codebook range | [-70, 50] dB linear (one shared codebook, 7.5 dB/level) | `dmel/features/config.py:76-77`, `tokenizer.py:152-165` |
| Bins per group | 1 (per-bin quantization) | `dmel/features/config.py:81` |
| Token ID formula | `group_index · 16 + level` | `dmel/features/tokenizer.py:194` |
| Tokens per step | 5 × 80 = **400** | `dmel/features/config.py:157-159` |
| Vocabulary size | 80 × 16 = **1280** | `dmel/features/config.py:161-163` |
| Steps per clip | `round(duration_ms / 50)` (integer arith., ties round up) | `tokenizer.py:63-76` |
| Token cache | `<sample>.dmel.npy`, `(n_steps, 400)` int32 | `dmel/features/precompute.py` (CLI), `dmel/training/dataset.py:117-126` |
| Embedding table | `nn.Embedding(1280, 256)` | `dmel/models/model.py:52` |
| Pooling op | **mean** over the 400-token axis (`dim=-2`), then `Linear(256→256)`, then add flag embedding | `dmel/models/model.py:103-105` |
| `agent_speaking` input | `nn.Embedding(2, 256)`, added to the pooled vector | `dmel/models/model.py:58,105` |
| Transformer | 6 layers × 256 hidden, 8 heads, FFN 640, causal, 40-step (2 s) window-relative sinusoidal positions | `dmel/models/config.py:44-49,58`, `dmel/models/backbones.py:36-87` |
| Measured params (transformer arm) | 4,115,846 (~4.12M) | runtime measurement, asserted order-of-magnitude by `dmel/models/tests/test_geometry.py` |
| Contract cross-refs | 16 kHz / 50 ms hop / 5 subframes (`docs/CONTRACT.md:28-29`); token id per (bin-group, level) (`docs/CONTRACT.md:102`); input = dMel token embeddings + `agent_speaking` (`docs/CONTRACT.md:129`) | |

The YAML at `dmel/features/configs/default.yaml` and the dataclass defaults in
`DmelConfig` are asserted equal by the existing
`test_default_yaml_matches_dataclass_defaults`; the training-side
`dmel/training/config.yaml` (`features.token_vocab_size: 1280`,
`features.tokens_per_step: 400`) is asserted consistent with `DmelConfig()`
by `dmel/features/tests/test_dmel_geometry.py`.

## 3. Worked example: one 50 ms timestep

Input: **800 int16 samples** (exactly one step) — a 440 Hz sine at amplitude
0.25. Runtime trace (reproduce with `python -m pytest
dmel/features/tests/test_dmel_geometry.py dmel/models/tests/test_geometry.py`):

1. **PCM slice** → `tokens = tokenize(pcm, config)` returns shape **(1, 400)**
   int32, ids in `[0, 1280)` (this example: min 2, max 1269).
2. **Position layout.** Position `p` → subframe `p // 80`, bin `p % 80`:
   positions 0 and 79 are subframe 0 (bins 0 and 79); position 80 opens
   subframe 1; position 160 opens subframe 2; position 399 is subframe 4,
   bin 79. Each 80-ID block **is** one 10 ms mel frame.
3. **ID values.** For bin 5: subframe 0's dB value quantizes to level 3 →
   `id = 5·16 + 3 = 83`; subframe 4's to level 10 → `id = 5·16 + 10 = 90`.
   Both match the tokenizer output exactly (independently recomputed via
   `log_mel_db` + `quantize_levels`).
4. **Sanity anchor.** Digital silence quantizes to level 0 everywhere, so a
   silent step's IDs are exactly the bin multiples `{0, 16, 32, …, 1264}` —
   80 IDs, all with `id % 16 == 0`.
5. **Embedding + pooling.** `token_embed(ids)` → `(400, 256)`;
   `mean(dim=-2)` → `(256,)`; `step_proj` → `(256,)`; plus
   `agent_speaking_embed(True)` → the step's final `(256,)` vector. A manual
   recomputation of this chain matches `_embed_steps` bit-for-bit
   (max |Δ| = 0.0).
6. **Token.** That `(256,)` vector is the **one** transformer token
   representing this 50 ms of audio; 40 of them form the 2 s context window.

## 4. The two questions

**(a) Why 400 IDs given 80 channels?** Because dMel quantizes each mel bin
separately: one mel frame contributes **one ID per channel**, not one ID for
the whole frame. A 50 ms step contains 5 mel subframes × 80 channels =
**400 channel-value IDs**. The 80 channels are never summed, collapsed, or
flattened into fewer IDs — they appear as 80 consecutive IDs within each of
the 5 per-step blocks (`p % 80` = bin), and the block index (`p // 80`) is
which subframe. (Generalization: with `bins_per_group > 1` a group's level is
the max of its bins and the per-frame ID count drops to `n_mels /
bins_per_group`; the default 1 bin/group keeps full per-bin resolution.)

**(b) Are the 400 IDs multiple dMel frames grouped into one JEV timestep?**
**Yes.** The mel analysis grid is 25 ms windows / 10 ms hops; a 50 ms JEV step
groups **five consecutive 10 ms mel frames** (`step_ms // hop_ms = 5`), and
each frame contributes its 80 channel IDs in order. So per transformer token
you have 5 frames' worth of fine-grained audio detail — but it is aggregated
at the embedding layer (mean over the 400 token embeddings), not at the
tokenizer: the policy network never sees sub-steps, only the pooled 256-dim
vector per 50 ms step, plus the `agent_speaking` flag.

## 5. Nuances and deviations from the hypothesis

Nothing in the hypothesis was contradicted. Three refinements worth knowing:

- **The 16 levels come from one shared codebook**, not 80 per-bin codebooks:
  a single linear scale over [-70, 50] dB is applied to every (subframe, bin)
  cell (`tokenizer.py:quantize_levels`, following dMel eq. 2–4 with the range
  pinned in config). Vocab 1280 is the 80 × 16 space of **(bin, level)
  pairs** — each embedding row is a (which channel, how intense) pair.
- **Pooling is a plain arithmetic mean** over the 400 token embeddings
  (`model.py:103`), followed by a learned `Linear(256→256)` mix
  (`step_proj`) and an **additive** `agent_speaking` embedding. The docstring
  records that a flattened `400·256 → 256` projection was rejected: ~26M
  params and ~2× the per-step latency budget (`model.py:54-56`).
- **Edge handling:** subframe windows past the end of a clip are zero-padded
  (no start padding), and mel frames are computed for whole steps only, so
  token counts always match label-row counts (`tokenizer.py:log_mel_db`,
  `tokenizer.py:14-23`).
