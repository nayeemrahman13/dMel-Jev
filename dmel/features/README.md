# dmel/features — dMel tokenization

Config-driven dMel tokenizer for the dMel-Jev POC (contract v1, features
area): 16 kHz mono int16 → log-mel (25 ms window / 10 ms hop, 80 bins
default) → per-cell intensity quantization (4-bit default) → 50 ms steps
(5 subframes) → token ids. Pure numpy; deterministic.

## Layout

| file | role |
|---|---|
| `config.py` | `DmelConfig` (frozen dataclass, validated) + YAML loader |
| `tokenizer.py` | pure-numpy pipeline: `tokenize`, `log_mel_db`, `quantize_levels`, `tokens_from_levels`, `decode_tokens`, `read_pcm_i16` |
| `precompute.py` | CLI: shard walk → `<sample_id>.dmel.npy` caches |
| `configs/default.yaml` | contract defaults |
| `tests/` | pytest suite (synthesized signals; consumes `dmel/data/fixtures` when present) |

## Token shape contract (read before consuming tokens)

`tokenize(pcm_i16, config)` returns **int32 `(n_steps, tokens_per_step)`**:

- `n_steps == round(duration_ms / 50)` — computed as
  `(n_samples + step_samples // 2) // step_samples` (exact integer math,
  half-up ties; documented in `n_steps_for_samples`).
- `tokens_per_step == subframes_per_step * n_groups` = **400** for the
  default 80 bins × 4 bits × 1 bin/group (5 subframes × 80 groups).
- Token **value** encodes `(bin-group, level)`:
  `id = group_index * n_levels + level` → vocabulary
  `n_groups * n_levels` = **1280** for the defaults. Position `p` within a
  step encodes subframe `p // n_groups` and bin-group `p % n_groups`.
- The common policy interface's per-step `dmel_token_ids` is the row
  `tokens[t]`, shape `(400,)`.

**Models task:** size embeddings from `DmelConfig().vocab_size` (1280), or use
the paper's factored form — `divmod(id, 16)` gives (bin, level), so a bin
embedding table (80) + level embedding table (16) summed is equivalent.

## Alignment & edge padding

Subframe `i` is the 25 ms window starting at sample `i * hop` (no start
padding — subframe 0 aligns to sample 0, matching the streaming runtime).
Windows reaching past the clip end are zero-padded; the tokenizer emits
exactly `n_steps * 5` subframes, so token rows align 1:1 with the contract's
per-50 ms label rows. Audio beyond the last emitted subframe's window is not
represented; that requires a clip whose duration is not a whole multiple of
50 ms *and* rounds down — corpus clips are whole-step multiples by
construction.

## Quantization

One shared linear codebook over log-mel dB (dMel paper eqs. 2–4): levels are
the nearest of `n_levels` evenly spaced points over
`[mel_min_db, mel_max_db]` (defaults −70…+50 dB; calibrated against the
synthetic-corpus signal levels — digital silence → level 0, full-scale tone →
level 15). The paper derives the range from dataset min/max; we pin it in
config so a step's tokens never depend on future audio and precompute matches
streaming exactly. Other deliberate paper deviations: 25/10 ms window/hop
(paper: 50/25), midpoint ties round half-to-even.

Input convention: int16 normalized by 1/32768 (same as the runtime's
`Pcm.i16_to_f32`). The dB range is calibrated to that convention — changing
it invalidates `mel_min_db`/`mel_max_db`.

## CLI

```bash
python -m dmel.features.precompute --shard data/pilot/shard-00001
python -m dmel.features.precompute --shard data/pilot/shard-00001 --stem agent --out-suffix .agent.dmel.npy
```

Writes `<sample_id>.dmel.npy` next to the pcms (sorted, deterministic; caches
are skipped unless `--overwrite`; `--dry-run` reports without writing; the
agent stem for ablation arm E uses `--stem agent`, with `--out-suffix` to
keep both caches). Exit codes: 0 ok, 1 corrupt pcm, 2 usage error.

## Determinism

Same input bytes + same config ⇒ identical token ids, always. The pipeline is
stateless pure numpy (no random, no adaptive statistics); the filterbank is
derived from config only; float64 is used for the spectral projection. Tests
assert bitwise-equal reruns in-process and via the CLI (byte-identical
`.npy` files).

## Tests

```bash
python -m pytest dmel/features/tests -q          # this area
python -m pytest                                  # whole dmel suite (root testpaths: dmel)
```

`tests/test_dmel_fixtures.py` consumes the committed fixtures at
`dmel/data/fixtures/` (skipped cleanly when absent),
including a cross-check that fixture label parquets have one row per token
step.

`tests/test_dmel_leakage_linter.py` is the contract-v1 guardrail: it asserts
the feature builder consumes only mix frames — never label columns, scenario,
or split (signature lint, config-surface lint, a source lint of the production
modules, and CLI checks that label/manifest sidecars are untouched and cannot
influence caches). Input flags (`agent_speaking`, `speaker_similarity`) are
policy-layer inputs and equally absent from the builder surface.

## Dependencies

Contract v1: install through the shared corpus pin file:

```bash
pip install -r dmel/features/requirements.txt -c dmel/constraints.txt
```
