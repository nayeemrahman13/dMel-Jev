# Capacity sweep: 6×256 vs 6×384 vs 6×512

**Question.** Does transformer capacity (width) improve barge-in policy quality on the
pilot corpus, and do wider models still fit the <10 ms single-core CPU step budget?
The keep/replace decision is made by the orchestrator; this document supplies the
comparison data.

**Status:** runs in progress — numbers below are placeholders until the nine runs finish.

## Design (identical across all nine runs)

- **Frontend — frozen** (docs/dmel_geometry.md): 16 kHz → 25 ms/10 ms mel, 80 bins →
  5 subframes per 50 ms step → 400 IDs (id = bin*16 + level, shared 16-level codebook,
  vocab 1280) → nn.Embedding(1280, 256) → MEAN pool → step_proj + agent_speaking add
  → transformer token. No per-bin tokens anywhere. Widths change ONLY d_model and the
  dims that necessarily scale (heads preserving head_dim=32, FFN keeping the 2.5×
  ratio, step_proj).
- **Dataset:** deterministic pilot corpus regenerated (`python -m dmel.data.generate
  --preset pilot`; 840 samples / 3.12 h, seed 1234). Train split only (266 samples,
  40-step windows, balanced-stop weighted sampling); selection on validation (296);
  test never touched.
- **Optimization:** 5 epochs, batch 32, AdamW lr 3e-4 wd 0.01 grad-clip 1.0,
  action-dominant loss with 0.3/0.2/0.1/0.1 aux weights, per-head BCE pos_weight from
  train-split positive rates (cap 20). Seeds 0/1/2 — the config's 3-seed list.
  All nine runs trained fresh and in parallel on Modal L4; no reuse of earlier runs.
- **Operating point:** per checkpoint, the stop threshold is swept over validation
  through the eval harness; ONE point selected by max stop F-beta (β=2), ties → lowest
  threshold. Metrics below are at that pre-selected point; bootstrap CIs are the
  harness defaults.

| Arm | d_model | heads | FFN | params |
|---|---|---|---|---|
| 6×256 (baseline) | 256 | 8 | 640 | 4,115,846 |
| 6×384 | 384 | 12 | 960 | 9,000,006 |
| 6×512 | 512 | 16 | 1280 | 15,768,326 |

## Results (validation, mean ± sd over 3 seeds)

TBD

## CPU inference latency (single core, 40-step window recompute)

TBD

## Repro

TBD

## Cost

TBD
