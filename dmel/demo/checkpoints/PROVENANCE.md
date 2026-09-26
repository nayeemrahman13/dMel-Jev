# Checkpoint provenance: w256_seed0_best.pt

The live demo's learned arm (arm C). Recovered from the Modal training volume on
2026-09-26 with `python -m dmel.demo.recover_checkpoint` — nothing about the
checkpoint was modified in transit; the file is a byte-for-byte copy of the
volume object.

| Field | Value |
|---|---|
| Source | Modal volume `dmel-jev-runs`, `/capacity_sweep/w256/seed_0/transformer/seed_0/best.pt` |
| Training run | Capacity sweep 6×256, seed 0 — protocol in `docs/capacity_sweep.md` (5 epochs, batch 32, AdamW 3e-4, pilot train split, selection on validation) |
| Selected epoch | 2 (best validation total loss of epochs 0–4) |
| Validation at save | total 0.2603, action 0.1406, action_accuracy 0.9628 |
| Parameters | 4,115,846 (transformer, 6 layers × 256, 8 heads, FFN 640) |
| Frontend | dMel defaults: 400 ids/step, vocab 1280 (`dmel_geometry.md`) |
| Validation stop threshold | 0.50 (from the sweep's 0.30–0.70 grid — `reports/capacity_sweep/eval/w256_seed0/`) |
| sha256 | `690ac00ef378ac227bb191610fbc2e73a4459cba122828ec7878f543dd66d480` |

The same volume directory also holds `last.pt` (epoch 4) and `loss_curves.json`;
the demo ships `best.pt` only. If this file is missing or corrupt,
`python -m dmel.demo.recover_checkpoint` re-fetches it (or retrains the seed
from scratch with the same protocol).
