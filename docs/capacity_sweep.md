# Capacity sweep: 6×256 vs 6×384 vs 6×512

**Question.** Does transformer capacity (width) improve barge-in policy quality on the
pilot corpus, and do wider models still fit the <10 ms single-core CPU step budget?
The keep/replace decision is made by the orchestrator; this document supplies the
comparison data.

**Headline.** Width does not buy quality on this corpus. 6×384 matches 6×256 within
seed noise; 6×512 collapses to a never-stop policy in 2 of 3 seeds. Both wider models
blow the CPU step budget (17.6 ms and 28.4 ms p95 vs 6.2 ms). Every width trained
under an identical, fixed protocol; nothing was re-rolled.

## Design (identical across all nine runs)

- **Frontend — frozen** (docs/dmel_geometry.md): 16 kHz → 25 ms/10 ms mel, 80 bins →
  5 subframes per 50 ms step → 400 IDs (id = bin*16 + level, shared 16-level codebook,
  vocab 1280) → nn.Embedding(1280, 256) → MEAN pool → step_proj + agent_speaking add
  → transformer token. No per-bin tokens anywhere. Widths change ONLY d_model and the
  dims that necessarily scale (heads preserving head_dim=32, FFN keeping the 2.5×
  ratio, step_proj).
- **Dataset:** deterministic pilot corpus regenerated (`python -m dmel.data.generate`
  with the pilot preset; 840 samples / 3.12 h). Train split only (266 samples,
  40-step windows, balanced-stop weighted sampling); selection on validation (296);
  test never touched.
- **Optimization:** 5 epochs, batch 32, AdamW lr 3e-4 wd 0.01 grad-clip 1.0,
  action-dominant loss with 0.3/0.2/0.1/0.1 aux weights, per-head BCE pos_weight from
  train-split positive rates (cap 20). Seeds 0/1/2 — the config's 3-seed list.
  All nine runs trained fresh and in parallel on Modal L4; no reuse of earlier runs.
- **Operating point:** per checkpoint, the stop threshold is swept over validation
  through the eval harness over the 0.30–0.70 grid (0.05 steps); ONE point selected
  by max stop F-beta (β=2), ties → lowest threshold. Metrics below are at that
  pre-selected point; per-head pos_weight, schedule, and every other hyperparameter
  are shared. No hyperparameter was changed for any width.

| Arm | d_model | heads | FFN | params |
|---|---|---|---|---|
| 6×256 (baseline) | 256 | 8 | 640 | 4,115,846 |
| 6×384 | 384 | 12 | 960 | 9,000,006 |
| 6×512 | 512 | 16 | 1280 | 15,768,326 |

## Results (validation, mean ± sd over 3 seeds)

| Metric | 6×256 | 6×384 | 6×512 |
|---|---|---|---|
| First in-window STOP recall | 0.347 ± 0.159 | 0.396 ± 0.178 | 0.165 ± 0.286 |
| Never stopped (censored rate) | 0.653 ± 0.159 | 0.604 ± 0.178 | 0.835 ± 0.286 |
| Premature stop — during overlap | 0.028 ± 0.026 | 0.021 ± 0.018 | 0.032 ± 0.055 |
| Premature stop — during non-overlap | 0.084 ± 0.032 | 0.112 ± 0.052 | 0.042 ± 0.073 |
| Hesitation-commit false stops (segment rate) | 0.579 ± 0.091 | 0.596 ± 0.061 | 0.211 ± 0.365 |
| Noise false stops (segment rate) | 0.147 ± 0.051 | 0.157 ± 0.034 | 0.059 ± 0.102 |
| Background-speech false stops (segment rate) | 0.233 ± 0.115 | 0.289 ± 0.204 | 0.100 ± 0.173 |
| Backchannel false stops (segment rate) | 0.217 ± 0.059 | 0.287 ± 0.117 | 0.085 ± 0.148 |
| Total false stops (frames) | 4035.0 ± 1279.6 | 4837.0 ± 1859.0 | 1637.0 ± 2835.4 |
| Decision latency median (ms) | — (censored 3/3) | 381.0 ± 0.0 (n=1) | — (censored) |
| Decision latency p95 (ms) | — (censored 3/3) | — (censored 3/3) | — (censored) |
| Stop-decision F-beta (β=2) | 0.285 ± 0.080 | 0.294 ± 0.067 | 0.360 ± 0.000 (n=1) |

Reads, per seed (`thr` = selected stop threshold, `fs` = false-stop frames):

| Run | thr | recall | F-beta β=2 | latency median | fs |
|---|---|---|---|---|---|
| w256 s0 | 0.50 | 0.400 | 0.333 | censored | 4514 |
| w256 s1 | 0.30 | 0.474 | 0.328 | censored | 5006 |
| w256 s2 | 0.30 | 0.168 | 0.193 | censored | 2585 |
| w384 s0 | 0.30 | 0.442 | 0.321 | censored | 5868 |
| w384 s1 | 0.30 | 0.200 | 0.217 | censored | 2691 |
| w384 s2 | 0.40 | 0.547 | 0.343 | 381 ms (n=52 handled, 43 censored) | 5952 |
| w512 s0 | 0.45 | 0.495 | 0.360 | censored | 4911 |
| w512 s1 | **none** | 0.000 | — | — | 0 |
| w512 s2 | **none** | 0.000 | — | — | 0 |

- **6×384 vs 6×256:** recall 0.396 ± 0.178 vs 0.347 ± 0.159 — the +0.05 shift is well
  inside one seed sd; every seed pair straddles. F-beta is likewise flat (0.294 vs
  0.285). 2.2× parameters buy nothing measurable here.
- **6×512 is unstable:** seeds 1 and 2 emitted zero predicted stop frames at every
  threshold (0.30–0.70) on validation — a never-stop collapse, so no operating point
  exists (F-beta undefined; the harness persists the degenerate rows with
  `no_operating_point: true` rather than inventing one). All of w512's nonzero
  numbers ride on the single surviving seed 0; its sd cells are not meaningful.
  The corpus has 266 train samples — a 15.8M-parameter policy overfits or collapses
  depending on the seed.
- **Thresholds sit at the grid edge** (0.30 = lowest allowed) in 5 of 7 runs with an
  operating point, so true optima may lie below 0.30 — identical across widths, so
  the *comparison* stands, but absolute numbers are conservative in F-beta terms.
- **Decision latency is mostly censored:** the harness treats never-stopped events as
  censored at infinity and reports a quantile only when it lands on finite values.
  With never-stopped rates of 60–84% per run, the median is censored in 8 of 9 runs.
  The one computable cell (w384 s2: 381 ms, ~7.6 steps of 50 ms to commit) is
  hesitation about *evidence*, not compute — per-step CPU cost is below.
- The w512 false-stop and scenario cells (0.211 hesitation, 0.059 noise, …) are
  averages over two never-stop seeds that cannot false-stop and one that does —
  do not read them as "wider false-stops less".

## CPU inference latency (single core, 40-step window recompute)

`dmel/models.bench_inference`, single core, 300 steps, one policy step with window
recompute over ≤40 frames — the same measurement approach that produced the 6.24 ms
baseline figure.

| Arm | p95 per step | < 10 ms budget |
|---|---|---|
| 6×256 | 6.21 ms | **meets** |
| 6×384 | 17.55 ms | ✗ 1.8× over |
| 6×512 | 28.35 ms | ✗ 2.8× over |

Only 6×256 fits the deployment requirement, and it reproduces the baseline figure
(6.24 ms) within measurement noise. 6×384 fails outright regardless of quality.

## Repro

All commands run from the repo root with the project venv active. Dataset and
frontend are frozen; `PYTHONHASHSEED`/corpus seed make every step deterministic.

```bash
# 1. Regenerate the deterministic pilot corpus + splits (840 samples / 3.12 h).
python -m dmel.data.generate            # pilot preset

# 2. Precompute token caches and upload to the Modal volume (dmel-jev-data).
.venv/bin/modal volume put dmel-jev-data data/pilot /pilot

# 3. Train one (width, seed) arm on Modal L4 — identical protocol, per run:
.venv/bin/modal run dmel/training/train_modal.py \
  --arm transformer --epochs 5 --seeds 1 --seed 0 \
  --d-model 256 --nhead 8 --dim-feedforward 640 \
  --no-generate-synthetic --data-root /data/pilot \
  --out-dir /runs/capacity_sweep/w256/seed_0
# repeat with --seed {1,2}; then the same for 384/12/960 and 512/16/1280.
# All nine runs were dispatched in parallel (no reuse of earlier runs).

# 4. Evaluate one checkpoint: 9-threshold val sweep + selected-point run.
python -m dmel.training.capacity_sweep --widths 256 --seeds 0 \
  --runs-root <local-mount-of-dmel-jev-runs> --out-dir reports/capacity_sweep/eval

# 5. Pool runs into mean ± sd + markdown table (aggregate.json / aggregate_table.md).
python -m dmel.training.capacity_sweep --aggregate \
  --widths 256 384 512 --seeds 0 1 2 --out-dir reports/capacity_sweep/eval

# 6. CPU latency bench per width.
python -m dmel.models.bench_inference --arms transformer --steps 300 \
  --d-model 256 --nhead 8 --dim-feedforward 640        # then 384/12/960, 512/16/1280
```

Artifacts land under `reports/capacity_sweep/eval/w{width}_seed{seed}/`:
`sweep.json` (9-threshold frontier), `metrics.json` (selected-point run, with
`meta.selected_stop_threshold`), `decisions.jsonl` (per-sample decision log);
`aggregate.json` / `aggregate_table.md` sit one level up. Checkpoints live on the
`dmel-jev-runs` Modal volume under `capacity_sweep/w{width}/seed_{seed}/` following
the existing `transformer/seed_{n}` layout.

A checkpoint whose frontier has zero predicted stop frames at every threshold is
recorded with `selected_stop_threshold: null` and `no_operating_point: true` (its
row metrics are persisted verbatim); `aggregate` carries
`no_operating_point_seeds` per width.

## Cost

- Training: nine fresh Modal L4 runs, 5 epochs each — recorded wall times 562/570/570 s
  (w256), 814/805/799 s (w384), 1098/1120/1162 s (w512): **7,500 GPU-seconds total ≈
  $1.67** at the stated L4 rate ($0.000222/s). This is an upper-bound estimate from
  run wall time, not the Modal itemized bill.
- Evaluation (9 checkpoints × 10 harness passes, streaming, CPU-only in-sandbox):
  $0 GPU. Total sweep spend stays far under the $30 guard.
