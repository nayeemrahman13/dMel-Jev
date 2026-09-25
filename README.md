# dMel-Jev

Learned barge-in control for voice agents: a small causal model over dMel audio tokens decides KEEP vs STOP_TTS every 50 ms while the agent is speaking. POC thesis: dMel + a tiny causal model can beat a VAD + heuristic stack on barge-in, measured as interruption recall, per-scenario false-stop rate, and StopLatency.

This repo is the self-contained home of the dMel POC — corpus generation, feature tokenizer, policies (baseline + learned), training, and evaluation. It is a transplant of the `dmel/` tree from the upstream JeVAD monorepo at JeVAD main commit `84f52a5` (contract-v1 workstreams: data, features, baselines, models/training, eval). JeVAD keeps its own copy untouched for later runtime-side integration; this repo imports nothing from JeVAD.

## Contract

The data & interface contract (v1) is [`docs/CONTRACT.md`](docs/CONTRACT.md) — shard format, label semantics, splits, scenario mix, the common `BargeInPolicy` interface, metric definitions, and threats to validity. Re-read it before changing any of those surfaces.

**V1 scope:** while the agent is speaking, policies output `KEEP` or `STOP_TTS` every 50 ms. Turn-end detection / turn commit is out of scope.

## Layout

| Path | What it is |
|---|---|
| `dmel/data/` | Deterministic corpus generator (espeak-ng synthesis, augmentation, labels, splits, manifests). Committed fixtures in `dmel/data/fixtures/` cover every scenario in ~10–20 s of audio. |
| `dmel/features/` | Config-driven dMel tokenizer: log-mel → 4-bit intensity quantization → 50 ms token steps. Includes the leakage-linter guardrail test. |
| `dmel/baselines/` | Arm A: Silero VAD + duration-threshold heuristic, calibration (eval step zero), offline runner, validation-split operating-point selection. |
| `dmel/models/` + `dmel/training/` | Arms B (2-layer LSTM) and C (6×256 causal transformer); torch training with config, ≥3 seeds, checkpoints; Modal GPU runner. |
| `dmel/eval/` | Harness: per-event recall, overlap-anchored decision latency, premature/false-stop rates, deployment hazards, bootstrap CIs, A–E markdown report. |
| `dmel/runtime/` | Vendored runtime assets (see below). |
| `dmel/constraints.txt` | Shared dependency floors — all areas install with `-c dmel/constraints.txt` so arms train on comparable versions. |
| `docs/CONTRACT.md` | The contract, transplanted. |

Vendored runtime assets (`dmel/runtime/`; provenance rules in its README): the espeak-ng `EspeakTTS` adapter, `Pcm` helpers, the Silero VAD v5 ONNX wrapper (512-sample windows + 64-sample context) with the ONNX binary at `dmel/runtime/models/silero_vad.onnx`, and two committed test speech assets. They are behavior-identical copies of the upstream runtime's pieces — the determinism claim depends on keeping them identical.

## Setup

Requires Python ≥ 3.12 and the `espeak-ng` system binary:

```bash
apt-get install espeak-ng            # or: brew install espeak-ng (macOS)
python -m venv .venv && source .venv/bin/activate
pip install -e .[dev]                # core deps + pytest
pip install -e .[torch] --extra-index-url https://download.pytorch.org/whl/cpu   # learned arms
```

Individual areas can instead be installed from their own `requirements.txt` files against the shared pin file, e.g. `pip install -r dmel/baselines/requirements.txt -c dmel/constraints.txt`.

## Tests

```bash
pytest                                # whole dmel suite (root testpaths: dmel)
```

The suite covers every area plus the contract guardrails: the leakage linter (features), the shuffled-label control (models), and the fixture smoke test asserting action-head recall > 0. Committed fixtures make the full suite runnable without generating any corpus. The Silero-backed baseline tests additionally need `onnxruntime` (installed with the root deps).

## Generate the pilot corpus

Deterministic: identical CLI args + seed ⇒ byte-identical outputs.

```bash
python -m dmel.data.generate --preset pilot     # writes shards under data/pilot/ (gitignored)
python -m dmel.data.generate --preset main      # 10–20 h scale-up
```

Fixtures are committed (`dmel/data/fixtures/`) and are what the test suite runs on. Use `python -m dmel.features.precompute --shard data/pilot/shard-00001` to materialize dMel token caches for a shard.

## Train arms B/C

Local (CPU or GPU):

```bash
python -m dmel.training.train --config dmel/training/config.yaml [--arm lstm|transformer] [--device cpu]
python -m dmel.training.smoke                    # fixture smoke: both arms must improve over baseline
```

Modal GPU (uploads the `dmel/` tree, mounts the `dmel-jev-data` / `dmel-jev-runs` volumes):

```bash
modal volume put dmel-jev-data data/pilot/shard-00001 /pilot/shard-00001
modal run dmel/training/train_modal.py --arm transformer --epochs 10 --seeds 3 \
    --no-generate-synthetic --data-root /data/pilot
```

Every learned arm trains ≥ 3 seeds on the pilot; reports give mean ± sd. The shuffled-label control must stay at chance — a leak fails the test suite, not the pilot run.

## Run eval

Arm A calibration first (eval step zero), then policy runs, a validation-split threshold sweep, and the comparison report:

```bash
python -m dmel.baselines.calibrate --shard data/pilot/shard-00001
python -m dmel.eval run --policy dmel.baselines.policy:BargeInPolicy \
    --labels-dir data/pilot --out-json out/metrics_a.json --out-decisions out/decisions_a.jsonl
python -m dmel.eval sweep --policy-factory mypkg:HeuristicPolicy \
    --labels-dir data/pilot --thresholds 100,150,200,300,400 --out-json out/sweep_a.json
python -m dmel.eval report --metrics A=out/metrics_a.json --out-md out/report.md
```

`--policy` / `--policy-factory` take `module:ClassName` specs; the factory is called as `factory(threshold_ms)`. Protocol: full operating frontier on the validation split; ONE pre-registered test-split operating point per arm (F-beta β=2 on stop decisions); event-level bootstrap CIs on headline metrics. See `dmel/eval/README.md` for full metric definitions and the decision-log format.
