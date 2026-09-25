# dmel/baselines — arm A: VAD + heuristic barge-in baseline

Silero VAD (vendored ONNX) + a duration-threshold decision rule, in the
contract `BargeInPolicy` shape: once per 50 ms step, output `KEEP` or
`STOP_TTS` while the agent is speaking.

## Layout

- `heuristic.py` — pure decision state machine: `agent_speaking ∧ speech ∧
  speech_duration > threshold → STOP_TTS`, latched until `agent_speaking`
  goes low. No audio/ONNX deps; unit-testable in isolation.
- `policy.py` — `BargeInPolicy` (= `SileroVADPolicy`): feeds each 800-sample
  int16 contract frame to the vendored `dmel.runtime.silero.SileroVAD`
  (a behavior-identical copy of the upstream runtime wrapper — it buffers to
  512-sample windows + 64-sample
  context and tracks continuous above-threshold speech duration), then applies
  the heuristic. Ignores `dmel_token_ids`; `agent_pcm_frame` /
  `speaker_similarity` are accepted-but-unused D/E plumbing.
- `runner.py` — offline CLI: streams a shard's `*.mix.pcm` + per-frame
  `agent_speaking` labels through the policy, writes per-frame decision logs.
- `shardio.py` — shared shard I/O: mix loading, labels tables (lazy pyarrow),
  contract sample-id parsing, and the per-frame decision stream.
- `calibrate.py` — **eval step zero (v1)**: runs Silero over a shard and
  reports speech-detection precision/recall/F1 **per scenario** against the
  acoustic reference (the `speech_present` label column, which counts speech
  from any non-agent speaker). If user-speech recall < 0.9 it flags that a
  kokoro-voiced calibration slice must be added to the corpus before the
  A-vs-B/C comparison is trusted; the flag and the number travel with the
  report.
- `metrics.py` — pure precision/recall/F-beta helpers.
- `select.py` — **v1 operating-point protocol**: sweeps the stop threshold on
  the **validation split only** (per-sample `split` from the shard manifest)
  and selects ONE point by maximizing F-beta (β=2, recall-weighted) on pooled
  stop decisions; ties break to the largest threshold (fewest false stops at
  equal F-beta). The selected point is pre-registered — test-split numbers are
  reported at it and nowhere else.

## Usage

```bash
pip install -r dmel/baselines/requirements.txt -c dmel/constraints.txt

# The Silero wrapper and test assets are vendored under dmel/runtime/ —
# no runtime package outside this repo is imported.

python -m dmel.baselines.runner --shard data/pilot/shard-00001 \
    [--sample ID ...] [--threshold-ms 200] [--out DIR]
# → <out>/<sample>.armA.jsonl: {"t_ms", "action", "p_stop", "speech_prob",
#   "speech_ms", "latched", "stop_threshold_ms"} per 50 ms frame

python -m dmel.baselines.calibrate --shard data/pilot/shard-00001 \
    [--prob-threshold 0.5] [--out calib.json]      # prints a per-scenario F1 table

python -m dmel.baselines.select --shard data/pilot/shard-00001 \
    [--thresholds 25,50,...] [--split val] [--out selected.json]

pytest dmel/baselines/tests
```

`p_stop` is the frozen v1 probs schema: arm A's calibrated heuristic score —
the duration ratio, saturated at 1.0 exactly when the action is `STOP_TTS`
(forced KEEP while the agent is silent scores 0). A p_stop cutoff sweep in
[0, 1) reproduces duration-threshold sweeps.

Threshold sweeps pass `--threshold-ms` (runner) or `stop_threshold_ms=`
(policy/heuristic). The repo's root pytest config sets `testpaths` to `dmel`,
so a bare `pytest` covers this suite; run it by path for just this area.

## Semantics and caveats

- Speech duration is the VAD's continuous above-threshold (0.5) time in 32 ms
  window quanta; the default 200 ms gate therefore fires ~224–256 ms after
  speech onset (measured 250 ms on the committed `inject_speech` asset).
- Hysteresis: `STOP_TTS` latches while `agent_speaking` stays high, even
  through VAD silence (a mid-barge-in pause must not un-stop playback); it
  clears when `agent_speaking` goes low. The agent-not-speaking case is always
  `KEEP` (V1 ignores turn-end).
- Echo caveat: `user_pcm_frame` is the mic signal, which on the synthetic
  corpus includes the agent's own playback; a raw VAD cannot separate the two,
  so false stops during agent speech are expected and are quantified by the
  eval harness per scenario. `agent_pcm_frame`-based cancellation is
  deliberately out of scope (reserved for ablation arms D/E).
- The committed-asset tests need `onnxruntime` + the vendored ONNX
  (`dmel/runtime/models/silero_vad.onnx`); the fixtures are committed at
  `dmel/data/fixtures/`, so the fixture tests run out of the box.
