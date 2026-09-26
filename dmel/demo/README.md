# Live barge-in demo — talk over the agent, A/B two interruption brains

`python -m dmel.demo` plays a scripted paragraph through your speakers while
it listens to your microphone and runs an interruption policy every 50 ms.
Speak over the agent and watch (hear) the playback stop. Toggle between the
two brains live and judge the difference by ear:

- **arm c** — the learned policy: the capacity sweep's 6×256 transformer
  (4,115,846 parameters, committed at `dmel/demo/checkpoints/w256_seed0_best.pt`,
  provenance in `dmel/demo/checkpoints/PROVENANCE.md`).
- **arm a** — the typical speech-to-speech barge-in: Silero VAD + a
  speech-duration threshold (200 ms default), `dmel/baselines`.

Same microphone, same espeak-ng voice, two interruption brains — that is the
whole comparison. There is no LLM, no ASR, no conversational response: this
demo exercises the barge-in policy alone.

## Setup

```bash
pip install -r dmel/demo/requirements.txt -c dmel/constraints.txt
sudo apt install espeak-ng libportaudio2   # Debian/Ubuntu (macOS: brew install portaudio)
```

The checkpoint is committed; if it is missing or corrupt, one command
recovers it from the Modal training volume — or retrains seed 0 with the
same protocol if the volume object is gone:

```bash
python -m dmel.demo.recover_checkpoint
```

## Run

```bash
python -m dmel.demo --arm c        # learned policy (default)
python -m dmel.demo --arm a        # Silero VAD + duration threshold
```

Keys: **c** / **a** switch the interruption brain mid-run, **r** re-speaks
the script, **q** quits. Useful flags: `--voice`, `--rate`, `--script`,
`--device IN[,OUT]`, `--list-devices`, `--verbose` (log every decision),
`--baseline-threshold-ms`, and `--flag-only` (run the learned policy without
dMel tokens — the contract v1.1 fallback rule).

The live log prints action, p_stop, and inter-decision gap per step; stops
flash with a timestamp. Example:

```
·    3.500s  KEEP     p_stop=0.021  (Δ 50 ms, dec 6.2 ms)
⏹    3.550s  STOP_TTS p_stop=0.834  playback aborted (decision 6.4 ms)
```

## Record a session

```bash
python -m dmel.demo --arm c --record sessions/demo
# → sessions/demo-<timestamp>.npz
```

The record holds the exact mic frames (int16, 16 kHz, 50 ms), every per-step
decision (action, probs, timing), and run metadata — enough to score the
session offline against `dmel/eval` later.

## Tests

```bash
pytest dmel/demo/tests -q
```

Includes the **corpus-loopback self-test** (`test_frontend.py::test_stream_decision_skew_is_zero`):
a committed fixture sample is streamed through the live path frame by frame,
and the resulting decisions must match the offline eval path (cache-fed
tokens) on the same sample — the pipeline-integrity check that the streaming
frontend introduces no skew.

## ⚠ Wear a headset

The training mix has **no echo cancellation**. Without headphones, the
agent's playback bleeds into the microphone, reads as user speech, and
causes false stops — in BOTH arms (the baseline's raw VAD cannot separate
playback from user speech either). Headset on: speaker bleed is eliminated
and the comparison is honest.

## ⚠ Honesty note

This policy was trained **purely on espeak-synthesized audio**. Expect
brittleness on real voices: in-domain, the hesitation-commit scenario's
false-stop rate was 0.579. This demo is a sign-of-life — evidence that the
learned policy runs live within budget and reacts to interruption
patterns — **not** a production claim.
