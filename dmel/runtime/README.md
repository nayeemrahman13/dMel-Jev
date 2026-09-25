# dmel.runtime — vendored runtime assets

Behavior-identical copies of the minimal realtime-gateway pieces the dMel POC
consumes, vendored so this repo is self-contained (the upstream monorepo keeps
its own copies for runtime integration; this repo must not import from it).

| Module | Contents |
|---|---|
| `config.py` | Constant subset: `SAMPLE_RATE` (16 kHz), `AGENT_CHUNK_SAMPLES` (512), `VAD_MIN_SILENCE_WINDOWS` (env `DMEL_VAD_SILENCE_WINDOWS`, default 5), vendored asset paths |
| `audio.py` | `Pcm` — int16→float32 scaling and WAV loading with linear resampling to 16 kHz |
| `tts.py` | `EspeakTTS` — espeak-ng WAV synthesis streamed in 512-sample chunks (corpus generation input) |
| `silero.py` | `SileroVAD` (ONNX Runtime), `VadEvent`, `StreamingVAD` protocol, and the 512-sample window / 64-sample context constants |
| `models/silero_vad.onnx` | Silero VAD v5 ONNX model (vendored binary) |
| `assets/*.wav` | Committed test speech assets: `inject_speech.wav` ("hey stop" barge-in), `bench_ack_uh_huh.wav` (backchannel-like burst) |

Provenance: vendored from the upstream realtime gateway at commit `84f52a5`,
unmodified except for import paths and the model default path. Keep the copies
behavior-identical — the determinism claim (identical seeds ⇒ byte-identical
corpora) and the baseline calibration numbers depend on it.

Install: `pip install -r dmel/runtime/requirements.txt -c dmel/constraints.txt`
(the Silero wrapper additionally needs the `espeak-ng` system binary for TTS).
