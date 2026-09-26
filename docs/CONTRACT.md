# dMel-Jev POC — data & interface contract v1.1

Fixed contract for the dMel barge-in POC in this repo (`nayeemrahman13/dMel-Jev`).
Originally written for parallel implementation in the upstream `nayeemrahman13/JeVAD`
monorepo; transplanted here at JeVAD commit `84f52a5` with paths updated to this
repo's layout (see the Transplant note at the end).
**V1 scope:** while the agent is speaking, the policy outputs `KEEP` or `STOP_TTS` every 50 ms.
Turn-end detection / turn commit is explicitly OUT of scope for V1.

**v1 changelog:** all amendments from the red-team review (art_Sqpz5pq7 — findings C1–C4, M1–M8, m1–m3) are adopted. Lines tagged **[v1]** changed or are new. Supersedes v0 everywhere the two differ; re-read this doc before finalizing any PR.

**v1.1 changelog:** capacity-sweep decision recorded (`docs/capacity_sweep.md`, merged 2026-09-26 as `aa78983`). Lines tagged **[v1.1]** changed or are new: the §Models parameter band is superseded by the binding <10 ms single-core CPU p95 sizing constraint (parameter count is descriptive, not a target — measured sizes: LSTM 1.61M, transformer 4.12M), and a new Capacity sweep record section documents the comparison and the RETAIN 6×256 decision. Cadence, label schema, split rules, metrics, and scenario mix are unchanged.

## Ownership map (parallel PRs — never touch another area's files)

| Area | Owns |
|---|---|
| data engine | `dmel/data/**`, this document (`docs/CONTRACT.md`), the `# dmel` block of `.gitignore`, `dmel/constraints.txt` **[v1]** |
| features | `dmel/features/**` |
| baselines | `dmel/baselines/**` |
| models + training | `dmel/models/**`, `dmel/training/**` |
| eval | `dmel/eval/**` |
| vendored runtime assets | `dmel/runtime/**` (PCM helpers, espeak-ng adapter, Silero VAD wrapper + ONNX, test speech assets) |
| runtime integration | lives in the upstream JeVAD repo (`apps/realtime/**`) — a later, separate task there |

Rules: declare dependencies in your own `dmel/<area>/requirements.txt`. **[v1]** All areas install from the shared pin file `dmel/constraints.txt` (torch / numpy / pandas / pyarrow / onnxruntime minimums; created by the data area, read-only for everyone else) so arms train on comparable versions and the determinism claim holds across sandboxes. Use implicit namespace packages (no `__init__.py` at `dmel/` level). Existing repo tests must stay green.

## Audio & frame parameters

- 16 kHz mono, int16 PCM. Frame hop = 50 ms (800 samples).
- Mel: 25 ms window / 10 ms hop, configurable bins (default 80) → 5 subframes per 50 ms step.
- All systems use the same rolling context: last 40 steps (2 s).
- A **sample** = one synthesized conversation clip. Stems stored separately: `<id>.user.pcm`, `<id>.agent.pcm`, `<id>.mix.pcm` (mix = what the microphone hears: agent playback + user + background voice + noise).

## Shard layout

`data/pilot/` and `data/main/` are gitignored (never commit bulk audio):

```
shard-XXXXX/manifest.json                 # seed, voices, augmentation params, scenario counts, split assignment [v1]
shard-XXXXX/<sample_id>.{user,agent,mix}.pcm
shard-XXXXX/<sample_id>.labels.parquet    # one row per 50 ms frame
```

Label columns: `frame_index, t_ms, agent_speaking, speech_present, primary_user, backchannel, interrupt_intent, action ("KEEP"|"STOP_TTS"), earliest_reasonable_stop_ms (int, -1 if none), overlap_onset_ms (int, -1 if none) **[v1]**, scenario (interruption|backchannel|normal_turn|noise|background_speaker|hesitation|hesitation_commit) **[v1 — new scenario value]**.

**[v1]** Manifest additionally records per interruption event: the sampled evidence-window jitter (ms), the shared-prefix (ambiguous-window) length for censored hesitation pairs, and the world variant (`truncated` | `untruncated`). Per sample: `split ∈ {train, val, test}` (see Splits) and `agent_runs` count.

A small fixture set (`dmel/data/fixtures/`, committed) covers every scenario **[v1: including one censored hesitation pair and one truncated-world interruption]** in ~10–20 s total audio.

**Determinism:** identical CLI args + seed ⇒ byte-identical outputs. The pilot corpus is reproduced locally with `python -m dmel.data.generate --preset pilot` — this is how the other parallel tasks get the data.

## Label semantics

- `agent_speaking`: ground truth from the agent stem (exact in synthesis). At deployment this flag comes from TTS playback state — it is legitimately available as a model input. **[v1]** The label column stays exact; the *input* flag is what may be jittered (see Scenario mix).
- `speech_present`: **[v1 — redefined]** acoustic speech activity on the mix from ANY speaker, including the background speaker (1 on background-speaker speech). Attribution lives in `primary_user`; scenario tags carry the eval cuts. The aux head now trains a VAD-like function, which is what makes the baseline calibration (Baselines section) measurable.
- `primary_user`: 1 on primary-user speech frames.
- `backchannel`: 1 on backchannel spans during agent speech.
- `interrupt_intent`: 1 on user frames of a genuine barge-in, from utterance onset until `earliest_reasonable_stop_ms`. **[v1]** These are causal-decision labels: onset is annotated from utterance onset even though commit-vs-hesitation is only knowable in the future; the size of the genuinely ambiguous prefix is quantified in the manifest (see hesitation).
- `overlap_onset_ms`: **[v1 — new]** onset of clear overlapping user speech (the physical event anchor). `-1` when no interruption.
- `earliest_reasonable_stop_ms`: `overlap_onset_ms` + 150–300 ms (jittered) evidence window. From this timestamp through the end of that agent speech run, `action=STOP_TTS`; earlier frames `action=KEEP`. The 150–300 ms window is the policy delay budget, not a latency reference point **[v1]**.
- `hesitation`: user speech with cut-offs/hesitations, `interrupt_intent=0`, action stays KEEP (false-stop trap). **[v1]** Hesitation events come in two kinds: **pure** (never commit) and **censored** — generated as prefix-identical pairs with a hesitation-then-commit partner (new scenario `hesitation_commit`): identical audio up to a censoring point, then the commit variant flows into real overlap and is labeled as an interruption from the commit point. The trap is therefore non-trivial and the ambiguity is measured, not hidden.
- `noise`: non-speech vocal events (cough/breath/bursts), `speech_present=0`.
- When the agent is not speaking, `action` is always `KEEP` (V1 ignores turn-end). **[v1]** In truncated-world samples (below), `agent_speaking` goes low shortly after the stop point; action stays KEEP from there — nothing left to cancel.

## Splits **[v1 — new section, mandatory]**

No random split by sample id. The split is by generation cell, written into the manifest:

- **Voice pairs:** hold out ~30% of (user-voice, agent-voice) pairings for val+test.
- **Template slots:** hold out ~30% of template slots WITHIN each semantic class. Do NOT hold out whole semantic classes — the barge-in lexicon is small and must appear in eval.
- **Augmentation:** at least one augmentation family (use: codec-ish degradation) appears ONLY in eval.
- **Seeds:** train samples draw from seed range [0, 10^6), eval from [10^6, 2×10^6). Augmentation RNG is keyed on sample id, so no augmentation stream crosses the split.
- **Assignment:** split voice-pair × slot cells into train 70% / val 15% / test 15%, stratified by scenario. `val` is for threshold/operating-point selection and model selection; `test` is touched once, for the report.

Guardrails (each is a test that must exist):
- **Leakage linter** (in the features PR): asserts the feature builder consumes only the mix frames plus input flags — never label columns, scenario, or split.
- **Shuffled-label control** (in the models/training PR): training on permuted action labels must yield chance-level eval on the test split — catches any pipeline path where labels reach the inputs.

## Scenario mix (pilot ≈ 2 h; main 10–20 h scales ×5–10)

| scenario | events (pilot) |
|---|---|
| interruption | 200 |
| backchannel | 150 |
| normal_turn | 150 |
| noise | 100 |
| background_speaker | 100 |
| hesitation (pure) | 60 |
| hesitation censored-pair | 40 pairs (→ 40 hesitation + 40 hesitation_commit samples) |

**[v1]** Of the 200 interruption events: ~50% are generated in the **truncated** world (agent audio ends shortly after the stop point — successful stop, `agent_speaking` goes low, action latches KEEP) and ~50% **untruncated** (full agent run continues — failed stop). Both worlds carry matching latching labels.

**[v1]** Onset placement: ~30% of interruption onsets and ~30% of backchannel onsets sit at agent inter-phrase gaps (flag still true — playback state hasn't changed); the rest mid-phrase. The two classes must be acoustically confusable exactly where the model has to use intent rather than continuity cues. (espeak's pause structure is unnatural — placement matters more than acoustics here.)

**[v1]** ~20% of samples contain ≥2 agent speech runs, so cross-run false stops (STOP leaking into the agent's next utterance) exist in the data and can be scored.

Synthesis: espeak-ng TTS with distinct voices per role — use the vendored adapter `dmel/runtime/tts.py` (`EspeakTTS(voice=...)`, behavior-identical copy of the upstream runtime adapter), and follow the upstream monorepo's corpus-generation patterns. Template + slot utterance banks. Deterministic per-sample seeds.

Augmentation: user/agent SNR −6..+6 dB, background voice −18..−10 dB, shaped noise bursts, gain/codec-ish degradation (codec family eval-only per Splits), start jitter. **[v1]** `agent_speaking` input-flag boundaries jittered ±50–100 ms on ~30% of samples (labels stay stem-exact) — the synthesized flag is frame-exact, the deployed one is not (protocol reports playback position at 5 Hz with up to 450 ms client buffering).

## dMel features (`dmel/features`)

log-mel (params above) → per-bin intensity quantization (default 4-bit, configurable) → group into 50 ms steps → token id per (bin-group, level). Emit token ids per 50 ms step; config-driven; deterministic. CLI precompute writes `<sample>.dmel.npy` caches next to shards.

**[v1]** The leakage linter test lives here (see Splits).

## Common policy interface (each area implements its own copy; signatures must match exactly)

```python
class BargeInPolicy:
    def reset(self) -> None: ...
    def step(self, user_pcm_frame: "np.ndarray",  # 800 int16 samples
             dmel_token_ids: "np.ndarray | None",
             *, agent_speaking: bool,
             agent_pcm_frame: "np.ndarray | None" = None,
             speaker_similarity: "float | None" = None) -> dict:
        # returns {"action": "KEEP"|"STOP_TTS", "probs": dict, "t_ms": int}
```

Called once per 50 ms. Policies may latch `STOP_TTS` until `agent_speaking` goes low (hysteresis).

**[v1]** `probs` schema is frozen: learned arms return `{"p_stop": float, "p_interrupt": float, "p_backchannel": float, "p_speech": float, "p_primary_user": float}`; arm A returns `{"p_stop": float}` (its calibrated heuristic score). No extra keys, no per-arm improvisation.

**[v1]** `dmel_token_ids=None` means the feature cache is unavailable: the policy must degrade to a defined flag-only rule (arm A is already flag-only; learned arms fall back to a flag-threshold rule documented in their PR). The eval harness runs both paths on the fixture set.

## Models (`dmel/models`, `dmel/training`)

- Heads: `action(2)` + aux binary heads `speech_present`, `primary_user`, `backchannel`, `interrupt_intent`.
- Loss: `L_action + 0.3·L_interrupt + 0.2·L_backchannel + 0.1·L_speech + 0.1·L_primary_user` (config-driven; action dominates). **[v1]** Per-head class weighting is specified, not optional: BCE-with-logits `pos_weight` per head from train-split positive rates (capped at 20). The training summary reports per-head train positive rates. A fixture smoke test asserts action-head recall > 0 on the committed fixtures — class collapse must fail CI, not the pilot run.
- Architectures **[v1.1]**: (1) 2-layer LSTM (1.61M params as built); (2) causal transformer ~6 layers × 256 hidden (4.12M params as built — 4,115,846 exactly, per `docs/dmel_geometry.md`). Input: dMel token embeddings + `agent_speaking` scalar; plumbing flags reserved for `speaker_similarity` / agent-stem dMel (ablation arms D/E). **[v1]** The model-input tensor contains ONLY tokens plus flags — never label columns (belt-and-suspenders behind the leakage linter).
- Inference budget (binding sizing constraint) **[v1.1]**: one step (window recompute over ≤40 frames) < 10 ms single-core CPU p95 per 50 ms decision step. Parameter count is descriptive, not a target. The superseded v0 parameter band was derived under a since-replaced 20-bin-group token interpretation; under the per-bin tokenizer (`docs/dmel_geometry.md`) a flattened per-bin input projection would be ~26M params and miss this limit. Sizing is settled by the capacity sweep — see Capacity sweep record.
- Training: torch, AdamW, config.yaml, checkpointing. **[v1]** Every learned arm trains ≥3 seeds on the pilot; the report gives mean ± sd. Modal GPU runner `dmel/training/train_modal.py` for real runs; CPU smoke on fixtures when no Modal creds are in the env. Shuffled-label control test lives here (see Splits).

## Capacity sweep record **[v1.1 — new]**

Full report: `docs/capacity_sweep.md` (committed eval outputs under `reports/capacity_sweep/eval/`; merged to main as `aa78983` on 2026-09-26). Recorded here because it settles the §Models sizing constraint above.

- **Question:** does transformer width improve pilot-corpus policy quality while meeting the step budget? Arms: 6×256 / 6×384 / 6×512, seeds 0/1/2 — nine fresh Modal L4 runs, nothing re-rolled.
- **Protocol invariants (identical across all nine runs):** frozen dMel frontend per `docs/dmel_geometry.md` (80 mel bins → 5 subframes per 50 ms step → 400 IDs, shared 16-level codebook, mean-pooled embedding + `agent_speaking`); deterministic pilot corpus (840 samples / 3.12 h), train split only with selection on validation, test untouched; 5 epochs, batch 32, AdamW 3e-4 (wd 0.01, grad-clip 1.0); action-dominant loss with 0.3/0.2/0.1/0.1 aux weights and per-head BCE pos_weight from train-split positive rates (cap 20); stop threshold swept on validation over the 0.30–0.70 grid, ONE point per checkpoint by max F-beta (β=2).
- **Results (validation, mean ± sd over 3 seeds):**

| Arm | params | CPU p95 / step (<10 ms budget) | First in-window STOP recall |
|---|---|---|---|
| 6×256 | 4,115,846 | **6.21 ms** — only width meeting the budget | 0.347 ± 0.159 |
| 6×384 | 9,000,006 | 17.55 ms — 1.8× over | 0.396 ± 0.178 (within one seed sd of 6×256) |
| 6×512 | 15,768,326 | 28.35 ms — 2.8× over | 0.165 ± 0.286 |

- **Degenerate-collapse handling:** 6×512 seeds 1 and 2 emitted zero predicted stop frames at every swept threshold (never-stop collapse), so no operating point exists. The harness persists those rows with `no_operating_point: true` and `selected_stop_threshold: null` rather than inventing one; the aggregate carries `no_operating_point_seeds` per width. All nonzero 6×512 numbers ride on the single surviving seed.
- **Decision (orchestrator, per the pre-agreed keep/replace rule):** **RETAIN the 6×256, 4,115,846-parameter transformer** as the V1 architecture — width buys no measurable quality on this corpus (6×384 within seed noise of 6×256; 6×512 unstable) and both wider arms miss the latency limit. No token flattening; the 50 ms cadence is unchanged. Pilot numbers remain a pipeline sign-of-life (see Eval); architecture ranking re-opens only on the main corpus.

## Baselines (`dmel/baselines`)

Silero VAD — the ONNX is vendored at `dmel/runtime/models/silero_vad.onnx` with the wrapper at `dmel/runtime/silero.py` (16 kHz, 512-sample windows + 64-sample context; buffer internally). Barge-in heuristic: `agent_speaking ∧ speech ∧ speech_duration > threshold` (default 200 ms) → `STOP_TTS`, with hangover/hysteresis. Threshold must be sweepable.

**[v1] Calibration is eval step zero.** Run Silero against the corpus and report speech-detection F1 per scenario against the acoustic reference (frames where the user or background speaker is talking — after the `speech_present` v1 redefinition this is a meaningful target). If user-speech detection recall < 0.9, add a kokoro-voiced calibration slice before trusting the A-vs-B/C comparison (the kokoro TTS adapter lives in the upstream JeVAD runtime, not vendored here) and carry the number into the report. An uncalibrated baseline makes the headline comparison a corpus artifact.

**[v1]** Arm A sweeps its threshold on the validation split like everyone else — see Eval, operating points.

## Eval (`dmel/eval`)

Per interruption event:
- **Event active** **[v1]** = `[user onset, earliest_reasonable_stop_ms + 500 ms grace]`.
- Recall: STOP_TTS emitted while the event is active. **[v1]** First STOP per event is the scored one.
- Premature-stop rate: **[v1 — split in two]** stops >100 ms before `earliest_reasonable_stop` during genuine user overlap (early but evidence-responsive) vs stops during non-overlap (spurious). Separate rows.
- **[v1]** Decision latency = `t_pred_stop − overlap_onset_ms` (physical anchor — NOT anchored to the jittered reference, which a prior-matching policy can game by stopping at the conditional mean). Median + P95. Standing diagnostic: per-event latency correlated with the sampled jitter magnitude — an evidence-following policy is independent of the jitter draw; a prior-matcher tracks it. Latency is never reported as a standalone column over handled events: never-stopped events are censored (imputed at infinity) and the never-stopped rate prints in the same cell.
- **[v1]** Estimated end-to-end latency column = decision latency + measured cancel-path constants from the realtime harness (`vad_ms`, `cancel_send_ms`, `audio_stop_ms` — measured fields of the upstream JeVAD harness's protocol events module; cite values + retrieval date in the report). V1 ranks policies only above the pipeline stop floor; set the evidence-window design target from measured physics.
- Never-stopped rate.

False-stop rates by scenario: backchannel, hesitation (pure), hesitation-then-commit **[v1 — its own row: interrupts labeled from the commit point]**, noise, background_speaker (STOP_TTS emitted during agent speech with no active interruption event). **[v1]** Additionally: cross-run false stops (STOP during a later agent run, on the ≥2-runs subset) and STOP emitted during agent silence — both are deployment hazards, scored as their own metrics even though a stray STOP is a no-op in V1 semantics.

**[v1] Protocol (identical for A, B, C):** every arm reports a full swept operating frontier on the validation split; test-split numbers are reported at ONE pre-registered operating point per arm, selected on validation (maximize F-beta on stop decisions, β=2 — recall-weighted). No eval-set tuning for anyone.

**[v1] Uncertainty:** event-level bootstrap CIs (95%, 10k resamples) on headline metrics; learned arms report mean ± sd across ≥3 seeds. Any "B beats C" claim must have a CI excluding zero. Explicit statement for the report: the pilot is a pipeline sign-of-life; architecture ranking happens on the main corpus (×5–10 events).

Output: json + markdown comparison table — arm A (VAD+heuristic), B (dMel+LSTM), C (dMel+transformer), D/E placeholders.

## Threats to validity (documented, not fixed in V1) **[v1 — new]**

- The mix sums agent playback with no room transfer function and no AEC residue — separating user from agent by timbre is easier here than in production.
- Half-duplex devices make the policy moot by physics; unrepresented.
- Agent self-interrupts (orchestrator cancels its own playback) are unrepresented; harmless in V1 (STOP during silence is a no-op) but the eval would not see a policy that learns spurious associations around flag drops.

- Main-corpus wishlist: a room-convolution / AEC-residue augmentation family.

**Load-bearing assumption** (must open the final report): the synthetic generator defines ground truth, so eval measures policy-vs-labeler agreement, not real-world barge-in quality. What survives that circularity: arm-vs-arm relative comparisons on detection behavior, scenario-conditioned claims, and architecture ranking on a clean split. What does not: absolute StopLatency as a deployment prediction and premature-stop as a real-world quality claim — the report carries the joint latency-plus-misses table and the baseline calibration number as standing evidence. Mitigations: held-out augmentation, human spot-check listening, real-call replay later.

## Transplant note (dMel-Jev, September 2026)

This document was authored in the upstream JeVAD monorepo and moved here with
the dmel/ tree at JeVAD main commit `84f52a5`. Changes made in transit —
contract semantics are untouched:

- Repo paths updated to this repo's layout; upstream-only pieces (runtime
  integration, kokoro TTS, the cancel-path protocol events module) are
  described as upstream references, not repo paths.
- Runtime assets the POC consumes are vendored under `dmel/runtime/`
  (behavior-identical copies; provenance in `dmel/runtime/README.md`).
- The parallel-PR ownership map doubles as the area map; in this repo the
  areas share one tree and one test suite (`pytest`, `testpaths = dmel`).
- The data engine's doc copy (`docs/dmel_data_contract.md`) is replaced by
  this file (`docs/CONTRACT.md`).

