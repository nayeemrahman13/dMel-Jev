# dmel/eval — evaluation harness (contract v1)

Evaluates any barge-in policy that conforms to the contract's `BargeInPolicy`
interface against contract-labeled shards. This area depends on no other
`dmel/` area's code: policies arrive as `module:ClassName` specs at runtime and
feature caches are consumed by path only.

## Metric definitions (contract v1)

- **Event active window** = `[user onset, earliest_reasonable_stop_ms + 500 ms grace]`.
- **Recall** = events whose first in-window `STOP_TTS` exists / total events.
  The first STOP per event is the scored one; later in-window stops are ignored.
- **Decision latency** = `t_pred_stop - overlap_onset_ms` — anchored to the
  physical overlap anchor, NOT the jittered evidence reference (a policy that
  matches the 150–300 ms jitter prior could game that). Median + P95 over
  handled events; never-stopped events are censored at infinity and the
  never-stopped rate prints in the same cell as latency (never a standalone
  handled-events column).
- **Premature stops** (first stop >100 ms before the stop point) split into
  *during overlap* (early but evidence-responsive) vs *during non-overlap*
  (spurious) via the overlap anchor.
- **False stops** = `STOP_TTS` during agent speech with no active interruption
  event, per scenario (backchannel, hesitation, hesitation_commit, noise,
  background_speaker). hesitation_commit false stops cover the pre-commit trap.
- **Deployment hazards**: cross-run false stops (STOP leaking into the agent's
  next utterance, scored on the >=2-agent-runs subset) and STOP during agent
  silence.
- **Stop-decision confusion** (frame-level vs the ground-truth `action`
  column) with F-beta(beta=2) — the pre-registered operating-point statistic.
- **Jitter-latency diagnostic**: per-event decision latency correlated with the
  sampled jitter (`erts - overlap onset`). Evidence-following policies are
  independent of the jitter draw; prior-matchers track it.
- **Uncertainty**: event-level bootstrap CIs (95%, 10k resamples) on headline
  metrics.

## Estimated end-to-end latency (opt-in, measured values only)

Supply the **measured** cancel-path constants from the realtime harness
(`vad_ms`, `cancel_send_ms`, `audio_stop_ms`) and every latency stat is shifted
by their sum, with the source and retrieval date recorded in the report:

```bash
python -m dmel.eval run --policy mypkg:MyPolicy --labels-dir data/pilot \
  --cancel-path-constants "vad_ms=3.2,cancel_send_ms=1.1,audio_stop_ms=12.0" \
  --out-json out/metrics_a.json
```

Values are per-event measured fields of the upstream realtime harness's
cancel path — measure them from the
running harness and pass what you measured; the tool refuses to invent
defaults and rejects incomplete/unknown keys.

## Commands

```bash
# Stream one policy over a shard; emit metrics json + decision log
python -m dmel.eval run --policy mypkg:MyPolicy --labels-dir data/pilot \
  --out-json out/metrics_a.json --out-decisions out/decisions_a.jsonl

# Contract flag-only path: policies receive dmel_token_ids=None
python -m dmel.eval run --policy mypkg:MyPolicy --labels-dir dmel/data/fixtures \
  --ignore-dmel-cache --no-bootstrap

# Threshold frontier for a policy factory (factory(threshold_ms) -> policy)
python -m dmel.eval sweep --policy-factory mypkg:HeuristicPolicy \
  --labels-dir data/val --thresholds 100,150,200,300,400 --out-json out/sweep_a.json

# Markdown A-E report from per-arm metrics (repeat --metrics per seed)
python -m dmel.eval report \
  --metrics A=out/metrics_a.json --metrics B=out/metrics_b_s1.json \
  --metrics B=out/metrics_b_s2.json --out-md out/report.md

# Re-score a pre-generated decision log without re-running the policy
python -m dmel.eval report --metrics A=out/decisions_a.jsonl \
  --recompute-from-log out/decisions_a.jsonl --labels-dir data/pilot
```

## Decision-log format

One JSON object per 50 ms step (every step, not just stops):

```
{"sample_id": "shard00042-a013", "frame_index": 120, "t_ms": 6000, "action": "STOP_TTS"}
```

The reader accepts common aliases (`sample`/`clip_id`, `frame`/`step`,
`pred_action`/`action_pred`/`policy_action`, `t`) so logs emitted by other
parallel tasks can be re-scored directly; anything unparseable raises instead
of silently skewing metrics.

## Protocol (identical for arms A, B, C)

Full swept operating frontier on the **validation** split; test-split numbers
at ONE pre-registered operating point per arm, selected on validation by
maximizing F-beta(beta=2) on stop decisions (ties -> lowest threshold; the
sweep table prints the selected point). Bootstrap CIs are seeded and
deterministic. Learned arms report mean ± sd across >=3 training seeds by
passing one metrics file per seed.

## Tests

```bash
pytest dmel/eval/tests            # metric math + harness pipeline (hand-computed cases)
pytest                            # whole dmel suite (root testpaths: dmel)
```
