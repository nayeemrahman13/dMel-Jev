# dMel-Jev

Learned barge-in control for voice agents: a small causal model over dMel audio tokens decides KEEP vs STOP_TTS every 50 ms while the agent is speaking. POC thesis: dMel + a tiny causal model can beat a VAD + heuristic stack on barge-in, measured as interruption recall, per-scenario false-stop rate, and StopLatency.

## Status

Implementation PRs (data engine, dMel features, VAD baseline, LSTM/transformer policies, eval harness) are landing in nayeemrahman13/JeVAD under dmel/ first; they get transplanted here as the initial code drop. After that, all development (corpus scale-up to 10-20h, Modal GPU training, checkpoints) happens in this repo. JeVAD keeps only the runtime-side integration.

The data and interface contract (shard format, label semantics, scenario mix, policy interface, eval definitions) lands at docs/dmel_data_contract.md with the transplant.
