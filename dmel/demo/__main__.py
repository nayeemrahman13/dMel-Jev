"""Live barge-in demo CLI: talk over a speaking agent, A/B two interruption brains.

    python -m dmel.demo --arm c          # learned 6x256 policy (capacity sweep)
    python -m dmel.demo --arm a          # Silero VAD + duration threshold (typical S2S)

Keys: c/a switch arm live, r re-speak, q quit. See dmel/demo/README.md
(headset requirement + honesty note).
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime

from dmel.demo import policies
from dmel.demo.keys import KeyboardListener
from dmel.demo.mic import MicStream, list_devices
from dmel.demo.player import DEFAULT_SCRIPT, AgentPlayback
from dmel.demo.recorder import save_session
from dmel.demo.session import DecisionSession, StepDecision

HEARTBEAT_EVERY = 10  # log every Nth KEEP decision (2 Hz at 50 ms steps)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m dmel.demo",
        description="Live barge-in demo: interrupt a speaking agent with your mic; "
        "toggle between the learned dMel policy (arm c) and the Silero+duration "
        "baseline (arm a).",
    )
    parser.add_argument(
        "--arm",
        default=policies.ARM_LEARNED,
        help="starting interruption brain: c = learned 6x256 transformer, a = Silero VAD + duration threshold",
    )
    parser.add_argument("--voice", default="en-us", help="espeak-ng voice (default en-us)")
    parser.add_argument("--rate", type=int, default=165, help="espeak-ng words per minute (default 165)")
    parser.add_argument("--script", help="agent script text (default: built-in paragraph)")
    parser.add_argument("--script-file", help="agent script from a text file")
    parser.add_argument(
        "--record",
        nargs="?",
        const="dmel-demo-session",
        metavar="PATH",
        help="dump mic frames + per-step decisions + timestamps to PATH-<timestamp>.npz",
    )
    parser.add_argument(
        "--device",
        help="PortAudio device: input id, or 'in,out' pair (see --list-devices)",
    )
    parser.add_argument("--list-devices", action="store_true", help="list audio devices and exit")
    parser.add_argument(
        "--flag-only",
        action="store_true",
        help="run the learned policy WITHOUT dMel tokens (contract v1.1 flag-only fallback)",
    )
    parser.add_argument(
        "--baseline-threshold-ms",
        type=float,
        default=200.0,
        help="arm A speech-duration threshold in ms (default 200)",
    )
    parser.add_argument("--verbose", action="store_true", help="log every decision, not just stops + heartbeats")
    return parser.parse_args(argv)


def _parse_device(spec: str | None) -> tuple[int | None, int | None]:
    if not spec:
        return None, None
    parts = [p.strip() for p in spec.split(",")]
    if len(parts) > 2 or not all(p.isdigit() or not p for p in parts):
        raise ValueError(f"--device expects 'IN' or 'IN,OUT' numeric ids, got {spec!r}")
    inp = int(parts[0]) if parts[0] else None
    out = int(parts[1]) if len(parts) > 1 and parts[1] else None
    return inp, out


def _resolve_script(args: argparse.Namespace) -> str:
    if args.script and args.script_file:
        raise SystemExit("pass either --script or --script-file, not both")
    if args.script_file:
        with open(args.script_file, encoding="utf-8") as handle:
            return handle.read().strip()
    return args.script or DEFAULT_SCRIPT


class _Stats:
    """Per-arm session counters for the summary line."""

    def __init__(self) -> None:
        self.per_arm: dict[str, dict[str, float]] = defaultdict(
            lambda: {"decisions": 0, "stops": 0, "dec_ms": 0.0}
        )

    def add(self, decision: StepDecision, arm: str) -> None:
        entry = self.per_arm[arm]
        entry["decisions"] += 1
        entry["dec_ms"] += decision.dec_ms
        if decision.stop_fired:
            entry["stops"] += 1

    def line(self, arm: str) -> str:
        entry = self.per_arm.get(arm, {"decisions": 0, "stops": 0, "dec_ms": 0.0})
        mean = entry["dec_ms"] / entry["decisions"] if entry["decisions"] else 0.0
        return (
            f"{entry['decisions']} decisions, {int(entry['stops'])} stop(s), "
            f"mean decision {mean:.1f} ms"
        )


def run_demo(args: argparse.Namespace) -> int:
    from dmel.demo.mic import _sounddevice  # noqa: F401 — fails fast (with install hint) if no PortAudio

    arm = policies.resolve_arm(args.arm)
    script = _resolve_script(args)
    device_in, device_out = _parse_device(args.device)
    recording = args.record is not None
    verbose = args.verbose
    stats = _Stats()

    policy = policies.build_policy(
        arm if not args.flag_only else policies.ARM_LEARNED,
        baseline_threshold_ms=args.baseline_threshold_ms,
    )
    if args.flag_only:
        arm = policies.ARM_LEARNED  # flag-only only meaningful for the learned arm

    def flash_stop(decision: StepDecision) -> None:
        player.stop_now()
        print(
            f"\n⏹ {decision.t_ms / 1000:8.3f}s  STOP_TTS  p_stop={decision.p_stop:.2f}  "
            f"[{policies.describe_arm(arm)}]  playback aborted "
            f"(decision {decision.dec_ms:.1f} ms)",
            flush=True,
        )

    session = DecisionSession(
        policy,
        arm_label=arm,
        on_stop=flash_stop,
        flag_only=args.flag_only,
        keep_frames=recording,
    )

    print("dMel live barge-in demo")
    print(f"  arm:      {arm} — {policies.describe_arm(arm)}")
    if args.flag_only:
        print("  mode:     FLAG-ONLY (no dMel tokens — contract fallback rule)")
    print(f"  script:   {len(script)} chars, espeak-ng {args.voice} @ {args.rate} wpm")
    print("  keys:     c / a switch interruption brain · r re-speak · q quit")
    print()
    print("  ⚠ Wear a headset. The training mix has no echo cancellation —")
    print("    speaker bleed into the mic reads as user speech and causes false stops.")
    print("  ⚠ Honesty note: this policy was trained purely on espeak-synthesized")
    print("    audio; expect brittleness on real voices (in-domain hesitation-commit")
    print("    false-stop rate was 0.579). This demo is a sign-of-life, not a production claim.")
    print()

    arm_history: list[str] = [arm]
    last_wall: dict[str, float | None] = {"wall": None}
    script_done = False

    def log(decision: StepDecision) -> None:
        delta = (
            decision.wall_s - last_wall["wall"] if last_wall["wall"] is not None else 0.0
        )
        last_wall["wall"] = decision.wall_s
        heartbeat = decision.step_index % HEARTBEAT_EVERY == 0
        if decision.stop_fired or verbose or heartbeat:
            mark = "⏹" if decision.stop_fired else "·"
            print(
                f"{mark} {decision.t_ms / 1000:8.3f}s  {decision.action:<8s} "
                f"p_stop={decision.p_stop:.3f}  "
                f"(Δ {delta * 1000:.0f} ms, dec {decision.dec_ms:.1f} ms)",
                flush=True,
            )

    keys = KeyboardListener()
    with MicStream(device=device_in) as mic, AgentPlayback(
        voice=args.voice, rate=args.rate, device=device_out
    ) as player:
        try:
            player.speak(script)
            while True:
                if (key := keys.poll()) is not None:
                    if key == "q":
                        break
                    if key in ("a", "c") and key != arm:
                        arm = key
                        policy = policies.build_policy(
                            arm, baseline_threshold_ms=args.baseline_threshold_ms
                        )
                        session.reset(policy, arm_label=arm)
                        arm_history.append(arm)
                        last_wall["wall"] = None
                        print(f"\n⟳ interruption brain → {arm}: {policies.describe_arm(arm)}", flush=True)
                    elif key == "r":
                        player.stop_now()
                        player.speak(script)
                        script_done = False
                        print("\n▶ re-speaking script", flush=True)
                    elif key == "\x03":  # Ctrl-C read as a raw char in raw mode
                        break

                frame = mic.read_frame()
                agent_speaking = player.speaking
                for decision in session.push(frame, agent_speaking=agent_speaking):
                    stats.add(decision, arm)
                    log(decision)
                speaking = player.tick()

                if script_done is False and agent_speaking and not speaking:
                    script_done = True
                    stops = int(stats.per_arm[arm]["stops"])
                    how = "stopped by barge-in" if stops else "script finished"
                    print(
                        f"\n— {how}. [r] re-speak · [c]/[a] switch arm · [q] quit —",
                        flush=True,
                    )
        except KeyboardInterrupt:
            print("\ninterrupted — shutting down", flush=True)
        finally:
            keys.close()
            if recording:
                stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
                path = f"{args.record}-{stamp}.npz"
                written = save_session(
                    path,
                    frames=session.frames,
                    decisions=session.decisions,
                    meta={
                        "arms": arm_history,
                        "final_arm": arm,
                        "flag_only": args.flag_only,
                        "voice": args.voice,
                        "rate": args.rate,
                        "script_source": "file" if args.script_file else ("cli" if args.script else "builtin"),
                        "baseline_threshold_ms": args.baseline_threshold_ms,
                        "stops_total": int(sum(int(s["stops"]) for s in stats.per_arm.values())),
                    },
                )
                print(f"\nsession recorded → {written}", flush=True)
            for seen_arm in dict.fromkeys(arm_history):
                print(f"  arm {seen_arm}: {stats.line(seen_arm)}")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.list_devices:
        print(list_devices())
        return 0
    return run_demo(args)


if __name__ == "__main__":
    raise SystemExit(main())
