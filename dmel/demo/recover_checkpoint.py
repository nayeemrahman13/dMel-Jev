"""One-command checkpoint recovery (or retrain) for the demo's learned arm.

The learned arm loads ``dmel/demo/checkpoints/w256_seed0_best.pt`` — a
byte-for-byte copy of the capacity-sweep run object on the Modal volume
``dmel-jev-runs`` (provenance: dmel/demo/checkpoints/PROVENANCE.md). This
module recreates that file when it is missing or corrupt:

1. already present and matching the recorded sha256 → nothing to do;
2. Modal reachable → ``modal volume get`` the object and verify its sha256;
3. Modal object gone (unreferenced volume objects expire) → retrain seed 0
   with the exact capacity-sweep protocol (``dmel/training/train_modal.py``),
   then fetch the fresh ``best.pt``.

Usage:  python -m dmel.demo.recover_checkpoint [--retrain]
Modal auth: credentials are read from MODAL_TOKEN_ID / MODAL_TOKEN_SECRET
when set, otherwise an existing `modal token` login is used.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

CHECKPOINT_PATH = Path(__file__).resolve().parent / "checkpoints" / "w256_seed0_best.pt"
SHA256 = "690ac00ef378ac227bb191610fbc2e73a4459cba122828ec7878f543dd66d480"
VOLUME = "dmel-jev-runs"
REMOTE_PATH = "/capacity_sweep/w256/seed_0/transformer/seed_0/best.pt"
# Capacity-sweep protocol for w256 seed 0 (docs/capacity_sweep.md).
RETRAIN_ARGS = [
    "modal", "run", "dmel/training/train_modal.py",
    "--arm", "transformer",
    "--epochs", "5",
    "--seeds", "1",
    "--seed", "0",
    "--d-model", "256",
    "--nhead", "8",
    "--dim-feedforward", "640",
    "--out-dir", "/capacity_sweep/w256/seed_0",
]


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _modal_env() -> dict[str, str]:
    env = dict(os.environ)
    token_id = env.get("MODAL_TOKEN_ID") or env.get("SECRET_MODAL_TOKEN_ID")
    token_secret = env.get("MODAL_TOKEN_SECRET") or env.get("SECRET_MODAL_TOKEN_SECRET")
    # modal reads its profile from ~/.modal.toml; seed it from the env when the
    # machine has no stored login (CI / fresh checkout).
    if token_id and token_secret and shutil.which("modal") is not None:
        config = Path.home() / ".modal.toml"
        if not config.exists():
            config.write_text(f"[default]\ntoken_id = {token_id}\ntoken_secret = {token_secret}\n")
    return env


def _volume_get(destination: Path) -> bool:
    if shutil.which("modal") is None:
        print("modal CLI not installed — pip install modal", file=sys.stderr)
        return False
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as tmp:
        tmp_path = Path(tmp.name)
    try:
        result = subprocess.run(
            ["modal", "volume", "get", VOLUME, REMOTE_PATH, str(tmp_path)],
            env=_modal_env(),
            check=False,
        )
        if result.returncode != 0 or not tmp_path.is_file():
            return False
        if sha256_of(tmp_path) != SHA256:
            print(
                f"volume object {REMOTE_PATH} does not match the recorded sha256 "
                "— refusing to install it",
                file=sys.stderr,
            )
            return False
        shutil.move(str(tmp_path), destination)
        return True
    finally:
        tmp_path.unlink(missing_ok=True)


def _retrain_and_fetch() -> bool:
    print("retraining w256 seed 0 with the capacity-sweep protocol (Modal L4)…")
    result = subprocess.run(RETRAIN_ARGS, env=_modal_env(), check=False)
    if result.returncode != 0:
        print("retrain failed — see Modal output above", file=sys.stderr)
        return False
    return _volume_get(CHECKPOINT_PATH)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--retrain", action="store_true", help="retrain seed 0 from scratch instead of fetching")
    args = parser.parse_args(argv)

    if CHECKPOINT_PATH.is_file() and sha256_of(CHECKPOINT_PATH) == SHA256:
        print(f"checkpoint already present and verified: {CHECKPOINT_PATH}")
        return 0

    print(f"checkpoint missing or corrupt at {CHECKPOINT_PATH}")
    if not args.retrain and _volume_get(CHECKPOINT_PATH):
        print(f"recovered from Modal volume {VOLUME}:{REMOTE_PATH} (sha256 verified)")
        return 0
    if _retrain_and_fetch():
        print(f"retrained, fetched, and verified: {CHECKPOINT_PATH}")
        return 0
    print(
        "could not recover the checkpoint — install modal, authenticate "
        "(modal token set), and retry, or retrain manually with: "
        f"{' '.join(RETRAIN_ARGS)}",
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
