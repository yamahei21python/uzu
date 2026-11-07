#!/usr/bin/env python3
"""Watch several Kaggle kernels and download their output when they finish.

Ten CPU lanes run at once; babysitting them by hand does not scale. This polls a
list of kernels, fetches output once per finished kernel, and writes a summary.

    python scripts/watch_kernels.py \
        --kernel yamahei21python/uzu-lane-runner \
        --kernel yamahei21python/uzu-lane-sweep-1 \
        --interval 300

    python scripts/watch_kernels.py --once          # single pass (cron friendly)

Per-kernel state lives in ``logs/kernel_watch.json`` so re-running does not
download the same output twice.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STATE = ROOT / "logs" / "kernel_watch.json"
TERMINAL = ("COMPLETE", "ERROR", "FAILED", "CANCEL", "CANCELLED")


def slug(kernel: str) -> str:
    return kernel.rstrip("/").split("/")[-1]


def is_terminal(status: str) -> bool:
    upper = status.upper()
    return any(marker in upper for marker in TERMINAL)


def load_state(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def save_state(path: Path, state: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")


def kernel_status(kernel: str) -> str:
    result = subprocess.run(
        ["kaggle", "kernels", "status", kernel], text=True, capture_output=True, check=False
    )
    text = (result.stdout or "") + (result.stderr or "")
    line = next((item.strip() for item in text.splitlines() if "status" in item.lower()), text.strip())
    return line


def download(kernel: str, out_dir: Path) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(
        ["kaggle", "kernels", "output", kernel, "-p", str(out_dir)],
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        print(f"  output download failed: {result.stderr.strip()[:200]}")
    return result.returncode


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--kernel", action="append", required=True)
    parser.add_argument("--out-root", type=Path, default=Path("/private/tmp/uzu-kernels"))
    parser.add_argument("--state", type=Path, default=STATE)
    parser.add_argument("--interval", type=float, default=300.0)
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--max-hours", type=float, default=14.0)
    args = parser.parse_args()

    state = load_state(args.state)
    started = time.time()
    while True:
        pending = 0
        for kernel in args.kernel:
            entry = state.setdefault(kernel, {"status": "", "downloaded": False})
            if entry.get("downloaded"):
                continue
            status = kernel_status(kernel)
            entry["status"] = status
            marker = "done" if is_terminal(status) else "running"
            print(f"{time.strftime('%H:%M:%S')} {slug(kernel):28s} {marker:8s} {status[:80]}")
            if is_terminal(status):
                out_dir = args.out_root / slug(kernel)
                if download(kernel, out_dir) == 0:
                    entry["downloaded"] = True
                    entry["out_dir"] = str(out_dir)
            else:
                pending += 1
        save_state(args.state, state)

        if args.once or pending == 0:
            break
        if (time.time() - started) / 3600 > args.max_hours:
            print("max-hours reached; exiting")
            break
        time.sleep(args.interval)

    finished = sum(1 for entry in state.values() if entry.get("downloaded"))
    print(f"downloaded {finished}/{len(args.kernel)}")


if __name__ == "__main__":
    main()
