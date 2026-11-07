#!/usr/bin/env python3
"""Check how many Kaggle sessions are running before pushing another one.

We accidentally had six CPU kernels alive. Kaggle silently drops whichever it
wants, so the rule is: never exceed the per-account cap; wait in line instead.

    python scripts/kaggle_slots.py                 # owner kernels from tasks/kernels.json
    python scripts/kaggle_slots.py --extra yamahei21python/uzu-foo
    python scripts/kaggle_slots.py --max 5 --json out.json

Exit code is 1 when the account is full, so a shell guard can refuse the push.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TASKS = ROOT / "tasks" / "kernels.json"

TERMINAL = ("COMPLETE", "ERROR", "FAILED", "CANCEL", "CANCELLED")


def status_of(kernel: str) -> str:
    try:
        result = subprocess.run(
            ["kaggle", "kernels", "status", kernel], text=True, capture_output=True, check=False
        )
    except OSError:
        return "unknown"
    text = (result.stdout or "") + (result.stderr or "")
    line = next((item.strip() for item in text.splitlines() if "status" in item.lower()), "")
    return line or "unknown"


def is_running(status: str) -> bool:
    upper = status.upper()
    if any(marker in upper for marker in TERMINAL):
        return False
    return "RUNNING" in upper or "QUEUED" in upper


def count_running(statuses: list[str]) -> int:
    return sum(1 for status in statuses if is_running(status))


def owner_entries() -> list[dict]:
    if not TASKS.exists():
        return []
    payload = json.loads(TASKS.read_text(encoding="utf-8"))
    return [entry for entry in payload.get("kernels", []) if entry.get("owner") == "owner"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--max-cpu", type=int, default=5)
    parser.add_argument("--max-gpu", type=int, default=2)
    parser.add_argument("--extra", action="append", default=[], help="additional kernel refs to count")
    parser.add_argument("--json", type=Path, default=None)
    args = parser.parse_args()

    rows = []
    for entry in owner_entries():
        status = status_of(entry["ref"])
        rows.append(
            {
                "kernel": entry["ref"],
                "kind": entry.get("kind", "cpu"),
                "status": status,
                "running": is_running(status),
            }
        )
    for kernel in args.extra:
        status = status_of(kernel)
        rows.append({"kernel": kernel, "kind": "cpu", "status": status, "running": is_running(status)})

    cpu_running = sum(1 for row in rows if row["running"] and row["kind"] == "cpu")
    gpu_running = sum(1 for row in rows if row["running"] and row["kind"] == "gpu")
    running = cpu_running + gpu_running
    print(f"{'kernel':34s} {'kind':5s} {'running':8s} status")
    for row in rows:
        print(
            f"{row['kernel'].split('/')[-1]:34s} {row['kind']:5s} "
            f"{'yes' if row['running'] else 'no':8s} {row['status'][:50]}"
        )
    print(f"\ncpu {cpu_running}/{args.max_cpu}   gpu {gpu_running}/{args.max_gpu}")

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(
            json.dumps(
                {"cpu_running": cpu_running, "gpu_running": gpu_running,
                 "max_cpu": args.max_cpu, "max_gpu": args.max_gpu, "rows": rows},
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )

    if cpu_running >= args.max_cpu or gpu_running >= args.max_gpu:
        print("account is full: wait for a slot instead of pushing another kernel")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
