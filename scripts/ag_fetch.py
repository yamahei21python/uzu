#!/usr/bin/env python3
"""Fetch Antigravity sandbox artifacts with retries and a log-based fallback.

``fetch`` regularly fails with HTTP 503 "Environment snapshot is temporarily
unavailable". It is transient: the same session usually fetches fine a minute
later. This wrapper retries with exponential backoff and, if the snapshot never
comes back, recovers the agent's final message from the run log so the analysis
is not lost.

Usage:
    python scripts/ag_fetch.py --state /tmp/ag_task.json --out /tmp/ag_task_fetch
    python scripts/ag_fetch.py --state ... --out ... --retries 8 --base-delay 15
    python scripts/ag_fetch.py --log logs/.../antigravity_task.log --out /tmp/ag_task_fetch --recover-only
"""

from __future__ import annotations

import argparse
import json
import random
import subprocess
import sys
import time
from pathlib import Path

DEFAULT_WRAPPER = Path.home() / ".config" / "opencode" / "antigravity" / "antigravity.py"

TRANSIENT = (
    "503",
    "UNAVAILABLE",
    "temporarily unavailable",
    "502",
    "504",
    "deadline",
    "timeout",
    "connection reset",
    "ECONNRESET",
)


def is_transient(output: str) -> bool:
    text = output.lower()
    return any(marker.lower() in text for marker in TRANSIENT)


def json_objects(text: str):
    """Yield every top-level JSON object embedded in a log, in order."""
    decoder = json.JSONDecoder()
    index = 0
    while True:
        start = text.find("{", index)
        if start < 0:
            return
        try:
            value, end = decoder.raw_decode(text, start)
        except json.JSONDecodeError:
            index = start + 1
            continue
        yield value
        index = end


def last_final(log_path: Path) -> str | None:
    if not log_path.exists():
        return None
    final = None
    for payload in json_objects(log_path.read_text(encoding="utf-8", errors="replace")):
        if isinstance(payload, dict) and isinstance(payload.get("final"), str) and payload["final"].strip():
            final = payload["final"]
    return final


def recover_from_log(log_path: Path, out_dir: Path) -> Path | None:
    final = last_final(log_path)
    if not final:
        return None
    target = out_dir / "final.md"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(final + "\n", encoding="utf-8")
    return target


def wrapper_fetch(wrapper: Path, state: Path, out_dir: Path) -> tuple[int, str]:
    command = [sys.executable, str(wrapper), "--state", str(state), "fetch", "--out", str(out_dir)]
    process = subprocess.run(command, text=True, capture_output=True, check=False)
    return process.returncode, (process.stdout or "") + (process.stderr or "")


def fetch_with_retry(
    wrapper: Path,
    state: Path,
    out_dir: Path,
    log_path: Path | None = None,
    retries: int = 6,
    base_delay: float = 15.0,
    max_delay: float = 180.0,
    sleep=time.sleep,
) -> dict:
    """Returns {"status": ok|recovered|failed, "attempts": n, "final": path|None}."""
    last_output = ""
    attempts_made = 0
    for attempt in range(1, retries + 1):
        attempts_made = attempt
        code, output = wrapper_fetch(wrapper, state, out_dir)
        last_output = output
        extracted = out_dir / "extracted"
        if code == 0 and extracted.exists():
            return {"status": "ok", "attempts": attempt, "final": None}
        if not is_transient(output) and code != 0:
            print(f"fetch failed with a non-transient error (attempt {attempt}): {output.strip()[:300]}")
            break
        if attempt < retries:
            delay = min(max_delay, base_delay * (2 ** (attempt - 1)))
            delay *= 1 + random.uniform(-0.1, 0.1)
            print(f"fetch attempt {attempt} failed (503/transient); retrying in {delay:.0f}s", flush=True)
            sleep(delay)

    print("fetch exhausted; trying log-based recovery", flush=True)
    recovered = recover_from_log(log_path, out_dir) if log_path else None
    if recovered:
        print(f"recovered final message from log -> {recovered}", flush=True)
        return {"status": "recovered", "attempts": attempts_made, "final": str(recovered)}
    return {"status": "failed", "attempts": attempts_made, "final": None, "error": last_output[-500:]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--log", type=Path, default=None, help="run log, used for recovery")
    parser.add_argument("--wrapper", type=Path, default=DEFAULT_WRAPPER)
    parser.add_argument("--retries", type=int, default=6)
    parser.add_argument("--base-delay", type=float, default=15.0)
    parser.add_argument("--max-delay", type=float, default=180.0)
    parser.add_argument("--recover-only", action="store_true")
    args = parser.parse_args()

    if args.recover_only:
        recovered = recover_from_log(args.log, args.out) if args.log else None
        if not recovered:
            raise SystemExit("nothing to recover")
        print(f"recovered -> {recovered}")
        return

    result = fetch_with_retry(
        args.wrapper,
        args.state,
        args.out,
        log_path=args.log,
        retries=args.retries,
        base_delay=args.base_delay,
        max_delay=args.max_delay,
    )
    print(json.dumps(result, indent=2))
    if result["status"] == "failed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
