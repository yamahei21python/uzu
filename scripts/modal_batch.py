#!/usr/bin/env python3
"""Run a Modal command matrix with a hard timeout per command.

``modal run`` has no client-side timeout. When a function fails during startup
(a missing mount, an import error) the client can sit there for hours instead of
returning, which is how a six-command smoke run consumed nine hours and produced
one line of log.

This wrapper makes each step bounded and visible:

    python scripts/modal_batch.py --sizes 4b,9b --commands import,rtn,eval \\
        --run-name 'smoke-{size}' --eval-candidate int4-rtn --max-sequences 20 \\
        --timeout 1800 --log logs/modal_smoke.log

Behaviour:
- one row per (size, command): rc, seconds, last useful line
- stop at the first failure unless --keep-going
- exit 1 if anything failed or timed out, so a CI step cannot pass silently

Usage:
    python scripts/modal_batch.py --sizes 4b --commands import --dry-run
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

DEFAULT_MODAL = ".deps/venv-lalamo/bin/modal"
DEFAULT_SCRIPT = "scripts/modal_pipeline.py"
EVAL_FLAGS = ("--candidate", "int4-rtn")

# The Modal client can exit 0 while the remote function failed during startup,
# and print "App completed" anyway. These markers override a zero exit code.
FAILURE_MARKERS = (
    "Traceback (most recent call last):",
    "Runner failed with exception",
    "failed during startup",
    "ModuleNotFoundError",
)


def build_commands(
    sizes: list[str],
    commands: list[str],
    run_name: str,
    modal_bin: str = DEFAULT_MODAL,
    script: str = DEFAULT_SCRIPT,
    eval_candidate: str = "int4-rtn",
    max_sequences: int = 20,
) -> list[list[str]]:
    """Expand the matrix into argv lists.

    The run name may contain ``{size}`` so each model keeps its own volume
    directory; passing the same name twice would silently reuse the first one.
    """
    argv_list: list[list[str]] = []
    for size in sizes:
        name = run_name.format(size=size)
        for command in commands:
            argv = [modal_bin, "run", script, "--command", command, "--size", size, "--run-name", name]
            if command == "eval":
                argv += [EVAL_FLAGS[0], eval_candidate, "--max-sequences", str(max_sequences)]
            argv_list.append(argv)
    return argv_list


def failure_marker(output: str) -> str | None:
    """Return the first remote-failure marker, if any.

    A zero exit code is not proof of success: Modal reports function startup
    failures in the output and still ends the local app normally.
    """
    for marker in FAILURE_MARKERS:
        if marker in output:
            return marker
    return None


def interesting_line(output: str) -> str:
    """Pick the line a human would want, not the traceback boilerplate."""
    lines = [line.strip() for line in output.splitlines() if line.strip()]
    for line in reversed(lines):
        if line.startswith(("skip ", "/out/", "Traceback", "Function ", "Runner ")):
            return line[:160]
    return (lines[-1][:160] if lines else "")


def format_table(rows: list[dict[str, Any]]) -> str:
    header = "| size | command | rc | seconds | last |"
    lines = [header, "|---|---|---:|---:|---|"]
    for row in rows:
        lines.append(
            f"| {row['size']} | {row['command']} | {row['rc']} | {row['seconds']} | {row['last']} |"
        )
    return "\n".join(lines)


def run_matrix(
    argv_list: list[list[str]],
    timeout: int,
    keep_going: bool,
    log: Path | None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if log:
        log.parent.mkdir(parents=True, exist_ok=True)
        log.write_text("", encoding="utf-8")
    for argv in argv_list:
        size = argv[argv.index("--size") + 1]
        command = argv[argv.index("--command") + 1]
        start = time.time()
        try:
            result = subprocess.run(argv, capture_output=True, text=True, timeout=timeout)
            output = (result.stdout or "") + "\n" + (result.stderr or "")
            rc: Any = result.returncode
            marker = failure_marker(output)
            if rc == 0 and marker:
                rc = f"failed: {marker}"
        except subprocess.TimeoutExpired:
            rc = "timeout"
            output = f"exceeded {timeout}s"
        row = {
            "size": size,
            "command": command,
            "rc": rc,
            "seconds": round(time.time() - start),
            "last": interesting_line(output),
        }
        rows.append(row)
        print(f"{size} {command}: rc={row['rc']} {row['seconds']}s {row['last']}", flush=True)
        if log:
            with log.open("a", encoding="utf-8") as handle:
                handle.write(f"=== {size} {command} rc={row['rc']} {row['seconds']}s\n{output}\n\n")
        if rc != 0 and not keep_going:
            print("stopping at first failure (use --keep-going to continue)", flush=True)
            break
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sizes", default="4b,9b")
    parser.add_argument("--commands", default="import,rtn,eval")
    parser.add_argument("--run-name", default="smoke-{size}")
    parser.add_argument("--eval-candidate", default="int4-rtn")
    parser.add_argument("--max-sequences", type=int, default=20)
    parser.add_argument("--timeout", type=int, default=1800, help="seconds per command")
    parser.add_argument("--modal-bin", default=DEFAULT_MODAL)
    parser.add_argument("--script", default=DEFAULT_SCRIPT)
    parser.add_argument("--log", type=Path, default=None)
    parser.add_argument("--keep-going", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    argv_list = build_commands(
        [item.strip() for item in args.sizes.split(",") if item.strip()],
        [item.strip() for item in args.commands.split(",") if item.strip()],
        args.run_name,
        modal_bin=args.modal_bin,
        script=args.script,
        eval_candidate=args.eval_candidate,
        max_sequences=args.max_sequences,
    )
    if args.dry_run:
        for argv in argv_list:
            print(" ".join(argv))
        return

    rows = run_matrix(argv_list, args.timeout, args.keep_going, args.log)
    print()
    print(format_table(rows))
    failed = [row for row in rows if row["rc"] != 0]
    if failed:
        print(f"\nFAILED {len(failed)}/{len(rows)}")
        raise SystemExit(1)
    print(f"\nALL OK {len(rows)}")


if __name__ == "__main__":
    main()
