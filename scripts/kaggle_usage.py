#!/usr/bin/env python3
"""Kaggle GPU usage ledger.

Kaggle gives each account ~30 GPU-hours per week, 2 concurrent sessions. The
number that matters is not "did it finish" but "how many GPU seconds did it
cost", and nobody was tracking it.

    # record a finished run by hand
    python scripts/kaggle_usage.py log --kernel yamahei21python/uzu-eval-4b-rtn-probe \
        --account owner --seconds 5420 --gpu T4x2 --note "4B RTN eval v3"

    # or track it until it stops
    python scripts/kaggle_usage.py track --kernel yamahei21python/uzu-emb8-08b-gpu --account owner

    # weekly report against the quota
    python scripts/kaggle_usage.py report --week 2026-W45 --quota-hours 30

The ledger is JSON Lines so a crashed run cannot corrupt it.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from datetime import date, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
LEDGER = ROOT / "logs" / "kaggle_usage.jsonl"
TERMINAL = ("COMPLETE", "ERROR", "FAILED", "CANCEL", "CANCELLED")


def week_key(value: date) -> str:
    year, week, _ = value.isocalendar()
    return f"{year}-W{week:02d}"


def is_terminal(status: str) -> bool:
    upper = status.upper()
    return any(marker in upper for marker in TERMINAL)


def append_entry(ledger: Path, entry: dict[str, Any]) -> None:
    ledger.parent.mkdir(parents=True, exist_ok=True)
    with ledger.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(entry, ensure_ascii=False) + "\n")


def read_entries(ledger: Path) -> list[dict[str, Any]]:
    if not ledger.exists():
        return []
    entries = []
    for line in ledger.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            entries.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return entries


def summarize(entries: list[dict[str, Any]], week: str, quota_hours: float) -> dict[str, Any]:
    filtered = [entry for entry in entries if entry.get("week") == week]
    by_account: dict[str, dict[str, Any]] = {}
    for entry in filtered:
        account = entry.get("account", "unknown")
        bucket = by_account.setdefault(account, {"runs": 0, "seconds": 0.0, "kernels": []})
        bucket["runs"] += 1
        bucket["seconds"] += float(entry.get("seconds", 0))
        bucket["kernels"].append(entry.get("kernel"))
    quota_seconds = quota_hours * 3600
    for account, bucket in by_account.items():
        bucket["hours"] = round(bucket["seconds"] / 3600, 2)
        bucket["remaining_hours"] = round((quota_seconds - bucket["seconds"]) / 3600, 2)
        bucket["used_pct"] = round(100 * bucket["seconds"] / quota_seconds, 1) if quota_seconds else None
    total_seconds = sum(bucket["seconds"] for bucket in by_account.values())
    return {
        "week": week,
        "quota_hours_per_account": quota_hours,
        "accounts": by_account,
        "total_hours": round(total_seconds / 3600, 2),
        "runs": len(filtered),
    }


def format_report(summary: dict[str, Any]) -> str:
    lines = [
        f"week {summary['week']}  quota {summary['quota_hours_per_account']}h/account  runs {summary['runs']}",
        f"{'account':10s} {'hours':>8s} {'left':>8s} {'used':>7s} kernels",
        "-" * 72,
    ]
    for account, bucket in sorted(summary["accounts"].items()):
        kernels = ", ".join(str(name) for name in bucket["kernels"][-3:])
        lines.append(
            f"{account:10s} {bucket['hours']:8.2f} {bucket['remaining_hours']:8.2f} "
            f"{(bucket['used_pct'] or 0):6.1f}% {kernels}"
        )
    return "\n".join(lines)


def kernel_status(kernel: str) -> str:
    result = subprocess.run(
        ["kaggle", "kernels", "status", kernel], text=True, capture_output=True, check=False
    )
    text = (result.stdout or "") + (result.stderr or "")
    for line in text.splitlines():
        if "status" in line.lower():
            return line.strip()
    return text.strip()


def track(kernel: str, account: str, interval: float, ledger: Path) -> dict[str, Any]:
    started = time.time()
    status = ""
    while True:
        status = kernel_status(kernel)
        if is_terminal(status):
            break
        print(f"{time.strftime('%H:%M:%S')} {status}", flush=True)
        time.sleep(interval)
    entry = {
        "kernel": kernel,
        "account": account,
        "seconds": round(time.time() - started, 1),
        "status": status,
        "week": week_key(date.today()),
        "finished": datetime.now().isoformat(timespec="seconds"),
        "note": "tracked",
    }
    append_entry(ledger, entry)
    return entry


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ledger", type=Path, default=LEDGER)
    sub = parser.add_subparsers(dest="command", required=True)

    log = sub.add_parser("log")
    log.add_argument("--kernel", required=True)
    log.add_argument("--account", default="owner")
    log.add_argument("--seconds", type=float, required=True)
    log.add_argument("--gpu", default="T4x2")
    log.add_argument("--week", default=None)
    log.add_argument("--note", default="")

    track_cmd = sub.add_parser("track")
    track_cmd.add_argument("--kernel", required=True)
    track_cmd.add_argument("--account", default="owner")
    track_cmd.add_argument("--interval", type=float, default=60.0)

    report = sub.add_parser("report")
    report.add_argument("--week", default=None)
    report.add_argument("--quota-hours", type=float, default=30.0)
    report.add_argument("--json", type=Path, default=None)

    args = parser.parse_args()

    if args.command == "log":
        entry = {
            "kernel": args.kernel,
            "account": args.account,
            "seconds": args.seconds,
            "gpu": args.gpu,
            "week": args.week or week_key(date.today()),
            "logged": datetime.now().isoformat(timespec="seconds"),
            "note": args.note,
        }
        append_entry(args.ledger, entry)
        print(json.dumps(entry, ensure_ascii=False))
    elif args.command == "track":
        entry = track(args.kernel, args.account, args.interval, args.ledger)
        print(json.dumps(entry, ensure_ascii=False))
    else:
        week = args.week or week_key(date.today())
        summary = summarize(read_entries(args.ledger), week, args.quota_hours)
        print(format_report(summary))
        if args.json:
            args.json.parent.mkdir(parents=True, exist_ok=True)
            args.json.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
            print(f"wrote {args.json}")


if __name__ == "__main__":
    main()
