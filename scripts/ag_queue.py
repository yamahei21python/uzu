#!/usr/bin/env python3
"""Ticket queue for Antigravity (アングラ) tasks.

Keeps a fixed number of tasks running in the background and leaves one slot
free for ad-hoc work. Tickets live next to the existing task records, so
``ag_tasks.py list`` keeps working.

    antigravity/tasks/<slug>/prompt.md
    antigravity/tasks/<slug>/meta.json   # queue_status, priority, attempts, pid, ...

Commands:
    python scripts/ag_queue.py add --slug vram-budget --prompt p.txt \
        --deliverables research/x.md scripts/y.py --priority 10
    python scripts/ag_queue.py list
    python scripts/ag_queue.py pump --max-active 2
    python scripts/ag_queue.py watch --max-active 2 --interval 300
    python scripts/ag_queue.py retry <slug>
    python scripts/ag_queue.py cancel <slug>

``pump`` does two things: reconcile finished tickets (fetch + collect) and
spawn queued ones until ``--max-active`` is reached. With the default capacity
of 3, ``--max-active 2`` leaves one slot for whatever you want to run by hand.

The wrapper writes its state file only when a run returns, so a ticket is
"running" while its spawned pid is alive and "done" once the state file exists.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
TASKS = ROOT / "antigravity" / "tasks"
LOCK = ROOT / "antigravity" / ".queue.lock"
DEFAULT_WRAPPER = Path.home() / ".config" / "opencode" / "antigravity" / "antigravity.py"
CAPACITY = 3
TERMINAL = {"done", "failed", "cancelled"}


def set_root(root: Path) -> None:
    global ROOT, TASKS, LOCK
    ROOT = root
    TASKS = root / "antigravity" / "tasks"
    LOCK = root / "antigravity" / ".queue.lock"


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S")


def tickets() -> list[dict[str, Any]]:
    if not TASKS.exists():
        return []
    found = []
    for directory in sorted(TASKS.iterdir()):
        meta_path = directory / "meta.json"
        if not meta_path.exists():
            continue
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        meta.setdefault("slug", directory.name)
        meta.setdefault("queue_status", meta.get("status", "queued"))
        meta.setdefault("priority", 50)
        meta.setdefault("attempts", 0)
        meta.setdefault("max_attempts", 3)
        meta.setdefault("deliverables", [])
        found.append(meta)
    return found


def save(meta: dict[str, Any]) -> None:
    directory = TASKS / meta["slug"]
    (directory / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def alive(pid: int | None) -> bool:
    if not pid:
        return False
    try:
        os.kill(int(pid), 0)
    except (OSError, ValueError):
        return False
    return True


def state_path(meta: dict[str, Any]) -> Path:
    return Path(meta.get("state", f"/tmp/ag_{meta['slug'].replace('-', '_')}.json")).expanduser()


def state_done(meta: dict[str, Any]) -> bool:
    path = state_path(meta)
    if not path.exists():
        return False
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return False
    return bool(payload.get("interaction_id"))


def tail(path: Path, limit: int = 4000) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="replace")[-limit:]
    except OSError:
        return ""


# ---------- commands ----------


def cmd_add(args: argparse.Namespace) -> None:
    directory = TASKS / args.slug
    directory.mkdir(parents=True, exist_ok=True)
    prompt = Path(args.prompt).read_text(encoding="utf-8") if args.prompt else sys.stdin.read()
    (directory / "prompt.md").write_text(prompt, encoding="utf-8")
    meta = {
        "slug": args.slug,
        "label": args.slug,
        "state": f"/tmp/ag_{args.slug.replace('-', '_')}.json",
        "log": f"logs/{time.strftime('%Y-%m-%d')}/local/antigravity_{args.slug}.log",
        "fetch_dir": f"/tmp/ag_{args.slug.replace('-', '_')}_fetch",
        "deliverables": args.deliverables or [],
        "status": "queued",
        "queue_status": "queued",
        "priority": args.priority,
        "attempts": 0,
        "max_attempts": args.max_attempts,
        "note": args.note or "",
        "created_at": now(),
    }
    save(meta)
    print(f"queued {args.slug} (priority {args.priority})")


def cmd_list(args: argparse.Namespace) -> None:
    rows = []
    for meta in tickets():
        rows.append(
            {
                "slug": meta["slug"],
                "status": meta["queue_status"],
                "prio": meta.get("priority", 50),
                "attempts": f"{meta.get('attempts', 0)}/{meta.get('max_attempts', 3)}",
                "pid": meta.get("pid") or "",
                "alive": "yes" if alive(meta.get("pid")) else "",
                "note": meta.get("note", ""),
            }
        )
    if args.json:
        print(json.dumps(rows, indent=2, ensure_ascii=False))
        return
    if not rows:
        print("no tickets")
        return
    print(f"{'slug':24s} {'status':10s} {'prio':4s} {'attempts':8s} {'pid':8s} {'alive':5s} note")
    print("-" * 92)
    for row in sorted(rows, key=lambda item: -int(item["prio"])):
        print(
            f"{row['slug']:24s} {row['status']:10s} {row['prio']:<4d} {row['attempts']:8s} "
            f"{str(row['pid']):8s} {row['alive']:5s} {row['note']}"
        )


def cmd_retry(args: argparse.Namespace) -> None:
    meta = next((item for item in tickets() if item["slug"] == args.slug), None)
    if meta is None:
        raise SystemExit(f"unknown ticket: {args.slug}")
    meta["queue_status"] = "queued"
    meta["attempts"] = 0
    meta["pid"] = None
    save(meta)
    print(f"requeued {args.slug}")


def cmd_cancel(args: argparse.Namespace) -> None:
    meta = next((item for item in tickets() if item["slug"] == args.slug), None)
    if meta is None:
        raise SystemExit(f"unknown ticket: {args.slug}")
    pid = meta.get("pid")
    if alive(pid):
        try:
            os.killpg(int(pid), signal.SIGTERM)
        except OSError:
            pass
    meta["queue_status"] = "cancelled"
    save(meta)
    print(f"cancelled {args.slug}")


def spawn(meta: dict[str, Any], wrapper: Path) -> None:
    prompt = (TASKS / meta["slug"] / "prompt.md").read_text(encoding="utf-8")
    log = ROOT / meta["log"]
    log.parent.mkdir(parents=True, exist_ok=True)
    handle = log.open("a", encoding="utf-8")
    handle.write(f"\n=== queue dispatch {now()} ===\n")
    handle.flush()
    command = [
        sys.executable,
        str(wrapper),
        "--state",
        str(state_path(meta)),
        "--timeout",
        str(meta.get("timeout", 1800)),
        "run",
        "--prompt",
        prompt,
        "--label",
        meta["slug"],
    ]
    process = subprocess.Popen(
        command,
        stdout=handle,
        stderr=subprocess.STDOUT,
        cwd=ROOT,
        start_new_session=True,
    )
    meta["queue_status"] = "running"
    meta["status"] = "running"
    meta["pid"] = process.pid
    meta["attempts"] = int(meta.get("attempts", 0)) + 1
    meta["started_at"] = now()
    save(meta)
    print(f"started {meta['slug']} pid={process.pid} (attempt {meta['attempts']})")


def finish(meta: dict[str, Any], wrapper: Path) -> None:
    fetch_dir = Path(meta.get("fetch_dir", f"/tmp/ag_{meta['slug'].replace('-', '_')}_fetch")).expanduser()
    command = [
        sys.executable,
        str(wrapper),
        "--state",
        str(state_path(meta)),
        "fetch",
        "--out",
        str(fetch_dir),
    ]
    print(f"fetch {meta['slug']}")
    subprocess.run(command, cwd=ROOT, check=False)
    collect = subprocess.run(
        [sys.executable, str(SCRIPT_DIR / "ag_tasks.py"), "--root", str(ROOT), "collect", meta["slug"]],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    print(collect.stdout.strip() or collect.stderr.strip())
    meta["queue_status"] = "done"
    meta["status"] = "collected" if collect.returncode == 0 else "fetched"
    meta["finished_at"] = now()
    save(meta)


def cmd_pump(args: argparse.Namespace) -> None:
    wrapper = Path(args.wrapper).expanduser()
    active = 0
    for meta in tickets():
        status = meta["queue_status"]
        if status == "manual":
            continue
        if status == "running":
            if state_done(meta):
                finish(meta, wrapper)
                continue
            if alive(meta.get("pid")):
                active += 1
                continue
            log_tail = tail(ROOT / meta["log"])
            meta["pid"] = None
            if "Rate limit" in log_tail and int(meta.get("attempts", 0)) < int(meta.get("max_attempts", 3)):
                meta["queue_status"] = "queued"
                print(f"requeue {meta['slug']} (rate limit)")
            else:
                meta["queue_status"] = "failed"
                print(f"failed {meta['slug']} (pid gone, no state)")
            save(meta)

    # manually started tasks (outside the queue) still occupy a slot until done
    manual_active = sum(
        1 for meta in tickets() if meta["queue_status"] == "manual" and not state_done(meta)
    )
    active += manual_active

    queued = [meta for meta in tickets() if meta["queue_status"] == "queued"]
    queued.sort(key=lambda item: (-int(item.get("priority", 50)), item.get("created_at", "")))
    for meta in queued:
        if active >= args.max_active:
            break
        spawn(meta, wrapper)
        active += 1

    running = sum(1 for meta in tickets() if meta["queue_status"] == "running") + manual_active
    print(f"active={running} capacity={CAPACITY} max_active={args.max_active} free_slots={max(0, CAPACITY - running)}")


def cmd_watch(args: argparse.Namespace) -> None:
    print(f"watching every {args.interval}s, max_active={args.max_active}", flush=True)
    while True:
        cmd_pump(args)
        time.sleep(args.interval)


def acquire_lock() -> Any:
    LOCK.parent.mkdir(parents=True, exist_ok=True)
    handle = LOCK.open("w")
    try:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        print("another queue process holds the lock; exiting")
        raise SystemExit(1)
    return handle


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", default=None, help="repo root override (for tests)")
    parser.add_argument("--wrapper", default=str(DEFAULT_WRAPPER), help="antigravity.py path")
    sub = parser.add_subparsers(dest="command", required=True)

    add = sub.add_parser("add")
    add.add_argument("--slug", required=True)
    add.add_argument("--prompt", default=None, help="prompt file; stdin when omitted")
    add.add_argument("--deliverables", nargs="*", default=[])
    add.add_argument("--priority", type=int, default=50)
    add.add_argument("--max-attempts", type=int, default=3)
    add.add_argument("--note", default="")

    listing = sub.add_parser("list")
    listing.add_argument("--json", action="store_true")

    pump = sub.add_parser("pump")
    pump.add_argument("--max-active", type=int, default=2)

    watch = sub.add_parser("watch")
    watch.add_argument("--max-active", type=int, default=2)
    watch.add_argument("--interval", type=int, default=300)

    for name in ("retry", "cancel"):
        command = sub.add_parser(name)
        command.add_argument("slug")

    args = parser.parse_args()
    if args.root:
        set_root(Path(args.root).expanduser())

    if args.command == "add":
        cmd_add(args)
    elif args.command == "list":
        cmd_list(args)
    elif args.command == "retry":
        cmd_retry(args)
    elif args.command == "cancel":
        cmd_cancel(args)
    elif args.command in ("pump", "watch"):
        lock = acquire_lock()
        try:
            if args.command == "pump":
                cmd_pump(args)
            else:
                cmd_watch(args)
        finally:
            lock.close()


if __name__ == "__main__":
    main()
