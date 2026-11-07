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
import urllib.error
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
TASKS = ROOT / "antigravity" / "tasks"
LOCK = ROOT / "antigravity" / ".queue.lock"
RUNTIME_DIR = ROOT / "antigravity" / ".state"
RUNTIME_KEYS = (
    "queue_status",
    "status",
    "pid",
    "attempts",
    "started_at",
    "finished_at",
    "pid_gone_at",
    "collected_at",
)
DEFAULT_WRAPPER = Path.home() / ".config" / "opencode" / "antigravity" / "antigravity.py"
AG_HOME = DEFAULT_WRAPPER.parent
CREDENTIALS = AG_HOME / "credentials.json"
API_BASE = "https://generativelanguage.googleapis.com/v1beta"
CAPACITY = 3
TERMINAL = {"done", "failed", "cancelled"}


def set_root(root: Path) -> None:
    global ROOT, TASKS, LOCK, RUNTIME_DIR
    ROOT = root
    TASKS = root / "antigravity" / "tasks"
    LOCK = root / "antigravity" / ".queue.lock"
    RUNTIME_DIR = root / "antigravity" / ".state"


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S")


def runtime_path(slug: str) -> Path:
    return RUNTIME_DIR / f"{slug}.json"


def load_runtime(slug: str) -> dict[str, Any]:
    path = runtime_path(slug)
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def save_runtime(slug: str, values: dict[str, Any]) -> None:
    """Runtime state lives outside meta.json so a git checkout cannot wipe it."""
    RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
    current = load_runtime(slug)
    current.update({key: values.get(key) for key in RUNTIME_KEYS if key in values})
    runtime_path(slug).write_text(json.dumps(current, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def tickets() -> list[dict[str, Any]]:
    if not TASKS.exists():
        return []
    found = []
    for directory in sorted(TASKS.iterdir()):
        meta_path = directory / "meta.json"
        if not meta_path.exists():
            continue
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        slug = meta.setdefault("slug", directory.name)
        # one-time migration: pull stale runtime fields out of the tracked file
        stale = {key: meta.pop(key) for key in RUNTIME_KEYS if key in meta}
        if stale:
            merged = load_runtime(slug)
            merged.update({key: value for key, value in stale.items() if key not in merged})
            save_runtime(slug, merged)
            meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        meta.update(load_runtime(slug))
        meta.setdefault("queue_status", "queued")
        meta.setdefault("priority", 50)
        meta.setdefault("attempts", 0)
        meta.setdefault("max_attempts", 3)
        meta.setdefault("deliverables", [])
        found.append(meta)
    return found


def save(meta: dict[str, Any]) -> None:
    """Persist runtime fields only; meta.json stays stable for git."""
    save_runtime(meta["slug"], meta)


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


# ---------- environment cleanup ----------

ENV_DELETED_LOG = ROOT / "logs" / time.strftime("%Y-%m-%d") / "antigravity_deleted_envs.txt"


def api_key() -> str | None:
    for name in ("ANTIGRAVITY_API_KEY", "GEMINI_API_KEY", "GOOGLE_API_KEY"):
        value = os.environ.get(name)
        if value:
            return value
    try:
        return json.loads(CREDENTIALS.read_text(encoding="utf-8"))["api_key"]
    except (OSError, KeyError, json.JSONDecodeError):
        return None


TERMINAL_STATES = {"done", "failed", "cancelled", "collected"}
PRESSURE_MARKERS = (
    "rate limit",
    "too_many_requests",
    "storage quota",
    "quota exceeded",
    "environment limit",
    "resource_exhausted",
)


def is_pressure(text: str) -> bool:
    """Environment saturation shows up under several names; all mean 'free old sandboxes'."""
    lowered = text.lower()
    return any(marker in lowered for marker in PRESSURE_MARKERS)


def active_env_ids(protect_completed: bool = False, protect_sessions: bool = True) -> set[str]:
    """Environments that must survive a cleanup.

    With ``protect_completed=False`` (the default) a finished task no longer pins
    its environment: finished sandboxes are the main storage hogs and blocking
    them is what caused "Project environment storage quota exceeded".
    ``pending/`` and ``sessions/`` are always protected because a ``continue``
    needs them.
    """
    ids: set[str] = set()
    for sub in (("sessions", "pending") if protect_sessions else ()):
        directory = AG_HOME / sub
        if not directory.exists():
            continue
        for path in directory.glob("*.json"):
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            env_id = payload.get("environment_id")
            if isinstance(env_id, str) and env_id:
                ids.add(env_id)

    live: list[Path] = []
    for meta_path in sorted((ROOT / "antigravity" / "tasks").glob("*/meta.json")):
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        state = meta.get("queue_status") or meta.get("status") or ""
        state_path = Path(meta.get("state", f"/tmp/ag_{meta_path.parent.name}.json")).expanduser()
        if not protect_completed:
            if state in TERMINAL_STATES:
                continue
            # a manual task that already produced its state file is finished
            if state == "manual" and state_path.exists():
                continue
        if state_path.exists():
            live.append(state_path)
    for path in live:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        env_id = payload.get("environment_id")
        if isinstance(env_id, str) and env_id:
            ids.add(env_id)
    return ids


def parse_created(value: Any) -> datetime | None:
    if not isinstance(value, str):
        return None
    text = value.replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is not None:
        parsed = parsed.astimezone(timezone.utc).replace(tzinfo=None)
    return parsed


def select_deletable(
    environments: list[dict[str, Any]],
    keep: set[str],
    older_than_days: float,
    now: datetime | None = None,
) -> list[dict[str, Any]]:
    """Pure selector so it can be tested without touching the API."""
    cutoff = (now or datetime.now(timezone.utc).replace(tzinfo=None)) - timedelta(days=older_than_days)
    selected = []
    for environment in environments:
        env_id = environment.get("id")
        if not isinstance(env_id, str) or env_id in keep:
            continue
        created = parse_created(environment.get("created"))
        if created is None:
            continue  # unknown age: keep it
        if created <= cutoff:
            selected.append(environment)
    return selected


def fetch_environments(api_base: str, key: str) -> list[dict[str, Any]]:
    request = urllib.request.Request(
        f"{api_base}/environments?pageSize=100", headers={"x-goog-api-key": key}
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        return json.loads(response.read().decode("utf-8")).get("environments", [])


def delete_environment(api_base: str, key: str, env_id: str) -> bool:
    request = urllib.request.Request(
        f"{api_base}/environments/{env_id}", headers={"x-goog-api-key": key}, method="DELETE"
    )
    try:
        with urllib.request.urlopen(request, timeout=30):
            return True
    except urllib.error.HTTPError as error:
        print(f"delete {env_id} failed: {error.code}")
        return False
    except urllib.error.URLError as error:
        print(f"delete {env_id} failed: {error.reason}")
        return False


def select_by_size(
    environments: list[dict[str, Any]],
    keep: set[str],
    target_gib: float,
    min_age_minutes: float,
    now: datetime | None = None,
) -> list[dict[str, Any]]:
    """Pick environments to delete until the total fits the storage target."""
    current = datetime.now(timezone.utc).replace(tzinfo=None) if now is None else now
    total = sum(int(env.get("size_bytes", 0) or 0) for env in environments)
    target = target_gib * 1024**3
    candidates = []
    for environment in environments:
        env_id = environment.get("id")
        if not isinstance(env_id, str) or env_id in keep:
            continue
        created = parse_created(environment.get("created"))
        if created is None:
            continue
        age_minutes = (current - created).total_seconds() / 60
        if age_minutes < min_age_minutes:
            continue
        candidates.append(environment)
    candidates.sort(key=lambda env: (-int(env.get("size_bytes", 0) or 0), env.get("created", "")))
    selected = []
    for environment in candidates:
        if total <= target:
            break
        total -= int(environment.get("size_bytes", 0) or 0)
        selected.append(environment)
    return selected


def cleanup_envs(
    older_than_days: float,
    dry_run: bool = False,
    api_base: str = API_BASE,
    env_list_file: Path | None = None,
    keep_extra: set[str] | None = None,
    mode: str = "age",
    target_gib: float = 0.8,
    min_age_minutes: float = 10.0,
    protect_completed: bool = False,
    protect_sessions: bool = True,
    force: bool = False,
) -> dict[str, Any]:
    """Delete idle Antigravity environments.

    ``mode="age"`` deletes everything older than ``older_than_days``.
    ``mode="auto"`` deletes the largest finished environments until the total
    fits ``target_gib``; this is what a storage-quota or 429 error needs.
    """
    keep = set() if force else active_env_ids(
        protect_completed=protect_completed, protect_sessions=protect_sessions
    )
    keep |= (keep_extra or set())
    if env_list_file is not None:
        environments = json.loads(env_list_file.read_text(encoding="utf-8"))
        if isinstance(environments, dict):
            environments = environments.get("environments", [])
    else:
        key = api_key()
        if not key:
            print("no API key: set ANTIGRAVITY_API_KEY or credentials.json")
            return {"error": "no-api-key"}
        try:
            environments = fetch_environments(api_base, key)
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError) as error:
            print(f"cannot list environments: {error}")
            return {"error": str(error)}

    if mode == "auto":
        selected = select_by_size(environments, keep, target_gib, min_age_minutes)
    else:
        selected = select_deletable(environments, keep, older_than_days)
    before_gib = sum(int(env.get("size_bytes", 0) or 0) for env in environments) / 1024**3
    deleted: list[str] = []
    for environment in selected:
        env_id = environment["id"]
        created = environment.get("created", "?")
        size = environment.get("size_bytes", "?")
        print(f"{'[dry-run] ' if dry_run else ''}delete {env_id} created={created} size={size}")
        if dry_run:
            continue
        key = api_key()
        if not key:
            return {"error": "no-api-key", "deleted": deleted}
        if delete_environment(api_base, key, env_id):
            deleted.append(env_id)

    if deleted and not dry_run:
        ENV_DELETED_LOG.parent.mkdir(parents=True, exist_ok=True)
        with ENV_DELETED_LOG.open("a", encoding="utf-8") as handle:
            for env_id in deleted:
                handle.write(f"{now()} {env_id}\n")

    summary = {
        "mode": mode,
        "total": len(environments),
        "total_gib_before": round(before_gib, 3),
        "total_gib_after": round(
            sum(
                int(env.get("size_bytes", 0) or 0)
                for env in environments
                if env.get("id") not in deleted
            )
            / 1024**3,
            3,
        ),
        "kept_active": len([env for env in environments if env.get("id") in keep]),
        "selected": len(selected),
        "deleted": len(deleted),
        "dry_run": dry_run,
        "older_than_days": older_than_days,
        "target_gib": target_gib,
    }
    print(json.dumps(summary, ensure_ascii=False))
    return summary


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
    static = {key: value for key, value in meta.items() if key not in RUNTIME_KEYS}
    (directory / "meta.json").write_text(json.dumps(static, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
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


def finish(meta: dict[str, Any], wrapper: Path, fetch_retries: int = 6, fetch_delay: float = 15.0) -> None:
    fetch_dir = Path(meta.get("fetch_dir", f"/tmp/ag_{meta['slug'].replace('-', '_')}_fetch")).expanduser()
    print(f"fetch {meta['slug']}", flush=True)
    sys.path.insert(0, str(SCRIPT_DIR))
    from ag_fetch import fetch_with_retry

    result = fetch_with_retry(
        wrapper,
        state_path(meta),
        fetch_dir,
        log_path=ROOT / meta["log"],
        retries=fetch_retries,
        base_delay=fetch_delay,
    )
    print(f"fetch status: {result['status']} (attempts {result['attempts']})", flush=True)
    collect = subprocess.run(
        [sys.executable, str(SCRIPT_DIR / "ag_tasks.py"), "--root", str(ROOT), "collect", meta["slug"]],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    print(collect.stdout.strip() or collect.stderr.strip())
    meta["queue_status"] = "done"
    if result["status"] == "recovered":
        meta["status"] = "recovered"
        meta["note"] = (meta.get("note", "") + " [fetched from log: snapshot 503]").strip()
    else:
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
                finish(meta, wrapper, args.fetch_retries, args.fetch_base_delay)
                continue
            if alive(meta.get("pid")):
                active += 1
                continue
            log_path = ROOT / meta["log"]
            sys.path.insert(0, str(SCRIPT_DIR))
            from ag_fetch import log_completed

            if log_completed(log_path):
                print(f"{meta['slug']} finished in the log; fetching", flush=True)
                finish(meta, wrapper, args.fetch_retries, args.fetch_base_delay)
                continue

            pid_gone_at = meta.get("pid_gone_at")
            if pid_gone_at:
                gone_for = time.time() - float(pid_gone_at)
                if gone_for < args.pid_grace_seconds:
                    active += 1
                    continue
            else:
                meta["pid_gone_at"] = time.time()
                save(meta)
                active += 1
                continue

            log_tail = tail(log_path)
            meta["pid"] = None
            if is_pressure(log_tail) and int(meta.get("attempts", 0)) < int(meta.get("max_attempts", 3)):
                meta["queue_status"] = "queued"
                print(f"requeue {meta['slug']} (environment pressure)")
                # quota/429 mean the sandbox store is full, not that the task is wrong.
                # Free the largest finished environments before retrying.
                cleanup_envs(
                    older_than_days=0,
                    mode="auto",
                    target_gib=getattr(args, "target_gib", 0.8),
                    min_age_minutes=getattr(args, "min_env_age_minutes", 10.0),
                )
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


def cmd_cleanup_envs(args: argparse.Namespace) -> None:
    summary = cleanup_envs(
        older_than_days=args.older_than_days,
        dry_run=args.dry_run,
        api_base=args.api_base,
        env_list_file=Path(args.env_list_file).expanduser() if args.env_list_file else None,
        keep_extra=set(args.keep or []),
        mode=args.mode,
        target_gib=args.target_gib,
        min_age_minutes=args.min_age_minutes,
        protect_completed=args.protect_completed,
        protect_sessions=not args.ignore_sessions,
        force=args.force,
    )
    if summary.get("error"):
        raise SystemExit(1)


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
    pump.add_argument("--env-ttl-days", type=float, default=1.0, help="delete idle environments older than this on 429")
    pump.add_argument("--target-gib", type=float, default=0.8, help="storage target for quota-driven cleanup")
    pump.add_argument("--min-env-age-minutes", type=float, default=10.0)
    pump.add_argument("--pid-grace-seconds", type=float, default=300.0, help="wait this long before treating a dead pid as failure")
    pump.add_argument("--fetch-retries", type=int, default=6)
    pump.add_argument("--fetch-base-delay", type=float, default=15.0)

    watch = sub.add_parser("watch")
    watch.add_argument("--max-active", type=int, default=2)
    watch.add_argument("--interval", type=int, default=300)
    watch.add_argument("--env-ttl-days", type=float, default=1.0)
    watch.add_argument("--target-gib", type=float, default=0.8)
    watch.add_argument("--min-env-age-minutes", type=float, default=10.0)
    watch.add_argument("--pid-grace-seconds", type=float, default=300.0)
    watch.add_argument("--fetch-retries", type=int, default=6)
    watch.add_argument("--fetch-base-delay", type=float, default=15.0)

    cleanup = sub.add_parser("cleanup-envs", help="delete idle Antigravity environments")
    cleanup.add_argument("--older-than-days", type=float, default=1.0)
    cleanup.add_argument("--dry-run", action="store_true")
    cleanup.add_argument("--api-base", default=API_BASE)
    cleanup.add_argument("--env-list-file", default=None, help="offline JSON list (tests)")
    cleanup.add_argument("--keep", nargs="*", default=[], help="extra environment ids to keep")
    cleanup.add_argument("--mode", choices=("age", "auto"), default="age")
    cleanup.add_argument("--target-gib", type=float, default=0.8)
    cleanup.add_argument("--min-age-minutes", type=float, default=10.0)
    cleanup.add_argument("--protect-completed", action="store_true", help="also keep environments of finished tasks")
    cleanup.add_argument("--ignore-sessions", action="store_true", help="do not protect environments referenced by sessions/ (needed when storage is full)")
    cleanup.add_argument("--force", action="store_true", help="ignore all protections except --keep (last resort)")

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
    elif args.command == "cleanup-envs":
        cmd_cleanup_envs(args)
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
