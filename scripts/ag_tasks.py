#!/usr/bin/env python3
"""Track Antigravity (アングラ) delegated tasks and collect their deliverables.

Layout:

    antigravity/tasks/<slug>/prompt.md    # exact prompt that was sent
    antigravity/tasks/<slug>/meta.json    # state path, log, deliverables, status

Commands:
    python scripts/ag_tasks.py list                 # status table
    python scripts/ag_tasks.py list --json
    python scripts/ag_tasks.py show vram-budget
    python scripts/ag_tasks.py collect vram-budget  # copy deliverables into the repo
    python scripts/ag_tasks.py collect --all

``collect`` looks for ``<fetch>/extracted/workspace/<deliverable>`` and copies it
to the same repo-relative path. Files outside research/ and scripts/ go to
``research/antigravity/<slug>/`` so the tree stays predictable.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
TASKS = ROOT / "antigravity" / "tasks"


def set_root(root: Path) -> None:
    global ROOT, TASKS
    ROOT = root
    TASKS = root / "antigravity" / "tasks"
ALLOWED_PREFIXES = ("research/", "scripts/", "lanes/", "tests/")


def task_dirs() -> list[Path]:
    if not TASKS.exists():
        return []
    return sorted(path for path in TASKS.iterdir() if (path / "meta.json").exists())


def load_meta(directory: Path) -> dict[str, Any]:
    meta = json.loads((directory / "meta.json").read_text(encoding="utf-8"))
    meta.setdefault("slug", directory.name)
    meta.setdefault("deliverables", [])
    # runtime fields (status, pid, attempts) live outside the tracked file
    runtime = ROOT / "antigravity" / ".state" / f"{directory.name}.json"
    if runtime.exists():
        try:
            meta.update(json.loads(runtime.read_text(encoding="utf-8")))
        except json.JSONDecodeError:
            pass
    return meta


def state_status(meta: dict[str, Any]) -> str:
    if meta.get("status") == "collected":
        return "collected"
    state_path = Path(meta.get("state", "")).expanduser()
    if not state_path.exists():
        return meta.get("status", "unknown")
    try:
        state = json.loads(state_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return "unreadable-state"
    for key in ("status", "state"):
        if isinstance(state.get(key), str):
            return state[key]
    if state.get("interaction_id"):
        return "started"
    return meta.get("status", "unknown")


def deliverable_status(meta: dict[str, Any]) -> tuple[int, int]:
    present = sum(1 for item in meta["deliverables"] if (ROOT / item).exists())
    return present, len(meta["deliverables"])


def cmd_list(as_json: bool) -> None:
    rows = []
    for directory in task_dirs():
        meta = load_meta(directory)
        present, total = deliverable_status(meta)
        rows.append(
            {
                "slug": meta["slug"],
                "label": meta.get("label", meta["slug"]),
                "status": state_status(meta),
                "deliverables": f"{present}/{total}",
                "started": meta.get("started", ""),
                "log": meta.get("log", ""),
                "note": meta.get("note", ""),
            }
        )
    if as_json:
        print(json.dumps(rows, indent=2, ensure_ascii=False))
        return
    if not rows:
        print("no tasks registered under antigravity/tasks/")
        return
    print(f"{'slug':24s} {'status':12s} {'files':6s} {'started':20s} note")
    print("-" * 88)
    for row in rows:
        print(
            f"{row['slug']:24s} {row['status']:12s} {row['deliverables']:6s} "
            f"{row['started']:20s} {row['note']}"
        )


def cmd_recover(slug: str, out: Path | None) -> None:
    """Write the agent's final message from the run log when the snapshot is gone."""
    directory = TASKS / slug
    if not (directory / "meta.json").exists():
        raise SystemExit(f"unknown task: {slug}")
    meta = load_meta(directory)
    log = ROOT / meta.get("log", "")
    target_dir = out or Path(meta.get("fetch_dir", f"/tmp/ag_{slug.replace('-', '_')}_fetch")).expanduser()
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from ag_fetch import recover_from_log

    recovered = recover_from_log(log, target_dir)
    if not recovered:
        raise SystemExit(f"no completed final message in {log}")
    print(f"recovered -> {recovered}")


def cmd_show(slug: str) -> None:
    directory = TASKS / slug
    if not (directory / "meta.json").exists():
        raise SystemExit(f"unknown task: {slug}")
    meta = load_meta(directory)
    meta["current_status"] = state_status(meta)
    meta["prompt"] = (directory / "prompt.md").read_text(encoding="utf-8")
    print(json.dumps(meta, indent=2, ensure_ascii=False))


def cmd_collect(slug: str, dry_run: bool) -> None:
    directory = TASKS / slug
    if not (directory / "meta.json").exists():
        raise SystemExit(f"unknown task: {slug}")
    meta = load_meta(directory)
    fetch = Path(meta.get("fetch_dir", f"/tmp/ag_{slug.replace('-', '_')}_fetch")).expanduser()
    source_root = fetch / "extracted" / "workspace"
    if not source_root.exists():
        raise SystemExit(f"nothing fetched yet: {source_root}")

    copied = []
    for item in meta["deliverables"]:
        source = source_root / item
        if not source.exists():
            print(f"  missing in fetch: {item}")
            continue
        if item.startswith(ALLOWED_PREFIXES):
            target = ROOT / item
        else:
            target = ROOT / "research" / "antigravity" / slug / Path(item).name
        print(f"  {source} -> {target}")
        if not dry_run:
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
            copied.append(item)

    if dry_run:
        return
    meta["status"] = "collected" if copied else "fetched"
    meta["collected_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    (directory / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"collected {len(copied)} file(s) for {slug}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", default=None, help="repo root override (for tests)")
    sub = parser.add_subparsers(dest="command", required=True)

    listing = sub.add_parser("list")
    listing.add_argument("--json", action="store_true")

    show = sub.add_parser("show")
    show.add_argument("slug")

    recover = sub.add_parser("recover", help="recover the final message from the run log")
    recover.add_argument("slug")
    recover.add_argument("--out", default=None)

    collect = sub.add_parser("collect")
    collect.add_argument("slug", nargs="?")
    collect.add_argument("--all", action="store_true")
    collect.add_argument("--dry-run", action="store_true")

    args = parser.parse_args()
    if args.root:
        set_root(Path(args.root).expanduser())
    if args.command == "list":
        cmd_list(args.json)
    elif args.command == "show":
        cmd_show(args.slug)
    elif args.command == "recover":
        cmd_recover(args.slug, Path(args.out).expanduser() if args.out else None)
    elif args.command == "collect":
        if args.all:
            for directory in task_dirs():
                cmd_collect(directory.name, args.dry_run)
        elif args.slug:
            cmd_collect(args.slug, args.dry_run)
        else:
            parser.error("collect needs a slug or --all")


if __name__ == "__main__":
    main()
