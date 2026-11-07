#!/usr/bin/env python3
"""One board for every task: what runs, what waits, what is blocked.

Sources:
- owner Kaggle kernels: live status via the kaggle CLI
- Muse kernels: recorded in tasks/kernels.json (their account is not queryable)
- Antigravity tickets: antigravity/.state + tasks/<slug>/meta.json
- local: anything registered under "local" in tasks/kernels.json

    python scripts/task_board.py
    python scripts/task_board.py --write docs/TASKS.md
    python scripts/task_board.py --json /tmp/tasks.json
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TASKS = ROOT / "tasks" / "kernels.json"
AG_TASKS = ROOT / "antigravity" / "tasks"
AG_STATE = ROOT / "antigravity" / ".state"

TERMINAL = ("COMPLETE", "ERROR", "FAILED", "CANCEL", "CANCELLED")


def status_of(kernel: str) -> str:
    try:
        result = subprocess.run(
            ["kaggle", "kernels", "status", kernel], text=True, capture_output=True, check=False
        )
    except OSError:
        return "unknown (no kaggle cli)"
    text = (result.stdout or "") + (result.stderr or "")
    line = next((item.strip() for item in text.splitlines() if "status" in item.lower()), "")
    if not line:
        return "unknown"
    upper = line.upper()
    if "COMPLETE" in upper:
        return "done"
    if any(marker in upper for marker in ("ERROR", "FAILED", "CANCEL")):
        return "failed"
    if "RUNNING" in upper or "QUEUED" in upper:
        return "running"
    return line[:40]


def antigravity_rows() -> list[dict]:
    rows = []
    if not AG_TASKS.exists():
        return rows
    for directory in sorted(AG_TASKS.iterdir()):
        meta_path = directory / "meta.json"
        if not meta_path.exists():
            continue
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        slug = meta.get("slug", directory.name)
        state = {}
        state_path = AG_STATE / f"{slug}.json"
        if state_path.exists():
            try:
                state = json.loads(state_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                state = {}
        rows.append(
            {
                "source": "antigravity",
                "name": slug,
                "lane": meta.get("note", ""),
                "status": state.get("queue_status", meta.get("queue_status", "queued")),
                "priority": meta.get("priority", 50),
            }
        )
    rows.sort(key=lambda row: -int(row["priority"]))
    return rows


def kernel_rows() -> list[dict]:
    if not TASKS.exists():
        return []
    payload = json.loads(TASKS.read_text(encoding="utf-8"))
    rows = []
    for entry in payload.get("kernels", []):
        owner = entry.get("owner", "owner")
        if owner == "owner":
            status = status_of(entry["ref"])
        elif owner == "muse":
            status = entry.get("status", "reported")
        else:
            status = entry.get("status", "planned")
        rows.append(
            {
                "source": owner,
                "name": entry["ref"].split("/")[-1],
                "lane": entry.get("lane", ""),
                "kind": entry.get("kind", ""),
                "status": status,
                "note": entry.get("note", ""),
            }
        )
    return rows


def render(rows: list[dict]) -> str:
    buckets = {"running": [], "queued": [], "waiting": [], "done": [], "failed": [], "planned": []}
    for row in rows:
        status = str(row["status"]).lower()
        if status in ("done", "collected"):
            buckets["done"].append(row)
        elif status in ("failed", "cancelled"):
            buckets["failed"].append(row)
        elif status in ("running", "reported"):
            buckets["running"].append(row)
        elif status in ("queued", "pending"):
            buckets["queued"].append(row)
        elif status == "manual":
            buckets["waiting"].append(row)
        else:
            buckets["planned"].append(row)

    lines = ["# タスクボード", "", f"全 {len(rows)} 件", ""]
    titles = {
        "running": "実行中",
        "queued": "待ち（キュー投入済み）",
        "waiting": "待ち（外部・手動）",
        "planned": "未投入",
        "failed": "失敗（要対応）",
        "done": "完了",
    }
    for key in ("running", "queued", "waiting", "planned", "failed", "done"):
        items = buckets[key]
        if not items:
            continue
        lines.append(f"## {titles[key]}（{len(items)}）")
        lines.append("| source | name | lane | status |")
        lines.append("|---|---|---|---|")
        for row in items:
            lines.append(f"| {row['source']} | {row['name']} | {row.get('lane','')} | {row['status']} |")
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--write", type=Path, default=None)
    parser.add_argument("--json", type=Path, default=None)
    args = parser.parse_args()

    rows = kernel_rows() + antigravity_rows()
    markdown = render(rows)
    print(markdown)
    if args.write:
        args.write.parent.mkdir(parents=True, exist_ok=True)
        args.write.write_text(markdown + "\n", encoding="utf-8")
        print(f"wrote {args.write}")
    if args.json:
        args.json.write_text(json.dumps(rows, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"wrote {args.json}")


if __name__ == "__main__":
    main()
