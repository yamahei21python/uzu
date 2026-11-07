from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def make_task(root: Path, slug: str, *, status: str, deliverables: list[str], fetch: Path) -> None:
    task = root / "antigravity" / "tasks" / slug
    task.mkdir(parents=True)
    (task / "prompt.md").write_text("do the thing\n", encoding="utf-8")
    (task / "meta.json").write_text(
        json.dumps(
            {
                "slug": slug,
                "label": slug,
                "state": str(root / "state.json"),
                "log": str(root / "task.log"),
                "fetch_dir": str(fetch),
                "deliverables": deliverables,
                "status": status,
                "started": "2026-11-07T00:00:00",
                "note": "fixture",
            }
        ),
        encoding="utf-8",
    )


def run(root: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "ag_tasks.py"), "--root", str(root), *args],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )


def test_list_reports_tasks_and_collected_status(tmp_path: Path) -> None:
    make_task(tmp_path, "demo", status="collected", deliverables=["research/demo.md"], fetch=tmp_path / "fetch")
    result = run(tmp_path, "list")
    assert result.returncode == 0, result.stderr
    assert "demo" in result.stdout
    assert "collected" in result.stdout
    assert "0/1" in result.stdout


def test_collect_copies_deliverable_to_repo_path(tmp_path: Path) -> None:
    fetch = tmp_path / "fetch"
    source = fetch / "extracted" / "workspace" / "research" / "demo.md"
    source.parent.mkdir(parents=True)
    source.write_text("findings\n", encoding="utf-8")
    make_task(tmp_path, "demo", status="running", deliverables=["research/demo.md"], fetch=fetch)

    result = run(tmp_path, "collect", "demo")
    assert result.returncode == 0, result.stdout + result.stderr
    target = tmp_path / "research" / "demo.md"
    assert target.read_text(encoding="utf-8") == "findings\n"

    meta = json.loads((tmp_path / "antigravity" / "tasks" / "demo" / "meta.json").read_text(encoding="utf-8"))
    assert meta["status"] == "collected"
    assert meta["collected_at"]


def test_collect_moves_unexpected_paths_under_research_antigravity(tmp_path: Path) -> None:
    fetch = tmp_path / "fetch"
    source = fetch / "extracted" / "workspace" / "tmp" / "notes.txt"
    source.parent.mkdir(parents=True)
    source.write_text("x\n", encoding="utf-8")
    make_task(tmp_path, "demo", status="running", deliverables=["tmp/notes.txt"], fetch=fetch)

    result = run(tmp_path, "collect", "demo")
    assert result.returncode == 0, result.stdout + result.stderr
    assert (tmp_path / "research" / "antigravity" / "demo" / "notes.txt").exists()


def test_collect_without_fetch_fails_loudly(tmp_path: Path) -> None:
    make_task(tmp_path, "demo", status="running", deliverables=["research/demo.md"], fetch=tmp_path / "nofetch")
    result = run(tmp_path, "collect", "demo")
    assert result.returncode != 0
    assert "nothing fetched" in result.stderr


def test_show_includes_prompt(tmp_path: Path) -> None:
    make_task(tmp_path, "demo", status="running", deliverables=[], fetch=tmp_path / "fetch")
    result = run(tmp_path, "show", "demo")
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["prompt"].strip() == "do the thing"
    assert payload["slug"] == "demo"
