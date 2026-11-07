from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

STUB = '''
import json, os, sys, time
from pathlib import Path

argv = sys.argv[1:]
state = Path(argv[argv.index("--state") + 1])
if "run" in argv:
    label = argv[argv.index("--label") + 1]
    time.sleep(0.2)
    state.write_text(json.dumps({"interaction_id": "i-" + label, "environment_id": "e-1",
                                 "updated": "test", "label": label}), encoding="utf-8")
elif "fetch" in argv:
    out = Path(argv[argv.index("--out") + 1])
    label = json.loads(state.read_text())["label"]
    target = out / "extracted" / "workspace" / "research" / f"{label}.md"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("result for " + label + "\\n", encoding="utf-8")
'''


def run(root: Path, wrapper: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "ag_queue.py"), "--root", str(root), "--wrapper", str(wrapper), *args],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )


def setup(tmp_path: Path) -> tuple[Path, Path]:
    wrapper = tmp_path / "stub_wrapper.py"
    wrapper.write_text(STUB, encoding="utf-8")
    prompt = tmp_path / "prompt.txt"
    prompt.write_text("do the thing\n", encoding="utf-8")
    return wrapper, prompt


def add_ticket(root: Path, wrapper: Path, prompt: Path, slug: str, priority: int = 50) -> None:
    result = run(
        root,
        wrapper,
        "add",
        "--slug",
        slug,
        "--prompt",
        str(prompt),
        "--deliverables",
        f"research/{slug}.md",
        "--priority",
        str(priority),
    )
    assert result.returncode == 0, result.stderr
    # keep per-test state files inside tmp so tests never touch /tmp/ag_*.json
    meta_path = root / "antigravity" / "tasks" / slug / "meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    meta["state"] = str(root / f"state-{slug}.json")
    meta["fetch_dir"] = str(root / f"fetch-{slug}")
    meta["log"] = str(Path("logs") / f"{slug}.log")
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")


def pump(root: Path, wrapper: Path, max_active: int = 2) -> subprocess.CompletedProcess[str]:
    return run(root, wrapper, "pump", "--max-active", str(max_active))


def test_add_and_list(tmp_path: Path) -> None:
    wrapper, prompt = setup(tmp_path)
    add_ticket(tmp_path, wrapper, prompt, "alpha", priority=90)
    result = run(tmp_path, wrapper, "list")
    assert result.returncode == 0, result.stderr
    assert "alpha" in result.stdout
    assert "queued" in result.stdout
    assert (tmp_path / "antigravity" / "tasks" / "alpha" / "prompt.md").exists()


def test_pump_respects_max_active(tmp_path: Path) -> None:
    wrapper, prompt = setup(tmp_path)
    for slug in ("a1", "a2", "a3"):
        add_ticket(tmp_path, wrapper, prompt, slug)
    first = pump(tmp_path, wrapper, max_active=2)
    assert first.returncode == 0, first.stderr
    assert first.stdout.count("started ") == 2
    listing = json.loads(run(tmp_path, wrapper, "list", "--json").stdout)
    statuses = {row["slug"]: row["status"] for row in listing}
    assert sorted(statuses.values()) == ["queued", "running", "running"]


def test_pump_reconciles_and_collects(tmp_path: Path) -> None:
    wrapper, prompt = setup(tmp_path)
    add_ticket(tmp_path, wrapper, prompt, "a1")
    pump(tmp_path, wrapper, max_active=2)
    time.sleep(0.8)  # stub writes its state after 0.2s
    second = pump(tmp_path, wrapper, max_active=2)
    assert second.returncode == 0, second.stderr
    listing = json.loads(run(tmp_path, wrapper, "list", "--json").stdout)
    assert listing[0]["status"] == "done"
    copied = tmp_path / "research" / "a1.md"
    assert copied.read_text(encoding="utf-8") == "result for a1\n"


def test_priority_order(tmp_path: Path) -> None:
    wrapper, prompt = setup(tmp_path)
    add_ticket(tmp_path, wrapper, prompt, "low", priority=10)
    add_ticket(tmp_path, wrapper, prompt, "high", priority=99)
    result = pump(tmp_path, wrapper, max_active=1)
    assert "started high" in result.stdout
    assert "started low" not in result.stdout


def test_select_deletable_keeps_active_and_recent(tmp_path: Path) -> None:
    sys.path.insert(0, str(ROOT / "scripts"))
    try:
        from ag_queue import select_deletable
    finally:
        sys.path.pop(0)

    environments = [
        {"id": "old-idle", "created": "2026-11-01T00:00:00Z"},
        {"id": "old-active", "created": "2026-11-01T00:00:00Z"},
        {"id": "no-created"},
    ]
    selected = select_deletable(environments, keep={"old-active"}, older_than_days=1)
    assert [env["id"] for env in selected] == ["old-idle"]


def test_cleanup_envs_offline_dry_run(tmp_path: Path) -> None:
    env_list = tmp_path / "envs.json"
    env_list.write_text(
        json.dumps(
            {
                "environments": [
                    {"id": "old-idle", "created": "2026-11-01T00:00:00Z", "size_bytes": "100"},
                    {"id": "old-active", "created": "2026-11-01T00:00:00Z", "size_bytes": "100"},
                    {"id": "fresh", "created": "2099-01-01T00:00:00Z", "size_bytes": "100"},
                ]
            }
        ),
        encoding="utf-8",
    )
    result = run(
        tmp_path,
        tmp_path / "stub_wrapper.py",
        "cleanup-envs",
        "--older-than-days",
        "1",
        "--dry-run",
        "--env-list-file",
        str(env_list),
        "--keep",
        "old-active",
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "old-idle" in result.stdout
    assert "old-active" not in result.stdout
    assert "fresh" not in result.stdout
    assert '"dry_run": true' in result.stdout


def test_manual_task_occupies_a_slot(tmp_path: Path) -> None:
    wrapper, prompt = setup(tmp_path)
    add_ticket(tmp_path, wrapper, prompt, "manual-one")
    meta_path = tmp_path / "antigravity" / "tasks" / "manual-one" / "meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    meta["queue_status"] = "manual"
    meta_path.write_text(json.dumps(meta), encoding="utf-8")
    for slug in ("q1", "q2"):
        add_ticket(tmp_path, wrapper, prompt, slug)

    result = pump(tmp_path, wrapper, max_active=2)
    assert result.returncode == 0, result.stderr
    assert result.stdout.count("started ") == 1
    assert "active=2" in result.stdout


def test_retry_requeues_failed_ticket(tmp_path: Path) -> None:
    wrapper, prompt = setup(tmp_path)
    add_ticket(tmp_path, wrapper, prompt, "a1")
    meta_path = tmp_path / "antigravity" / "tasks" / "a1" / "meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    meta["queue_status"] = "failed"
    meta_path.write_text(json.dumps(meta), encoding="utf-8")
    result = run(tmp_path, wrapper, "retry", "a1")
    assert result.returncode == 0, result.stderr
    assert json.loads(meta_path.read_text(encoding="utf-8"))["queue_status"] == "queued"


def test_cancel_marks_terminal(tmp_path: Path) -> None:
    wrapper, prompt = setup(tmp_path)
    add_ticket(tmp_path, wrapper, prompt, "a1")
    result = run(tmp_path, wrapper, "cancel", "a1")
    assert result.returncode == 0, result.stderr
    meta = json.loads((tmp_path / "antigravity" / "tasks" / "a1" / "meta.json").read_text(encoding="utf-8"))
    assert meta["queue_status"] == "cancelled"
