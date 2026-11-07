from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from ag_fetch import log_completed  # noqa: E402

STUB = '''
import json, sys
from pathlib import Path
argv = sys.argv[1:]
if "fetch" in argv:
    out = Path(argv[argv.index("--out") + 1])
    state = Path(argv[argv.index("--state") + 1])
    label = json.loads(state.read_text())["label"]
    target = out / "extracted" / "workspace" / "research" / f"{label}.md"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("result\\n")
'''


def test_log_completed_detects_a_finished_run(tmp_path: Path) -> None:
    log = tmp_path / "run.log"
    log.write_text('{"status": "failed"}\n{"status": "completed", "final": "ok"}\n', encoding="utf-8")
    assert log_completed(log) is True
    log.write_text('{"status": "failed"}\n', encoding="utf-8")
    assert log_completed(log) is False
    assert log_completed(tmp_path / "absent.log") is False


def test_dead_pid_with_completed_log_is_fetched_not_failed(tmp_path: Path) -> None:
    stub = tmp_path / "stub.py"
    stub.write_text(STUB, encoding="utf-8")

    task = tmp_path / "antigravity" / "tasks" / "demo"
    task.mkdir(parents=True)
    (task / "prompt.md").write_text("p\n", encoding="utf-8")
    log = tmp_path / "demo.log"
    log.write_text('{"status": "completed", "final": "done"}\n', encoding="utf-8")
    state = tmp_path / "state.json"
    # no state file: the wrapper only writes it after returning, so the log is
    # the only evidence that the run finished
    state.write_text(json.dumps({"label": "demo"}), encoding="utf-8")
    (task / "meta.json").write_text(
        json.dumps(
            {
                "slug": "demo",
                "label": "demo",
                "state": str(state),
                "log": str(log),
                "fetch_dir": str(tmp_path / "fetch"),
                "deliverables": ["research/demo.md"],
                "queue_status": "running",
                "status": "running",
                "pid": 999999,
                "attempts": 1,
                "priority": 50,
            }
        ),
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "ag_queue.py"),
            "--root",
            str(tmp_path),
            "--wrapper",
            str(stub),
            "pump",
            "--max-active",
            "2",
            "--fetch-retries",
            "1",
            "--fetch-base-delay",
            "0",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "finished in the log" in result.stdout
    meta = json.loads((task / "meta.json").read_text(encoding="utf-8"))
    assert meta["queue_status"] == "done"
    assert meta["status"] in {"collected", "recovered"}
    assert (tmp_path / "research" / "demo.md").exists()


def test_dead_pid_without_log_completion_waits_before_failing(tmp_path: Path) -> None:
    task = tmp_path / "antigravity" / "tasks" / "demo"
    task.mkdir(parents=True)
    (task / "prompt.md").write_text("p\n", encoding="utf-8")
    log = tmp_path / "demo.log"
    log.write_text("nothing yet\n", encoding="utf-8")
    (task / "meta.json").write_text(
        json.dumps(
            {
                "slug": "demo",
                "state": str(tmp_path / "state.json"),
                "log": str(log),
                "fetch_dir": str(tmp_path / "fetch"),
                "deliverables": [],
                "queue_status": "running",
                "status": "running",
                "pid": 999999,
                "attempts": 1,
                "priority": 50,
            }
        ),
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "ag_queue.py"),
            "--root",
            str(tmp_path),
            "pump",
            "--max-active",
            "2",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    meta = json.loads((task / "meta.json").read_text(encoding="utf-8"))
    # first sighting only starts the grace timer
    assert meta["queue_status"] == "running"
    assert "pid_gone_at" in meta
