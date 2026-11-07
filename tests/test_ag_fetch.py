from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from ag_fetch import fetch_with_retry, is_transient, last_final, recover_from_log  # noqa: E402

STUB = '''
import json, os, sys
from pathlib import Path

argv = sys.argv[1:]
out = Path(argv[argv.index("--out") + 1])
state = Path(argv[argv.index("--state") + 1])
counter = state.with_suffix(".count")
attempt = int(counter.read_text()) if counter.exists() else 0
counter.write_text(str(attempt + 1))
fails_file = state.with_suffix(".fails")
fails = int(fails_file.read_text()) if fails_file.exists() else 0
if attempt < fails:
    print('{"http": 503, "error": "Environment snapshot is temporarily unavailable."}')
    sys.exit(1)
target = out / "extracted" / "workspace" / "research" / "demo.md"
target.parent.mkdir(parents=True, exist_ok=True)
target.write_text("ok\\n")
print(json.dumps({"bytes": 2}))
'''


def make_stub(tmp_path: Path, fails: int) -> Path:
    stub = tmp_path / "stub_fetch.py"
    stub.write_text(STUB, encoding="utf-8")
    (tmp_path / "state.json").write_text(json.dumps({"interaction_id": "i"}), encoding="utf-8")
    (tmp_path / "state.fails").write_text(str(fails), encoding="utf-8")
    return stub


def test_fetch_retries_until_success(tmp_path: Path) -> None:
    stub = make_stub(tmp_path, fails=2)
    state = tmp_path / "state.json"
    result = fetch_with_retry(
        stub,
        state,
        tmp_path / "out",
        retries=4,
        base_delay=0.0,
        sleep=lambda _delay: None,
    )
    assert result["status"] == "ok"
    assert result["attempts"] == 3
    assert (tmp_path / "out" / "extracted" / "workspace" / "research" / "demo.md").exists()


def test_fetch_falls_back_to_log(tmp_path: Path) -> None:
    stub = make_stub(tmp_path, fails=99)
    log = tmp_path / "run.log"
    log.write_text(
        'noise\n{"http": 200, "status": "completed", "final": "## report\\nfindings"}\n',
        encoding="utf-8",
    )
    result = fetch_with_retry(
        stub,
        tmp_path / "state.json",
        tmp_path / "out",
        log_path=log,
        retries=2,
        base_delay=0.0,
        sleep=lambda _delay: None,
    )
    assert result["status"] == "recovered"
    recovered = Path(result["final"]).read_text(encoding="utf-8")
    assert "findings" in recovered


def test_non_transient_error_stops_immediately(tmp_path: Path) -> None:
    stub = tmp_path / "bad.py"
    stub.write_text("import sys\nprint('403 forbidden: bad token')\nsys.exit(1)\n", encoding="utf-8")
    (tmp_path / "state.json").write_text("{}", encoding="utf-8")
    result = fetch_with_retry(
        stub,
        tmp_path / "state.json",
        tmp_path / "out",
        retries=5,
        base_delay=0.0,
        sleep=lambda _delay: None,
    )
    assert result["status"] == "failed"
    assert result["attempts"] == 1  # non-transient errors are not retried


def test_is_transient_classifies_503() -> None:
    assert is_transient('{"http": 503, "message": "temporarily unavailable"}')
    assert is_transient("Deadline exceeded")
    assert not is_transient("403 forbidden")


def test_last_final_takes_the_newest_object(tmp_path: Path) -> None:
    log = tmp_path / "run.log"
    log.write_text(
        '{"final": "old"}\n{"status": "completed", "final": "new"}\n', encoding="utf-8"
    )
    assert last_final(log) == "new"
    assert recover_from_log(log, tmp_path / "out").exists()


def test_ag_tasks_recover_uses_the_log(tmp_path: Path) -> None:
    task = tmp_path / "antigravity" / "tasks" / "demo"
    task.mkdir(parents=True)
    log = tmp_path / "demo.log"
    log.write_text('{"final": "recovered text"}\n', encoding="utf-8")
    (task / "prompt.md").write_text("p\n", encoding="utf-8")
    (task / "meta.json").write_text(
        json.dumps({"slug": "demo", "log": str(log), "deliverables": []}), encoding="utf-8"
    )
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "ag_tasks.py"),
            "--root",
            str(tmp_path),
            "recover",
            "demo",
            "--out",
            str(tmp_path / "out"),
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert (tmp_path / "out" / "final.md").exists()
