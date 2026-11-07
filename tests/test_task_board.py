from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from task_board import render  # noqa: E402


def test_render_buckets_by_status() -> None:
    rows = [
        {"source": "owner", "name": "a", "lane": "x", "status": "running"},
        {"source": "muse", "name": "b", "lane": "y", "status": "reported"},
        {"source": "antigravity", "name": "c", "lane": "z", "status": "queued"},
        {"source": "planned", "name": "d", "lane": "w", "status": "planned"},
        {"source": "owner", "name": "e", "lane": "v", "status": "failed"},
        {"source": "owner", "name": "f", "lane": "u", "status": "done"},
    ]
    markdown = render(rows)
    assert "## 実行中（2）" in markdown
    assert "## 待ち（キュー投入済み）（1）" in markdown
    assert "## 未投入（1）" in markdown
    assert "## 失敗（要対応）（1）" in markdown
    assert "## 完了（1）" in markdown
    assert "| owner | a | x | running |" in markdown
