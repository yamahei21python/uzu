from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from cleanup_local import candidates  # noqa: E402
from watch_kernels import is_terminal, slug  # noqa: E402


def test_candidates_keep_newest_per_prefix(tmp_path: Path) -> None:
    for index in range(3):
        path = tmp_path / f"emb8_kaggle_out_v{index}"
        path.mkdir()
        (path / "f").write_text("x")
        import os, time
        os.utime(path, (time.time() + index, time.time() + index))
    selected = candidates(tmp_path, ("emb8_kaggle_out",), keep=1)
    assert sorted(path.name for path in selected) == ["emb8_kaggle_out_v0", "emb8_kaggle_out_v1"]


def test_slug_and_terminal() -> None:
    assert slug("yamahei21python/uzu-lane-runner") == "uzu-lane-runner"
    assert is_terminal('kernel has status "KernelWorkerStatus.COMPLETE"')
    assert is_terminal("KernelWorkerStatus.ERROR")
    assert not is_terminal("KernelWorkerStatus.RUNNING")
