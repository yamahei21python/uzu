from __future__ import annotations

import ast
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from build_kit import KIT_DATA, KIT_LANES, KIT_SCRIPTS, KIT_TESTS, check  # noqa: E402


def test_every_manifest_file_exists_in_repo() -> None:
    missing = []
    for name in KIT_SCRIPTS:
        if not (ROOT / "scripts" / name).exists():
            missing.append(name)
    for name in KIT_DATA:
        if not (ROOT / "experiments" / "qwen35-0.8b" / "data" / name).exists():
            missing.append(name)
    for name in KIT_LANES:
        if not (ROOT / "lanes" / name).exists():
            missing.append(name)
    for name in KIT_TESTS:
        if not (ROOT / "tests" / name).exists():
            missing.append(name)
    assert missing == []


def test_lazy_imports_resolve_inside_the_kit() -> None:
    """rht_int4_convert imports rewrite_lalamo_hybrid_spec inside a branch.

    A plain ``grep import`` misses it; AST walking does not.
    """
    tree = ast.parse((ROOT / "scripts" / "rht_int4_convert.py").read_text(encoding="utf-8"))
    imported = {
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module
    }
    imported |= {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    assert "rewrite_lalamo_hybrid_spec" in imported
    assert "rewrite_lalamo_hybrid_spec.py" in KIT_SCRIPTS


def test_build_kit_produces_a_valid_kit(tmp_path: Path) -> None:
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "build_kit.py"), "--output", str(tmp_path / "kit")],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "KIT OK" in result.stdout
    assert (tmp_path / "kit" / "scripts" / "rewrite_lalamo_hybrid_spec.py").exists()


def test_check_kit_flags_a_missing_script(tmp_path: Path) -> None:
    kit = tmp_path / "kit"
    subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "build_kit.py"), "--output", str(kit)],
        cwd=ROOT,
        check=True,
        capture_output=True,
    )
    (kit / "scripts" / "rewrite_lalamo_hybrid_spec.py").unlink()
    problems = check(kit)
    assert any("rewrite_lalamo_hybrid_spec" in problem for problem in problems)


def test_runtime_helpers_are_packaged() -> None:
    """These are only ever imported from inside a kernel, so a missing entry
    in the whitelist fails at run time rather than at import time."""
    assert "eval_external_model.py" in KIT_SCRIPTS
    assert "quantize_speculator.py" in KIT_SCRIPTS
