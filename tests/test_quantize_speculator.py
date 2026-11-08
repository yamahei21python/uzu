"""The speculator quantizer must produce uzu-loadable metadata."""

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "quantize_speculator.py"


def test_help_lists_the_quantization_options() -> None:
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--help"],
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    for flag in ("--source", "--output", "--bits", "--group-size", "--shapes"):
        assert flag in result.stdout


def test_int8_requires_symmetric() -> None:
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--source", "/tmp/x", "--output", "/tmp/y", "--bits", "8"],
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode != 0
    assert "symmetric" in (result.stdout + result.stderr)


def test_rewrite_is_applied_unless_disabled() -> None:
    text = SCRIPT.read_text(encoding="utf-8")
    # uzu rejects the nested form lalamo writes, so the rewrite must be the default
    assert "--no-rewrite" in text
    assert "rewrite(args.output" in text
