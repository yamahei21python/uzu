from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from build_kit import detect_capabilities  # noqa: E402


def test_capabilities_are_detected_from_source(tmp_path: Path) -> None:
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    (scripts / "rht_int4_convert.py").write_text('add_argument("--layer-filter")\n', encoding="utf-8")
    (scripts / "eval_quantized.py").write_text('add_argument("--dtype")\n', encoding="utf-8")
    (scripts / "preflight.py").write_text("def verdict():\n    pass\n# sdpa_xla\n", encoding="utf-8")
    caps = detect_capabilities(tmp_path)
    assert "rht_int4_convert.layer_filter" in caps
    assert "eval_quantized.dtype" in caps
    assert "preflight.sdpa_xla" in caps
    assert "env_fingerprint" not in caps


def test_missing_files_are_not_claimed(tmp_path: Path) -> None:
    assert detect_capabilities(tmp_path) == []
