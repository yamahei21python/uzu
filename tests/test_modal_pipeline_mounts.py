from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_entrypoint_imports_are_mounted_next_to_it() -> None:
    text = (ROOT / "scripts" / "modal_pipeline.py").read_text(encoding="utf-8")
    imports = re.findall(r"^from (modal_targets) import", text, flags=re.MULTILINE)
    assert imports, "expected modal_pipeline to import modal_targets"
    assert 'add_local_file("scripts/modal_targets.py", "/root/modal_targets.py")' in text
