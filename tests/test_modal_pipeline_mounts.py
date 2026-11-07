from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PIPELINE = ROOT / "scripts" / "modal_pipeline.py"


def mounted_files(text: str) -> dict[str, str]:
    """Map local path -> remote path for every add_local_file call."""
    pairs = re.findall(r'add_local_file\(\s*"([^"]+)"\s*,\s*"([^"]+)"', text, flags=re.S)
    return dict(pairs)


def local_modules_in(script: Path) -> set[str]:
    """Modules imported by a script that also exist as scripts/<name>.py."""
    text = script.read_text(encoding="utf-8")
    names = set(re.findall(r"^from ([a-zA-Z_][\w]*) import", text, flags=re.MULTILINE))
    names |= set(re.findall(r"^import ([a-zA-Z_][\w]*)", text, flags=re.MULTILINE))
    return {name for name in names if (ROOT / "scripts" / f"{name}.py").exists()}


def test_entrypoint_imports_are_mounted_next_to_it() -> None:
    text = PIPELINE.read_text(encoding="utf-8")
    assert re.search(r"^from modal_targets import", text, flags=re.MULTILINE)
    assert 'add_local_file("scripts/modal_targets.py", "/root/modal_targets.py")' in text


def test_every_mounted_script_has_its_local_imports_mounted() -> None:
    text = PIPELINE.read_text(encoding="utf-8")
    mounts = mounted_files(text)
    missing = []
    for local_path in mounts:
        if not local_path.startswith("scripts/"):
            continue
        script = ROOT / local_path
        if not script.exists():
            continue
        for module in local_modules_in(script):
            target = f"scripts/{module}.py"
            if target not in mounts:
                missing.append(f"{local_path} imports {module}, but {target} is not mounted")
    assert not missing, "\n".join(sorted(set(missing)))
