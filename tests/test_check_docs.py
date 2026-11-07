from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from check_docs import check, extract_commands, declaration_flags  # noqa: E402


def write_doc(root: Path, text: str) -> None:
    (root / "docs").mkdir(parents=True, exist_ok=True)
    (root / "docs" / "test.md").write_text(text, encoding="utf-8")


def write_script(root: Path, name: str, body: str) -> None:
    (root / "scripts").mkdir(parents=True, exist_ok=True)
    (root / "scripts" / name).write_text(body, encoding="utf-8")


def test_extract_commands_ignores_comments_and_prose(tmp_path: Path) -> None:
    doc = """
prose

```bash
# a comment
JAX_PLATFORM_NAME=cpu python scripts/eval_quantized.py --bf16-dir x --corpus y
python scripts/other.py
```
"""
    (tmp_path / "doc.md").write_text(doc, encoding="utf-8")
    commands = extract_commands(doc)
    assert [command["script"] for command in commands] == [
        "scripts/eval_quantized.py",
        "scripts/other.py",
    ]
    assert commands[0]["line"] == 6


def test_declaration_flags_reads_argparse(tmp_path: Path) -> None:
    script = tmp_path / "x.py"
    script.write_text(
        'parser.add_argument("--alpha")\nparser.add_argument("--beta-long", default=1)\n',
        encoding="utf-8",
    )
    assert declaration_flags(script) == {"--alpha", "--beta-long"}


def test_check_flags_unknown_flag_and_missing_script(tmp_path: Path) -> None:
    write_script(tmp_path, "ok.py", 'parser.add_argument("--known")\n')
    write_doc(
        tmp_path,
        """
```bash
python scripts/ok.py --known 1 --nope 2
python scripts/gone.py --x
```
""",
    )
    problems = check(tmp_path, ["docs/test.md"])
    kinds = sorted(problem["kind"] for problem in problems)
    assert kinds == ["missing-script", "unknown-flag"]


def test_check_flags_missing_spec_file(tmp_path: Path) -> None:
    write_script(tmp_path, "run_lane.py", 'parser.add_argument("--spec")\n')
    write_doc(tmp_path, "```bash\npython scripts/run_lane.py --spec lanes/none.json\n```\n")
    problems = check(tmp_path, ["docs/test.md"])
    assert problems[0]["kind"] == "missing-spec"


def test_cli_strict_exit_code(tmp_path: Path) -> None:
    write_script(tmp_path, "ok.py", 'parser.add_argument("--known")\n')
    write_doc(tmp_path, "```bash\npython scripts/ok.py --bad\n```\n")
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "check_docs.py"), "--root", str(tmp_path), "--file", "docs/test.md", "--strict"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 1
    assert "unknown-flag" in result.stdout


def test_real_docs_pass() -> None:
    problems = check(ROOT, ["experiments/qwen35-0.8b/RUNBOOK.md", "lanes/README.md", "docs/EVAL.md"])
    assert problems == [], json.dumps(problems, indent=2)
