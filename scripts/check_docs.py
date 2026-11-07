#!/usr/bin/env python3
"""Check that commands in the docs still match the scripts.

Docs drift caught us twice: RUNBOOK E2/E3 lacked --eval-corpus (the guard exits
immediately), and Step 3 lacked --calibration-json (provenance becomes
overlap_checked=false and QAD later refuses the model). Both were executable
lines in markdown.

This scans fenced code blocks for ``python scripts/<name>.py ...`` lines, then
verifies:
- the script exists
- every ``--flag`` is declared in that script's ``add_argument`` calls
- ``--spec lanes/*.json`` points at an existing file

Usage:
    python scripts/check_docs.py
    python scripts/check_docs.py --strict
    python scripts/check_docs.py --root . --file experiments/qwen35-0.8b/RUNBOOK.md
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

DEFAULT_DOCS = [
    "README.md",
    "experiments/PROTOCOL.md",
    "experiments/qwen35-0.8b/RUNBOOK.md",
    "experiments/qwen35-0.8b/EXPERIMENT_MATRIX.md",
    "lanes/README.md",
    "docs/LOCAL_POLICY.md",
    "docs/KAGGLE_SLOTS.md",
    "docs/EVAL.md",
    "docs/MODAL.md",
    "docs/ENVIRONMENT.md",
    "docs/STATUS.md",
    "tests/README.md",
    "antigravity/README.md",
]

FENCE = re.compile(r"```(?:bash|sh)\n(.*?)```", re.S)
COMMAND = re.compile(r"(?:^|\s)(?:[A-Z_]+=\S+\s+)*python3?\s+(scripts/[\w.-]+\.py)([^\n|]*)")


def code_blocks(text: str) -> list[tuple[int, str]]:
    blocks = []
    for match in FENCE.finditer(text):
        line = text[: match.start()].count("\n") + 2
        blocks.append((line, match.group(1)))
    return blocks


def extract_commands(text: str) -> list[dict]:
    commands = []
    for block_line, block in code_blocks(text):
        for offset, line in enumerate(block.splitlines()):
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            for match in COMMAND.finditer(stripped):
                commands.append(
                    {
                        "line": block_line + offset,
                        "script": match.group(1),
                        "args": match.group(2).strip(),
                    }
                )
    return commands


def declaration_flags(script: Path) -> set[str]:
    text = script.read_text(encoding="utf-8")
    return set(re.findall(r'["\'](--[a-z0-9][a-z0-9-]*)["\']', text))


def command_flags(args: str) -> set[str]:
    # stop at a shell pipe/redirect
    cleaned = re.split(r"[|>]", args)[0]
    return set(re.findall(r"(--[a-z0-9][a-z0-9-]*)", cleaned))


def spec_paths(args: str) -> list[str]:
    cleaned = re.split(r"[|>]", args)[0]
    return re.findall(r"--spec\s+(\S+)", cleaned)


def check(root: Path, files: list[str]) -> list[dict]:
    problems: list[dict] = []
    for name in files:
        doc = root / name
        if not doc.exists():
            continue
        for command in extract_commands(doc.read_text(encoding="utf-8")):
            script = root / command["script"]
            if not script.exists():
                problems.append(
                    {"doc": name, "line": command["line"], "kind": "missing-script", "detail": command["script"]}
                )
                continue
            known = declaration_flags(script)
            unknown = sorted(command_flags(command["args"]) - known)
            if unknown:
                problems.append(
                    {
                        "doc": name,
                        "line": command["line"],
                        "kind": "unknown-flag",
                        "detail": f"{command['script']}: {', '.join(unknown)}",
                    }
                )
            for spec in spec_paths(command["args"]):
                if spec.startswith("$") or spec.startswith("/"):
                    continue
                if not (root / spec).exists():
                    problems.append(
                        {"doc": name, "line": command["line"], "kind": "missing-spec", "detail": spec}
                    )
    return problems


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--file", action="append", default=None, help="check only these docs")
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()

    files = args.file or DEFAULT_DOCS
    problems = check(args.root, files)
    checked = sum(1 for name in files if (args.root / name).exists())
    print(f"checked {checked} docs")
    if not problems:
        print("DOCS OK")
        return
    for problem in problems:
        print(f"{problem['doc']}:{problem['line']} [{problem['kind']}] {problem['detail']}")
    print(f"{len(problems)} problem(s)")
    if args.strict:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
