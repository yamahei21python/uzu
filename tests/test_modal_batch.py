from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from modal_batch import build_commands, format_table, interesting_line  # noqa: E402


def test_matrix_expands_per_size_and_command() -> None:
    argv = build_commands(["4b", "9b"], ["import", "rtn"], "smoke-{size}")
    assert len(argv) == 4
    assert argv[0] == [
        ".deps/venv-lalamo/bin/modal",
        "run",
        "scripts/modal_pipeline.py",
        "--command",
        "import",
        "--size",
        "4b",
        "--run-name",
        "smoke-4b",
    ]
    assert argv[3][-1] == "smoke-9b"


def test_eval_gets_candidate_and_sequence_limit() -> None:
    argv = build_commands(["4b"], ["eval"], "smoke-{size}", eval_candidate="int4-rtn", max_sequences=20)
    assert argv[0][-4:] == ["--candidate", "int4-rtn", "--max-sequences", "20"]


def test_import_does_not_get_eval_flags() -> None:
    argv = build_commands(["4b"], ["import"], "smoke-{size}")
    assert "--candidate" not in argv[0]


def test_reused_run_name_would_collide_which_is_why_size_is_formatted() -> None:
    argv = build_commands(["4b", "9b"], ["import"], "smoke-{size}")
    names = [item[item.index("--run-name") + 1] for item in argv]
    assert names == ["smoke-4b", "smoke-9b"]


def test_output_line_prefers_result_paths_over_traceback() -> None:
    output = (
        "Traceback (most recent call last):\n"
        "ModuleNotFoundError: No module named 'modal_targets'\n"
        "/out/smoke-4b/int4-rtn\n"
    )
    assert interesting_line(output) == "/out/smoke-4b/int4-rtn"


def test_output_line_falls_back_to_last_line() -> None:
    assert interesting_line("a\nb\n") == "b"


def test_table_reports_timeout() -> None:
    text = format_table(
        [{"size": "4b", "command": "eval", "rc": "timeout", "seconds": 1800, "last": "exceeded 1800s"}]
    )
    assert "| 4b | eval | timeout | 1800 | exceeded 1800s |" in text


def test_remote_startup_failure_is_not_reported_as_success() -> None:
    from modal_batch import failure_marker

    output = (
        "Function fu-abc failed during startup.\n"
        "Runner failed with exception: ModuleNotFoundError(\"No module named 'modal_targets'\")\n"
    )
    assert failure_marker(output) == "failed during startup"


def test_generic_traceback_from_handled_code_is_not_a_failure_marker() -> None:
    from modal_batch import failure_marker

    output = (
        "Traceback (most recent call last):\n"
        "ModuleNotFoundError: No module named 'optional_dep'\n"
        "falling back to the slow path\n"
        "/out/smoke-4b/int4-rtn\n"
    )
    assert failure_marker(output) is None


def test_clean_output_has_no_failure_marker() -> None:
    from modal_batch import failure_marker

    assert failure_marker("/out/smoke-4b/int4-rtn\nApp completed\n") is None
