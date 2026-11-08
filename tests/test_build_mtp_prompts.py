"""Prompt collection must dedupe and parse the dialogue formats it meets."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from build_mtp_prompts import first_user_turn, key_of, normalize  # noqa: E402


def test_normalize_collapses_whitespace() -> None:
    assert normalize("  a\n\n b\t c ") == "a b c"


def test_key_is_case_insensitive_and_whitespace_insensitive() -> None:
    assert key_of("Hello   World") == key_of("hello world")


def test_parses_stringified_dialogue() -> None:
    row = {"conversations": "[{'from': 'human', 'value': '質問です'}, {'from': 'gpt', 'value': '答え'}]"}
    assert first_user_turn(row) == "質問です"


def test_parses_list_dialogue_with_input_keys() -> None:
    row = {"conversation": [{"input": "first question", "output": "answer"}]}
    assert first_user_turn(row) == "first question"


def test_ignores_rows_without_a_user_turn() -> None:
    assert first_user_turn({"conversation": [{"output": "only"}]}) is None
