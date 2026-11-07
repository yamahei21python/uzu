from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from eval_quantized import bpc, scored_chars  # noqa: E402


class StubTokenizer:
    """decode() maps ids to single letters so char counts are easy to reason about."""

    def decode(self, ids: list[int]) -> str:
        return "".join(chr(ord("a") + index) for index in ids)


def test_scored_chars_drops_context_tokens() -> None:
    chunks = [[0, 1, 2], [3, 4]]
    # full decode = abcde (5 chars); skipped first tokens = a + d (2 chars)
    assert scored_chars(StubTokenizer(), chunks) == 3


def test_bpc_matches_the_definition() -> None:
    # PPL 2, 3 scored tokens, 4 chars -> log2(2) * 3/4 = 0.75
    assert bpc(2.0, 3, 4) == 0.75


def test_bpc_rejects_degenerate_inputs() -> None:
    assert bpc(2.0, 0, 4) is None
    assert bpc(2.0, 3, 0) is None
    assert bpc(0.0, 3, 4) is None
