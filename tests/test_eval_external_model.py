"""External-model eval must know which corpus is which language."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from eval_external_model import language_of  # noqa: E402


def test_language_from_known_corpora() -> None:
    assert language_of(Path("data/eval_en_wikitext2_dedup.txt")) == "en"
    assert language_of(Path("data/eval_ja_wikipedia_dedup.txt")) == "ja"


def test_unknown_corpus_falls_back_on_the_prefix() -> None:
    assert language_of(Path("eval_en_custom.txt")) == "en"
    assert language_of(Path("custom.txt")) == "unknown"
