from __future__ import annotations

import json
import math
import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("tokenizers")

from tokenizers import Tokenizer, models, pre_tokenizers  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def build_tokenizer(path: Path) -> None:
    vocab = {
        "<unk>": 0,
        "the": 1,
        "quick": 2,
        "brown": 3,
        "fox": 4,
        "jumps": 5,
        "over": 6,
        "lazy": 7,
        "dog": 8,
        "second": 9,
        "line": 10,
        "here": 11,
        "third": 12,
        "another": 13,
    }
    tokenizer = Tokenizer(models.WordLevel(vocab=vocab, unk_token="<unk>"))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer.save(str(path / "tokenizer.json"))


def write_fixture(tmp_path: Path, *, tokens: int, chunks: int, ppl: float) -> tuple[Path, Path]:
    corpus = tmp_path / "corpus.txt"
    corpus.write_text(
        "the quick brown fox jumps over the lazy dog\n"
        "second line here\n"
        "third another line here\n" * 4,
        encoding="utf-8",
    )
    tokenizer_dir = tmp_path / "tok"
    tokenizer_dir.mkdir()
    build_tokenizer(tokenizer_dir)
    payload = {
        "corpus": str(corpus),
        "config": {
            "corpus": str(corpus),
            "max_sequences": chunks,
            "max_tokens": 4,
            "min_tokens": 1,
        },
        "chunks": chunks,
        "tokens": tokens,
        "bf16_ppl": ppl,
        "candidate_ppl": ppl * 1.1,
    }
    ppl_json = tmp_path / "ppl_en.json"
    ppl_json.write_text(json.dumps(payload), encoding="utf-8")
    return ppl_json, tokenizer_dir


def run(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "compute_bpc.py"), *args],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )


def test_bpc_is_log2_ppl_times_fertility(tmp_path: Path) -> None:
    ppl_json, tokenizer_dir = write_fixture(tmp_path, tokens=9, chunks=3, ppl=8.0)
    out = tmp_path / "report.json"
    result = run("--ppl-json", str(ppl_json), "--tokenizer", str(tokenizer_dir), "--json", str(out))
    assert result.returncode == 0, result.stdout + result.stderr
    report = json.loads(out.read_text(encoding="utf-8"))[0]
    assert report["chunks"] == 3
    # every chunk contributes len(chunk) - 1 scored predictions
    assert report["tokens"] == report["chunks"] * 3
    assert report["chars"] > 0
    expected = math.log2(8.0) * report["tokens"] / report["chars"]
    assert report["bpc_bf16"] == pytest.approx(expected)
    assert report["bpc_candidate"] == pytest.approx(math.log2(8.8) * report["tokens"] / report["chars"])
    assert report["reproduced"]["chunks_match"] is True
    assert report["reproduced"]["tokens_match"] is True


def test_strict_mode_flags_token_drift(tmp_path: Path) -> None:
    ppl_json, tokenizer_dir = write_fixture(tmp_path, tokens=99, chunks=3, ppl=8.0)
    result = run("--ppl-json", str(ppl_json), "--tokenizer", str(tokenizer_dir), "--strict")
    assert result.returncode == 1
    assert "tokens_match" in result.stdout


def test_accepts_corpus_override(tmp_path: Path) -> None:
    ppl_json, tokenizer_dir = write_fixture(tmp_path, tokens=9, chunks=3, ppl=8.0)
    payload = json.loads(ppl_json.read_text(encoding="utf-8"))
    payload["corpus"] = str(tmp_path / "missing.txt")
    payload["config"]["corpus"] = str(tmp_path / "missing.txt")
    ppl_json.write_text(json.dumps(payload), encoding="utf-8")
    override = tmp_path / "corpus.txt"
    result = run("--ppl-json", str(ppl_json), "--tokenizer", str(tokenizer_dir), "--corpus", str(override))
    assert result.returncode == 0, result.stdout + result.stderr
