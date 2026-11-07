from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

CORPORA = {
    "eval_en_wikitext2_dedup.txt": "the english evaluation sentence that is long enough to count\n",
    "eval_ja_wikipedia_dedup.txt": "これは日本語の評価用文章で、十分な長さを持っています。\n",
    "calibration_en_wikitext2_dedup.txt": "the english calibration sentence that is long enough to count\n",
    "calibration_ja_wikipedia_dedup.txt": "これは日本語の校正用文章で、十分な長さを持っています。\n",
    "calibration_mixed_en_ja_dedup.txt": "the english calibration sentence that is long enough to count\n"
    "これは日本語の校正用文章で、十分な長さを持っています。\n",
    "calibration_mixed_ja65_en35_dedup.txt": "the english calibration sentence that is long enough to count\n"
    "これは日本語の校正用文章で、十分な長さを持っています。\n",
}


def write_corpora(root: Path) -> None:
    for name, text in CORPORA.items():
        (root / name).write_text(text * 2, encoding="utf-8")


def run_audit(corpus_dir: Path, output_dir: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "audit_public_corpus.py"),
            "--no-public",
            "--corpus-dir",
            str(corpus_dir),
            "--output-dir",
            str(output_dir),
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )


def test_cross_corpus_audit_keeps_eval_and_calibration_disjoint(tmp_path: Path) -> None:
    corpus_dir = tmp_path / "corpora"
    corpus_dir.mkdir()
    write_corpora(corpus_dir)
    output = tmp_path / "out"

    result = run_audit(corpus_dir, output)
    assert result.returncode == 0, result.stdout + result.stderr
    assert (output / "public_corpus_audit.json").exists()
    assert (output / "public_corpus_audit.md").exists()

    payload = json.loads((output / "public_corpus_audit.json").read_text(encoding="utf-8"))
    cross = {(row["corpus"], row["target"]): row for row in payload["cross_results"]}
    assert cross[("eval_en", "cal_en")]["norm_matched_lines"] == 0
    assert cross[("eval_ja", "cal_ja")]["norm_matched_lines"] == 0
    assert cross[("eval_en", "eval_ja")]["norm_matched_lines"] == 0
    # the mixed corpora intentionally reuse the same calibration lines
    assert cross[("cal_en", "cal_mixed_en_ja")]["norm_matched_lines"] > 0


def test_no_public_run_marks_the_report(tmp_path: Path) -> None:
    corpus_dir = tmp_path / "corpora"
    corpus_dir.mkdir()
    write_corpora(corpus_dir)
    output = tmp_path / "out"
    result = run_audit(corpus_dir, output)
    assert result.returncode == 0, result.stdout + result.stderr
    report = (output / "public_corpus_audit.md").read_text(encoding="utf-8")
    assert "--no-public" in report


def test_audit_prints_its_progress(tmp_path: Path) -> None:
    corpus_dir = tmp_path / "corpora"
    corpus_dir.mkdir()
    write_corpora(corpus_dir)
    result = run_audit(corpus_dir, tmp_path / "out")
    # the script exits via os._exit; stdout must be flushed before that
    assert "Audit completed successfully" in result.stdout
