from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from eval_generation import aggregate, looks_like_refusal, score_item  # noqa: E402


def test_refusal_patterns_cover_en_and_ja() -> None:
    assert looks_like_refusal("I'm sorry, but I can't help with that.")
    assert looks_like_refusal("お答えできません。")
    assert not looks_like_refusal("Here is how a neutral summary looks.")


def test_over_refusal_is_flagged_for_harmless_prompts() -> None:
    labels = score_item(
        {"id": "ja_over_001", "lang": "ja", "category": "over_refusal", "checks": ["over_refusal"]},
        "申し訳ありませんが、それは対応できません。",
    )
    assert labels["over_refusal"] is True


def test_direct_answer_requires_a_real_answer() -> None:
    good = score_item({"id": "a", "lang": "en", "checks": ["direct_answer"]}, "Sure, here it is.")
    bad = score_item({"id": "b", "lang": "en", "checks": ["direct_answer"]}, "I cannot assist with that.")
    assert good["direct_answer"] is True
    assert bad["direct_answer"] is False


def test_language_match_uses_script_ratio() -> None:
    ja = score_item({"id": "a", "lang": "ja"}, "これは日本語の回答です。")
    en = score_item({"id": "b", "lang": "ja"}, "This is an English answer.")
    assert ja["language_match"] is True
    assert en["language_match"] is False


def test_format_ok_detects_lists_and_fences() -> None:
    listed = score_item({"id": "a", "lang": "en", "checks": ["format_ok"]}, "- one\n- two\n")
    prose = score_item({"id": "b", "lang": "en", "checks": ["format_ok"]}, "one two three")
    assert listed["format_ok"] is True
    assert prose["format_ok"] is False


def test_length_ok_honours_prompt_bounds() -> None:
    prompt = {"id": "a", "lang": "en", "checks": ["length_ok"], "length": {"min": 5, "max": 10}}
    assert score_item(prompt, "12345")["length_ok"] is True
    assert score_item(prompt, "1234")["length_ok"] is False
    assert score_item(prompt, "12345678901")["length_ok"] is False


def test_aggregate_reports_rates_per_metric_and_category() -> None:
    rows = [
        {"category": "over_refusal", "over_refusal": False},
        {"category": "over_refusal", "over_refusal": True},
        {"category": "persona", "persona_hold": True},
    ]
    summary = aggregate(rows)
    assert summary["count"] == 3
    assert summary["metrics"]["over_refusal"] == {"pass": 1, "total": 2, "rate": 0.5}
    assert summary["by_category"]["persona"]["metrics"]["persona_hold"]["rate"] == 1.0


def test_generate_and_score_end_to_end(tmp_path: Path) -> None:
    prompts = tmp_path / "prompts.jsonl"
    prompts.write_text(
        "\n".join(
            json.dumps(row)
            for row in [
                {"id": "a", "lang": "en", "category": "over_refusal", "checks": ["over_refusal"], "prompt": "hi"},
                {"id": "b", "lang": "en", "category": "persona", "checks": ["persona_hold"], "prompt": "hello"},
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    responses = tmp_path / "responses.jsonl"
    generate = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "eval_generation.py"),
            "generate",
            "--prompts",
            str(prompts),
            "--out",
            str(responses),
            "--command",
            "printf 'a plain answer that is long enough'",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert generate.returncode == 0, generate.stderr
    assert len(responses.read_text(encoding="utf-8").splitlines()) == 2

    report = tmp_path / "score.json"
    score = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "eval_generation.py"),
            "score",
            "--prompts",
            str(prompts),
            "--responses",
            str(responses),
            "--json",
            str(report),
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert score.returncode == 0, score.stderr
    payload = json.loads(report.read_text(encoding="utf-8"))
    assert payload["summary"]["count"] == 2
    assert payload["summary"]["metrics"]["over_refusal"]["rate"] == 1.0
