from __future__ import annotations

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load():
    spec = importlib.util.spec_from_file_location(
        "vllm_acceptance", ROOT / "scripts" / "vllm_acceptance.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


SAMPLE = """\
# HELP vllm:spec_decode_num_drafts_total drafts
# TYPE vllm:spec_decode_num_drafts_total counter
vllm:spec_decode_num_drafts_total{model_name="x"} 400.0
vllm:spec_decode_num_draft_tokens_total{model_name="x"} 3200.0
vllm:spec_decode_num_accepted_tokens_total{model_name="x"} 1600.0
vllm:spec_decode_num_accepted_tokens_total{model_name="y"} 40.0
other_metric 7.0
"""

WINDOWED = SAMPLE.replace("400.0", "460.0").replace("3200.0", "3880.0").replace(
    '"x"} 1600.0', '"x"} 1840.0'
).replace('"y"} 40.0', '"y"} 60.0')


def test_parse_sums_labels_and_ignores_others() -> None:
    module = load()
    totals = module.parse_metrics(SAMPLE)
    assert totals["vllm:spec_decode_num_drafts_total"] == 400.0
    assert totals["vllm:spec_decode_num_draft_tokens_total"] == 3200.0
    # both label sets are summed
    assert totals["vllm:spec_decode_num_accepted_tokens_total"] == 1640.0


def test_summarize_reports_acceptance_and_tokens_per_draft() -> None:
    module = load()
    line = module.summarize(module.parse_metrics(SAMPLE))
    assert "acceptance=0.512" in line
    assert "tokens/draft=8.00" in line


def test_windowed_summary_uses_the_delta_only() -> None:
    module = load()
    previous = module.parse_metrics(SAMPLE)
    current = module.parse_metrics(WINDOWED)
    line = module.summarize(current, previous)
    # 60 drafts, 680 proposed, 260 accepted in the window
    assert "drafts=60" in line
    assert "proposed=680" in line
    assert "acceptance=0.382" in line


def test_zero_drafts_does_not_divide_by_zero() -> None:
    module = load()
    empty = {name: 0.0 for name in module.COUNTERS}
    line = module.summarize(empty)
    assert "acceptance=0.000" in line
    assert "tokens/draft=0.00" in line
