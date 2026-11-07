"""Model targets and defaults for the Modal pipeline (no modal import)."""

from __future__ import annotations

MODEL_SPECS = {
    "4b": {
        "repo": "huihui-ai/Huihui-Qwen3.5-4B-Claude-4.6-Opus-abliterated",
        "spec_name": "Qwen3.5-4B",
        "layers": 32,
        "hidden": 2560,
    },
    "9b": {
        "repo": "huihui-ai/Huihui-Qwen3.5-9B-Claude-4.6-Opus-abliterated",
        "spec_name": "Qwen3.5-9B",
        "layers": 32,
        "hidden": 4096,
    },
}

COMMANDS = ("check", "import", "rtn", "calibrate", "yaqa", "eval", "qad", "fetch")


def spec(size: str) -> dict:
    if size not in MODEL_SPECS:
        raise SystemExit(f"unknown size {size}; choose {sorted(MODEL_SPECS)}")
    return MODEL_SPECS[size]


def default_run_name(size: str) -> str:
    spec(size)
    return f"huihui-{size}"
