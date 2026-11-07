"""Model targets and defaults for the Modal pipeline (no modal import)."""

from __future__ import annotations

import json
from pathlib import Path

CONFIG_ROOT = Path(__file__).resolve().parents[1] / "experiments"

MODEL_SPECS = {
    "0.8b": {
        "repo": "Qwen/Qwen3.5-0.8B",
        "spec_name": "Qwen3.5-0.8B",
        "layers": 24,
        "hidden": 1024,
    },
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


def model_config(size: str) -> dict:
    """Read experiments/qwen35-<size>/config.json (single source of truth)."""
    spec(size)
    path = CONFIG_ROOT / f"qwen35-{size}" / "config.json"
    return json.loads(path.read_text(encoding="utf-8"))


def calibration_defaults(size: str) -> dict:
    """Calibration hyperparameters for this size.

    The Modal entrypoint used to hardcode damping 1e-2 and max_tokens 512,
    which silently disagreed with the config (1e-3 and 256 for 4B).
    """
    return dict(model_config(size)["calibration"])


def qad_defaults(size: str) -> dict:
    return dict(model_config(size).get("qad", {}))
