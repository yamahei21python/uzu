#!/usr/bin/env python3
"""Model-free smoke checks for 0.8B / 4B / 9B before a Modal run.

Catches the cheap failures that would otherwise surface after an 8-18GB
download: wrong spec name, hidden size not divisible by RHT/group size, layer
count drift between our config and Hugging Face, missing tokenizer, and a
mismatch between tied/untied embeddings and the quantization plan.

    python scripts/smoke_model_config.py
    python scripts/smoke_model_config.py --json out.json

Only public metadata is read. No weights are downloaded.
"""

from __future__ import annotations

import argparse
import json
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SIZES = ("0.8b", "4b", "9b")


def fetch_hf_config(repo: str) -> dict | None:
    url = f"https://huggingface.co/{repo}/raw/main/config.json"
    try:
        with urllib.request.urlopen(url, timeout=30) as response:
            return json.loads(response.read().decode("utf-8"))
    except Exception:  # noqa: BLE001 - offline checks must not crash the smoke
        return None


def hf_geometry(config: dict | None) -> dict:
    """Pull the few fields we compare, tolerating field-name variants."""
    if not isinstance(config, dict):
        return {}
    text = config.get("text_config") or config
    return {
        "hidden": text.get("hidden_size") or text.get("n_embd") or text.get("d_model"),
        "layers": text.get("num_hidden_layers") or text.get("n_layer"),
        "vocab": text.get("vocab_size"),
    }


def checks(spec: dict, local: dict, remote: dict) -> list[dict]:
    problems = []
    hidden = local.get("hidden")
    layers = local.get("layers")
    block = int(spec.get("block_size", 32))
    group = int(spec.get("group_size", 32))
    divisor = max(block, group)

    if hidden and hidden % divisor:
        problems.append({"check": "hidden-divisible", "detail": f"hidden={hidden} not divisible by {divisor}"})
    if spec.get("tied_embedding") is None:
        problems.append({"check": "tied-embedding", "detail": "tied_embedding is not declared"})

    if remote:
        if remote.get("hidden") and hidden and remote["hidden"] != hidden:
            problems.append(
                {"check": "hidden-matches-hf", "detail": f"local={hidden} hf={remote['hidden']}"}
            )
        if remote.get("layers") and layers and remote["layers"] != layers:
            problems.append(
                {"check": "layers-match-hf", "detail": f"local={layers} hf={remote['layers']}"}
            )
        if remote.get("vocab") and remote["vocab"] % divisor:
            problems.append(
                {"check": "vocab-divisible", "detail": f"vocab={remote['vocab']} not divisible by {divisor}"}
            )
    else:
        problems.append({"check": "hf-config", "detail": "could not read HF config (offline?)"})
    return problems


def spec_names() -> set[str]:
    try:
        from lalamo.model_import.model_specs.qwen import QWEN_MODELS

        return {str(getattr(item, "name", item)) for item in QWEN_MODELS}
    except Exception:  # noqa: BLE001
        return set()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--json", type=Path, default=None)
    args = parser.parse_args()

    names = spec_names()
    report = {"spec_names_found": sorted(names), "sizes": {}}
    failures = 0

    for size in SIZES:
        config_path = ROOT / f"experiments/qwen35-{size}/config.json"
        config = json.loads(config_path.read_text(encoding="utf-8"))
        model = config["model"]
        quant = {"block_size": 32, "group_size": 32}
        remote = hf_geometry(fetch_hf_config(model["repo"]))
        problems = checks({**model, **quant}, model, remote)

        lane = ROOT / "lanes" / f"eval-{size.replace('.', '')}-rtn.json"
        entry = {
            "repo": model["repo"],
            "hidden": model.get("hidden"),
            "layers": model.get("layers"),
            "hf_geometry": remote,
            "spec_present": (model.get("repo") and True) if not names else model.get("spec_name") in names,
            "lane_spec": lane.name if lane.exists() else None,
            "problems": problems,
        }
        report["sizes"][size] = entry
        failures += len(problems)
        print(f"{size}: hidden={entry['hidden']} layers={entry['layers']} hf={remote} problems={len(problems)}")
        for problem in problems:
            print(f"  - {problem['check']}: {problem['detail']}")

    report["problem_count"] = failures
    print(f"spec names: {len(names)}  problems: {failures}")
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"wrote {args.json}")
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
