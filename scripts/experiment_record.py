#!/usr/bin/env python3
"""Append-only experiment records for quantization runs.

Every run writes one JSON line so failed or partially completed experiments are
still reproducible later.

Usage:
  python scripts/experiment_record.py append experiments.jsonl record.json
  python scripts/experiment_record.py validate experiments.jsonl
  python scripts/experiment_record.py show experiments.jsonl
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path

REQUIRED = (
    "experiment_id",
    "model_name",
    "model_revision",
    "source_commit",
    "quantization_config",
    "calibration_dataset",
    "calibration_seed",
    "damping",
    "rht_config",
    "yaqa_config",
    "target_layers",
    "status",
)

OPTIONAL_DEFAULTS = {
    "uzu_commit": None,
    "quantizer": "lalamo",
    "weight_error": None,
    "logits_metrics": None,
    "generation_metrics": None,
    "per_layer_metrics": None,
    "peak_memory_bytes": None,
    "elapsed_seconds": None,
    "failure_reason": None,
    "notes": None,
    "created_at": None,
}

VALID_STATUS = {"planned", "running", "succeeded", "failed", "aborted"}


def normalize(record: dict) -> dict:
    missing = [key for key in REQUIRED if key not in record]
    if missing:
        raise ValueError(f"missing required fields: {missing}")
    if record["status"] not in VALID_STATUS:
        raise ValueError(f"invalid status {record['status']!r}; expected one of {sorted(VALID_STATUS)}")
    normalized = {**OPTIONAL_DEFAULTS, **record}
    if normalized["created_at"] is None:
        normalized["created_at"] = dt.datetime.now(dt.timezone.utc).isoformat()
    return normalized


def append(path: Path, record: dict) -> None:
    normalized = normalize(record)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(normalized, ensure_ascii=False, sort_keys=True) + "\n")


def validate(path: Path) -> int:
    errors = 0
    seen: set[str] = set()
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = normalize(json.loads(line))
            except (json.JSONDecodeError, ValueError) as error:
                print(f"{path}:{line_number}: {error}")
                errors += 1
                continue
            experiment_id = record["experiment_id"]
            if experiment_id in seen:
                print(f"{path}:{line_number}: duplicate experiment_id {experiment_id!r}")
                errors += 1
            seen.add(experiment_id)
    print(f"{path}: {len(seen)} records, {errors} errors")
    return errors


def show(path: Path) -> None:
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            print(
                f"{record.get('experiment_id')}: status={record.get('status')} "
                f"model={record.get('model_name')} yaqa={record.get('yaqa_config')}"
            )


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    append_parser = subparsers.add_parser("append")
    append_parser.add_argument("records", type=Path)
    append_parser.add_argument("record", type=Path, help="JSON file with a single record")

    validate_parser = subparsers.add_parser("validate")
    validate_parser.add_argument("records", type=Path)

    show_parser = subparsers.add_parser("show")
    show_parser.add_argument("records", type=Path)

    args = parser.parse_args()
    if args.command == "append":
        append(args.records, json.loads(args.record.read_text()))
    elif args.command == "validate":
        sys.exit(1 if validate(args.records) else 0)
    elif args.command == "show":
        show(args.records)


if __name__ == "__main__":
    main()
